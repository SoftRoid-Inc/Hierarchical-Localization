from typing import Optional, List, Dict, Any
import multiprocessing
from pathlib import Path
import pycolmap
import shutil
import subprocess
import os
import os.path as osp

from detectorfreesfm.sfm_runner.utils.make_database import load_intrin_to_database
from loguru import logger
from natsort import natsorted
from .utils.chunking import split_image_chunks
from .utils.database import COLMAPDatabase
from .triangulation import (
    import_features, import_matches, estimation_and_geometric_verification,
    OutputCapture, NOT_EXPO_COLMAP_CFGS, COLMAP_PATH)

def create_empty_db(database_path: Path):
    if database_path.exists():
        logger.warning('The database already exists, deleting it.')
        database_path.unlink()
    logger.info('Creating an empty database...')
    db = COLMAPDatabase.connect(database_path)
    db.create_tables()
    db.commit()
    db.close()


def import_images(image_dir: Path,
                  database_path: Path,
                  camera_mode: pycolmap.CameraMode,
                  image_list: Optional[List[str]] = None,
                  options: Optional[Dict[str, Any]] = None):
    logger.info(f'Importing images into the database, camera mode is {camera_mode}...')
    if options is None:
        options = {}
    images = list(image_dir.iterdir())
    if len(images) == 0:
        raise IOError(f'No images found in {image_dir}.')
    with pycolmap.ostream():
        pycolmap.import_images(str(database_path), str(image_dir), camera_mode,
                               image_names=image_list or [],
                               options=options)


def get_image_ids(database_path: Path) -> Dict[str, int]:
    db = COLMAPDatabase.connect(database_path)
    images = {}
    for name, image_id in db.execute("SELECT name, image_id FROM images;"):
        images[name] = image_id
    db.close()
    return images

def _prior_mapper_options(colmap_configs):
    """位置プライア付きマッピング用の IncrementalPipelineOptions を構築する。

    プライア有効時は CLI 設定の有無に関わらず pycolmap パスを使う
    （CLI の mapper は pose_priors を読まないため）。colmap_mapper_cfgs の
    チューニング値は setattr で opts.mapper へ引き継ぐ。
    """
    opts = pycolmap.IncrementalPipelineOptions(
        ba_refine_focal_length=not colmap_configs['no_refine_intrinsics'],
        ba_refine_extra_params=not colmap_configs['no_refine_intrinsics'],
        ba_refine_principal_point=not colmap_configs['no_refine_intrinsics'],
        num_threads=min(multiprocessing.cpu_count(),
                        colmap_configs.get('n_threads', 16)))
    opts.use_prior_position = True
    opts.use_robust_loss_on_prior_position = True
    if 'min_model_size' in colmap_configs:
        opts.min_model_size = colmap_configs['min_model_size']
    for key, value in (colmap_configs.get('colmap_mapper_cfgs') or {}).items():
        target = opts.mapper if hasattr(opts.mapper, key) else (
            opts if hasattr(opts, key) else None)
        if target is None:
            logger.warning(f"colmap_mapper_cfgs.{key} は pycolmap の "
                           "IncrementalPipelineOptions に無いため無視します")
            continue
        setattr(target, key, value)
    return opts


def _build_mapper_cmd(database_path, image_dir, output_path, colmap_configs,
                      image_list_path: Optional[Path] = None):
    cmd = [COLMAP_PATH, "mapper"]
    cmd += ["--image_path", str(image_dir)]
    cmd += ["--database_path", str(database_path)]
    cmd += ["--output_path", str(output_path)]
    if image_list_path is not None:
        # 登録対象を制限するのは Mapper 配下のオプション（トップレベルの
        # --image_list_path は automatic_reconstructor 専用で mapper には無い）
        cmd += ["--Mapper.image_list_path", str(image_list_path)]
    if colmap_configs is not None and "min_model_size" in colmap_configs:
        cmd += ["--Mapper.min_model_size", str(colmap_configs["min_model_size"])]
    cmd += ["--Mapper.num_threads", str(min(multiprocessing.cpu_count(), colmap_configs['n_threads'] if 'n_threads' in colmap_configs else 16))]

    if colmap_configs['use_pba']:
        logger.warning("PBA (--Mapper.ba_global_use_pba) is not supported by stock COLMAP 4.1.0; ignoring use_pba.")

    if colmap_configs.get('ba_backend') == 'CASPAR':
        # GPU BA バックエンド（COLMAP を -DCASPAR_ENABLED=ON でビルドした
        # 場合のみ有効）。CASPAR は glog の WARNING をフレーム毎に大量に
        # 吐き /tmp のファイルログが GB 級になるため stderr のみへ抑制する
        cmd += [
            "--Mapper.ba_local_backend", "CASPAR",
            "--Mapper.ba_global_backend", "CASPAR",
            "--log_target", "stderr",
        ]

    if colmap_configs['colmap_mapper_cfgs'] is not None:
        for config_name, value in colmap_configs["colmap_mapper_cfgs"].items():
            if config_name in NOT_EXPO_COLMAP_CFGS:
                cmd += [NOT_EXPO_COLMAP_CFGS[config_name], str(value)]

    if (
        colmap_configs is not None
        and colmap_configs["no_refine_intrinsics"] is True
    ):
        cmd += [
            "--Mapper.ba_refine_focal_length",
            "0",
            "--Mapper.ba_refine_extra_params",
            "0",
        ]
    return cmd


def _run_mapper_logged(cmd, models_path):
    """mapper を実行し stdout/stderr を models_path/output.txt へ追記する。

    失敗しても raise しない（呼び出し側はモデル 0 個の既存フォールバックで
    処理を継続できる）。"""
    logger.info(' '.join(cmd))
    colmap_res = subprocess.run(cmd, capture_output=True)
    with open(osp.join(models_path, "output.txt"), "a") as f:
        f.write(colmap_res.stdout.decode())
        f.write(colmap_res.stderr.decode())
    if colmap_res.returncode != 0:
        logger.error(f"COLMAP mapper failed with exit code {colmap_res.returncode}:\n"
                     f"{colmap_res.stderr.decode()}")


def _run_chunked_mapper(models_path, database_path, image_dir, image_list,
                        chunk_size, colmap_configs):
    """画像リストを時系列チャンクへ等分し、チャンクごとに独立した mapper を実行する。

    共有 database.db に対し --Mapper.image_list_path で登録対象を制限する。チャンク間の
    モデル統合は行わず、生成モデルを models_path 直下の連番ディレクトリへ平坦化
    する（チャンク間の座標合わせは下流のフラグメント単位 IMU/PDR アラインに
    任せる）。1 チャンクの失敗は残りのチャンクの実行を妨げない。
    """
    chunks = split_image_chunks(natsorted(image_list), chunk_size)
    logger.info(f"Chunked mapper: {len(image_list)} images > mapper_chunk_size={chunk_size}"
                f" -> {len(chunks)} chunks")

    chunk_model_records = []
    next_id = 0
    for k, chunk in enumerate(chunks):
        logger.info(f"[chunk {k}/{len(chunks)}] {len(chunk)} images: {chunk[0]} .. {chunk[-1]}")
        chunk_dir = models_path / f"chunk_{k}"
        chunk_dir.mkdir(exist_ok=True, parents=True)
        image_list_path = chunk_dir / "image_list.txt"
        image_list_path.write_text("\n".join(chunk) + "\n")

        cmd = _build_mapper_cmd(database_path, image_dir, chunk_dir, colmap_configs,
                                image_list_path=image_list_path)
        _run_mapper_logged(cmd, models_path)

        model_dirs = sorted(
            (p for p in chunk_dir.iterdir() if p.is_dir() and p.name.isdigit()),
            key=lambda p: int(p.name))
        for model_dir in model_dirs:
            shutil.move(str(model_dir), str(models_path / str(next_id)))
            chunk_model_records.append((next_id, k, len(chunk), chunk[0], chunk[-1]))
            logger.info(f"model {next_id} <- chunk {k} ({len(chunk)} images)")
            next_id += 1
        # chunk_* を残すと後段の mv で colmap_coarse へ漏れ、モデルディレクトリは
        # 整数名のみという下流 (_colmap_to_pandas) の前提が壊れる
        shutil.rmtree(chunk_dir)

    with open(models_path / "mapper_chunks.txt", "w") as f:
        f.write("model_id\tchunk\tchunk_num_images\tchunk_first_image\tchunk_last_image\n")
        for record in chunk_model_records:
            f.write("\t".join(str(v) for v in record) + "\n")


def run_reconstruction(sfm_dir, database_path, image_dir, colmap_configs, verbose=False,
                       image_list: Optional[List[str]] = None):
    models_path = sfm_dir / 'models'

    models_path.mkdir(exist_ok=True, parents=True)
    if colmap_configs.get('use_prior_position'):
        logger.info("Use PyCOLMAP with pose priors for reconstruction...")
        mapper_options = _prior_mapper_options(colmap_configs)
        with OutputCapture(verbose):
            with pycolmap.ostream():
                logger.info(mapper_options.summary())
                reconstructions = pycolmap.incremental_mapping(
                    database_path, image_dir, models_path,
                    mapper_options,)
    elif colmap_configs['colmap_mapper_cfgs'] is None:
        logger.info(f"Use PyCOLMAP for reconstruction...")
        if colmap_configs["use_pba"]:
            logger.warning("PBA is not supported in stock pycolmap 4.1.0 IncrementalPipelineOptions; ignoring use_pba on the pycolmap path.")
        mapper_options = pycolmap.IncrementalPipelineOptions(ba_refine_focal_length=not colmap_configs['no_refine_intrinsics'],
                                                           ba_refine_extra_params=not colmap_configs['no_refine_intrinsics'],
                                                           ba_refine_principal_point=not colmap_configs['no_refine_intrinsics'],
                                                           num_threads=min(multiprocessing.cpu_count(),
                                                                           colmap_configs['n_threads'] if 'n_threads' in colmap_configs else 16)
        )
        logger.info('Running 3D reconstruction...')
        with OutputCapture(verbose):
            with pycolmap.ostream():
                logger.info(f"use: {min(multiprocessing.cpu_count(), colmap_configs['n_threads'] if 'n_threads' in colmap_configs else 16)} cpus")
                logger.info(mapper_options.summary())
                reconstructions = pycolmap.incremental_mapping(
                    database_path, image_dir, models_path,
                    mapper_options,)
    else:
        logger.info(f"Use command line COLMAP for reconstruction...")
        chunk_size = colmap_configs.get('mapper_chunk_size') or 0
        supplement_prefix = colmap_configs.get('supplement_data_prefix') or ''
        has_supplement = bool(supplement_prefix) and any(
            name.startswith(supplement_prefix) for name in (image_list or []))
        if chunk_size > 0 and not image_list:
            logger.warning("mapper_chunk_size が指定されていますが image_list が"
                           "渡されていないため分割せずに実行します")
        if chunk_size > 0 and has_supplement:
            # 時系列分割すると images_ref/ の補完フレームが 1 チャンクへ固まり
            # 新旧データの共登録が成立しなくなるため、調整 run は分割対象外
            logger.warning("補完データ (images_ref) を含む調整 run のため "
                           "mapper のチャンク分割をスキップします")
        if chunk_size > 0 and image_list and len(image_list) > chunk_size and not has_supplement:
            _run_chunked_mapper(models_path, database_path, image_dir,
                                image_list, chunk_size, colmap_configs)
        else:
            cmd = _build_mapper_cmd(database_path, image_dir, models_path, colmap_configs)
            _run_mapper_logged(cmd, models_path)

        # モデルディレクトリ名を id に使う（辞書順 enumerate だとモデルが
        # 10 個を超えた際に "10" < "2" で id とディレクトリ名がずれ、
        # 後段のマージループが別モデルを読んでしまう）
        reconstructions = {
            int(model_path.name): pycolmap.Reconstruction(model_path)
            for model_path in models_path.iterdir()
            if model_path.is_dir() and model_path.name.isdigit()
        }

    if len(reconstructions) == 0:
        logger.error('Could not reconstruct any model!')
        os.system(f"mv {models_path}/* {sfm_dir}")
        os.system(f"rm -rf {models_path}")
        return None
    logger.info(f'Reconstructed {len(reconstructions)} model(s).')

    largest_index = None
    largest_num_images = 0
    for index, rec in reconstructions.items():
        num_images = rec.num_reg_images()
        if num_images > largest_num_images:
            largest_index = index
            largest_num_images = num_images
    assert largest_index is not None


    sorted_models = sorted(reconstructions.items(), key=lambda x: x[1].num_reg_images(), reverse=True)
    logger.info(f"Models sorted by number of registered images: {sorted_models[0][0]} with {sorted_models[0][1].num_reg_images()} images.")
    logger.info(f"model sizes {[(f'{idx}: {rec.num_reg_images()}') for idx, rec in sorted_models]}")
    delete_models = []
    for k in reversed(range(1, len(sorted_models))):
        small_idx, _ = sorted_models[k]
        small_rec = pycolmap.Reconstruction(models_path/str(small_idx))

        for i in reversed(range(0, k)):
            big_idx, _ = sorted_models[i]
            big_rec = pycolmap.Reconstruction(models_path/str(big_idx))

            common_images = big_rec.find_common_reg_image_ids(small_rec)
            logger.info(f'Model #{big_idx} and #{small_idx} have {len(common_images)} common images.')
            if len(common_images) > 2:
                if merge_reconstruction(models_path/str(big_idx), models_path/str(small_idx), models_path/str(big_idx), models_path, colmap_configs):
                    delete_models.append(small_idx)

            # logger.info(f'Model #{index} has {rec.num_reg_images()} images.')
    for idx in delete_models:
        os.system(f"rm -rf {models_path}/{idx}")
        logger.info(f"Deleted model #{idx}.")
    logger.info("Merge reconstruction models with common images...")
    logger.info(f"省略できたモデル: {delete_models}")
    logger.info(f"biggest new model:{sorted_models[0][0]} images: {pycolmap.Reconstruction(models_path/str(sorted_models[0][0])).num_reg_images()}")

    if colmap_configs is not None and colmap_configs.get('enable_model_clustering'):
        _cluster_and_refine_models(models_path, colmap_configs)

    os.system(f"mv {models_path}/* {sfm_dir}")
    os.system(f"rm -rf {models_path}")
    return reconstructions[largest_index]


def _ba_backend_args(colmap_configs):
    """bundle_adjuster 用のバックエンド引数。CASPAR は glog の WARNING を
    /tmp にファイルとして大量に吐くため stderr のみへ抑制する（mapper と同様）。"""
    if colmap_configs is not None and colmap_configs.get('ba_backend') == 'CASPAR':
        return ["--BundleAdjustment.backend", "CASPAR", "--log_target", "stderr"]
    return []


def _run_colmap_logged(cmd, log_path):
    res = subprocess.run(cmd, capture_output=True)
    with open(osp.join(log_path, "output.txt"), "a") as f:
        f.write(res.stdout.decode())
        f.write(res.stderr.decode())
    if res.returncode != 0:
        logger.error(f"colmap command failed (rc={res.returncode}): {' '.join(map(str, cmd))}")
    return res.returncode == 0


def _cluster_and_refine_models(models_path, colmap_configs):
    """マージ後のモデル群を共可視クラスタリングで検疫し、クラスターごとに再最適化する。

    model_clusterer は「共有 3D 点数を辺重みとする共可視グラフ」を適応しきい値
    （median - MAD）+ Union-Find で分割し、弱くしか繋がっていないフレーム塊
    （誤マッチによる貼り合わせ・軌跡端の孤立区間）を独立モデルに切り出す。
    小さすぎるクラスター（min_num_reg_frames 未満）のフレームは破棄される。

    分割で観測が抜けた 3D 点が退化して残るため、point_filtering で除去してから
    bundle_adjuster で縮小後の問題を最適化し直す。

    処理後は全モデルディレクトリを 0..N-1 に連番リネームする。下流
    （get_best_colmap_index / sfm_project の _colmap_to_pandas）は
    「colmap_coarse 直下の全サブディレクトリ = int 名のモデル」を前提とするため、
    余計なディレクトリを残してはならない。
    """
    cluster_tmp = models_path.parent / "cluster_tmp"
    shutil.rmtree(cluster_tmp, ignore_errors=True)

    # model_clusterer / point_filtering / bundle_adjuster の出力は mapper と違い
    # project.ini を書かないが、下流（init_sfm_run_cloud_data のモデル
    # アップロード等）はモデル dir に project.ini がある前提。元モデルのものを
    # 控えておき、最後に欠けている dir へ複製する
    project_ini_bytes = None

    final_models = []
    for model_dir in sorted([d for d in models_path.iterdir() if d.is_dir()]):
        if project_ini_bytes is None and (model_dir / "project.ini").exists():
            project_ini_bytes = (model_dir / "project.ini").read_bytes()
        out_dir = cluster_tmp / model_dir.name
        out_dir.mkdir(parents=True)
        cmd = [COLMAP_PATH, "model_clusterer"]
        cmd += ["--input_path", str(model_dir)]
        cmd += ["--output_path", str(out_dir)]
        cmd += ["--ReconstructionClusterer.min_num_reg_frames",
                str(colmap_configs.get('model_clustering_min_num_reg_frames', 3))]
        cmd += ["--ReconstructionClusterer.min_edge_weight_threshold",
                str(colmap_configs.get('model_clustering_min_edge_weight', 10.0))]
        ok = _run_colmap_logged(cmd, models_path)
        clusters = sorted([d for d in out_dir.iterdir() if d.is_dir()]) if ok else []
        if not clusters:
            # クラスタリング失敗 / 全クラスターが最小フレーム数未満 → 元モデルを維持
            logger.warning(f"model_clusterer produced no clusters for {model_dir.name}; keeping original model")
            final_models.append(model_dir)
            continue
        if len(clusters) > 1:
            logger.info(f"Model {model_dir.name} split into {len(clusters)} clusters")
        shutil.rmtree(model_dir)
        final_models.extend(clusters)

    # 衝突しないよう一旦テンポラリー名に退避してから 0..N-1 に連番リネーム
    staged = []
    for i, src in enumerate(final_models):
        dst = models_path / f"__staged_{i}"
        shutil.move(str(src), str(dst))
        staged.append(dst)
    for i, src in enumerate(staged):
        dst = models_path / str(i)
        src.rename(dst)
        if project_ini_bytes is not None and not (dst / "project.ini").exists():
            (dst / "project.ini").write_bytes(project_ini_bytes)
    shutil.rmtree(cluster_tmp, ignore_errors=True)

    ba_args = _ba_backend_args(colmap_configs)
    for i in range(len(staged)):
        model_dir = models_path / str(i)
        _run_colmap_logged(
            [COLMAP_PATH, "point_filtering",
             "--input_path", str(model_dir), "--output_path", str(model_dir)],
            models_path)
        _run_colmap_logged(
            [COLMAP_PATH, "bundle_adjuster",
             "--input_path", str(model_dir), "--output_path", str(model_dir)] + ba_args,
            models_path)
    logger.info(f"Model clustering done: {len(staged)} final model(s)")


# Reconstruction::Write が書き出すモデル実体。project.ini は COLMAP が書かないため
# 置き換え対象から外す（下流がモデル dir に project.ini がある前提のため）
_MODEL_FILES = ("cameras.bin", "images.bin", "points3D.bin", "rigs.bin", "frames.bin")


def merge_reconstruction(input_path1, input_path2, output_path, log_path, colmap_configs=None):
    """2 つのモデルを統合し、成功したときだけ output_path を置き換える。

    colmap model_merger は input2 側へ統合した結果を、成功・失敗に関わらず
    output_path へ書き出して EXIT_SUCCESS を返す（失敗時の中身は input2 のまま）。
    呼び出し側は output_path に大きい方のモデルを指定するので、直接書かせると
    統合失敗時に大きいモデルが小さいモデルで上書きされて消える。さらに COLMAP は
    成否を stderr へ吐くため、stdout だけを見る判定では失敗を検出できない。
    一時ディレクトリーへ出力し、stdout と stderr の両方で成功を確認してから反映する。
    """
    output_path = Path(output_path)
    merge_tmp = Path(log_path).parent / "merge_tmp"
    shutil.rmtree(merge_tmp, ignore_errors=True)
    merge_tmp.mkdir(parents=True)
    try:
        cmd = [COLMAP_PATH, "model_merger",
               "--input_path1", str(input_path1),
               "--input_path2", str(input_path2),
               "--output_path", str(merge_tmp)]
        colmap_res = subprocess.run(cmd, capture_output=True)
        merge_output = colmap_res.stdout.decode() + colmap_res.stderr.decode()
        with open(osp.join(log_path, "output.txt"), "a") as f:
            f.write(merge_output)

        if colmap_res.returncode != 0 or "Merge succeeded" not in merge_output:
            logger.error(f"Merge failed ({input_path1} + {input_path2}); "
                         "元のモデルを維持します")
            return False

        for name in _MODEL_FILES:
            src = merge_tmp / name
            if src.exists():
                shutil.copy2(src, output_path / name)
    finally:
        shutil.rmtree(merge_tmp, ignore_errors=True)

    cmd2 = [COLMAP_PATH, "bundle_adjuster"]
    cmd2 += ["--input_path", str(output_path)]
    cmd2 += ["--output_path", str(output_path)]
    cmd2 += _ba_backend_args(colmap_configs)
    _run_colmap_logged(cmd2, log_path)
    return True

def main(sfm_dir, image_dir, pairs, features, matches, prior_intrin,
         colmap_configs: Dict[str, Any],
         camera_mode=pycolmap.CameraMode.AUTO, verbose=False,
         skip_geometric_verification=False, min_match_score=None,
         image_list: Optional[List[str]] = None,
         prior_pose_path=None,
    ):
    assert features.exists(), features
    assert pairs.exists(), pairs
    assert matches.exists(), matches

    sfm_dir.mkdir(parents=True, exist_ok=True)
    database = sfm_dir / 'database.db'

    create_empty_db(database)
    if colmap_configs['use_pba'] or colmap_configs["ImageReader_camera_mode"] == 'per_image':
        camera_mode = pycolmap.CameraMode.PER_IMAGE
    elif colmap_configs["ImageReader_camera_mode"] == 'single_camera':
        camera_mode = pycolmap.CameraMode.SINGLE

    camera_model = "SIMPLE_RADIAL" if 'ImageReader_camera_model' not in colmap_configs else colmap_configs['ImageReader_camera_model']
    if colmap_configs['use_pba']:
        camera_model = "SIMPLE_RADIAL"

    img_import_opts = pycolmap.ImageReaderOptions(camera_model=camera_model)
    import_images(image_dir, database, camera_mode, image_list, img_import_opts)

    if prior_intrin is not None:
        logger.info(f"Load prior intrin into db...")
        if colmap_configs['use_pba']:
            logger.warning('Currently PBA not support fix (known) intrin and optimize poses and point clouds. Moreover, the loaded PINHOLE camera model is not supported.\n PBA is disabled automatically.')
            colmap_configs['use_pba'] = False
        if colmap_configs['ImageReader_camera_mode'] == "per_image":
            logger.warning('Currently PBA !')
            colmap_configs['use_pba'] = True
        load_intrin_to_database(database, prior_intrin, colmap_configs)

    image_ids = get_image_ids(database)
    import_features(image_ids, database, features, verbose=verbose, replace_slash=True)
    import_matches(image_ids, database, pairs, matches,
                   min_match_score, skip_geometric_verification, verbose=verbose, replace_slash=True)
    if not skip_geometric_verification:
        max_error = 4.0 if 'geometry_verify_thr' not in colmap_configs else colmap_configs['geometry_verify_thr']
        estimation_and_geometric_verification(database, pairs, verbose, max_error=max_error)

    if prior_pose_path is not None:
        from .utils.pose_priors import inject_pose_priors_from_csv
        sigma = colmap_configs.get('prior_position_std_m') or 3.0
        n_prior, n_skip = inject_pose_priors_from_csv(
            database, prior_pose_path, sigma=sigma)
        logger.info(f"pose_priors 注入: {n_prior} 件 "
                    f"(対応無し {n_skip} 件, σ={sigma} m)")
        # 1 件も注入できなければ従来マッパーへフォールバック
        colmap_configs['use_prior_position'] = n_prior > 0

    reconstruction = run_reconstruction(sfm_dir, database, image_dir, colmap_configs=colmap_configs,
                                        image_list=image_list)
    if reconstruction is not None and verbose:
        logger.info(f'Reconstruction statistics:\n{reconstruction.summary()}'
                    + f'\n\tnum_input_images = {len(image_ids)}')
    return reconstruction
