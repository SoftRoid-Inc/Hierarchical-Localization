from typing import Optional, List, Dict, Any
import multiprocessing
from pathlib import Path
import pycolmap
import subprocess
import os
import os.path as osp
import numpy as np
from PIL import Image
from detectorfreesfm.sfm_runner.utils.make_database import load_intrin_to_database
from . import logger
from .utils.database import COLMAPDatabase
from .triangulation import (
    import_features, import_matches, estimation_and_geometric_verification,
    OutputCapture, NOT_EXPO_COLMAP_CFGS)
COLMAP_PATH = os.environ.get("COLMAP_PATH", 'colmap') # 'colmap is default value

def create_empty_db(database_path: Path):
    if database_path.exists():
        logger.warning('The database already exists, deleting it.')
        database_path.unlink()
    logger.info('Creating an empty database...')
    db = COLMAPDatabase.connect(database_path)
    db.create_tables()
    db.commit()
    db.close()


def setup_two_camera_database(image_dir: Path,
                             database_path: Path,
                             camera_mode: pycolmap.CameraMode,
                             image_list: Optional[List[str]] = None,
                             options: Optional[Dict[str, Any]] = None,
                             colmap_configs: Optional[Dict[str, Any]] = None):
    """
    2つのカメラ（iPhone: PINHOLE, Sphere: SPHERE）の事前知識をデータベースに組み込む
    カメラIDも事前に固定する
    """
    logger.info('Setting up database with two-camera prior knowledge and fixed camera IDs...')
    if options is None:
        options = {}
    
    images = list(image_dir.iterdir())
    if len(images) == 0:
        raise IOError(f'No images found in {image_dir}.')
    
    # 画像をカメラタイプ別に分類
    perspective_images = []
    sphere_images = []
    
    for img_path in images:
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.tiff', '.tif']:
            img_name = img_path.name.lower()
            if 'iphone' in img_name:
                perspective_images.append(img_path.name)
            else:
                sphere_images.append(img_path.name)
    
    logger.info(f'Found {len(perspective_images)} perspective images (iPhone) and {len(sphere_images)} sphere images')
    
    # データベースに接続
    db = COLMAPDatabase.connect(database_path)
    
    # 既存のカメラと画像を削除
    db.execute("DELETE FROM images")
    db.execute("DELETE FROM cameras")
    
    # 固定カメラIDの定義（設定から読み取り）
    if colmap_configs and colmap_configs.get('fixed_camera_ids', False):
        PERSPECTIVE_CAMERA_ID = colmap_configs.get('perspective_camera_id', 1)
        SPHERE_CAMERA_ID = colmap_configs.get('sphere_camera_id', 2)
        logger.info(f'Using fixed camera IDs: Perspective={PERSPECTIVE_CAMERA_ID}, Sphere={SPHERE_CAMERA_ID}')
    else:
        PERSPECTIVE_CAMERA_ID = 1
        SPHERE_CAMERA_ID = 2
        logger.info(f'Using default camera IDs: Perspective={PERSPECTIVE_CAMERA_ID}, Sphere={SPHERE_CAMERA_ID}')
    
    # カメラ1: iPhone (PINHOLE) - 固定ID: 1
    if perspective_images:
        # 最初のiPhone画像から解像度を取得
        first_img = image_dir / perspective_images[0]
        with Image.open(first_img) as img:
            width, height = img.size
        
        # PINHOLEカメラを追加 (fx, fy, cx, cy)
        camera1_id = db.add_camera(
            model=1,  # PINHOLE
            width=width,
            height=height,
            params=np.array([717.0655, 717.0655, 385.20782, 512.9991]),  # 仮のパラメータ TODO
            camera_id=PERSPECTIVE_CAMERA_ID  # 固定ID: 1
        )
        logger.info(f'Added PINHOLE camera (ID: {camera1_id}) for iPhone images')
    
    # カメラ2: Sphere (SPHERE) - 固定ID: 2
    if sphere_images:
        # 最初のsphere画像から解像度を取得
        first_img = image_dir / sphere_images[0]
        with Image.open(first_img) as img:
            width, height = img.size
        
        # SPHEREカメラを追加 (f, cx, cy)
        camera2_id = db.add_camera(
            model=11,  # SPHERE (COLMAPのモデルID)
            width=width,
            height=height,
            params=np.array([1.0, width/2, height/2]),  # SPHEREのパラメータ: f=1.0, cx=width/2, cy=height/2
            camera_id=SPHERE_CAMERA_ID  # 固定ID: 2
        )
        logger.info(f'Added SPHERE camera (ID: {camera2_id}) for sphere images')
    
    db.commit()
    db.close()
    
    # 画像をインポート（固定カメラIDを使用）
    if perspective_images:
        logger.info(f'Importing perspective images with fixed camera ID {PERSPECTIVE_CAMERA_ID}...')
        perspective_opts = pycolmap.ImageReaderOptions(
            camera_model="PINHOLE", # this is ignored when existing_camera_id is specified
            existing_camera_id=PERSPECTIVE_CAMERA_ID  # 既存のカメラIDを指定
        )
        logger.info(f'Perspective options: existing_camera_id={perspective_opts.existing_camera_id}')
        with pycolmap.ostream():
            pycolmap.import_images(database_path, image_dir, pycolmap.CameraMode.SINGLE,  # SINGLEモードを強制
                                   image_list=perspective_images,
                                   options=perspective_opts)
    
    if sphere_images:
        logger.info(f'Importing sphere images with fixed camera ID {SPHERE_CAMERA_ID}...')
        sphere_opts = pycolmap.ImageReaderOptions(
            camera_model="SPHERE", # this is ignored when existing_camera_id is specified
            existing_camera_id=SPHERE_CAMERA_ID  # 既存のカメラIDを指定
        )
        logger.info(f'Sphere options: existing_camera_id={sphere_opts.existing_camera_id}')
        with pycolmap.ostream():
            pycolmap.import_images(database_path, image_dir, pycolmap.CameraMode.SINGLE,  # SINGLEモードを強制
                                   image_list=sphere_images,
                                   options=sphere_opts)
    
    # デバッグ: データベースのカメラ数を確認
    db = COLMAPDatabase.connect(database_path)
    camera_count = db.execute("SELECT COUNT(*) FROM cameras").fetchone()[0]
    image_count = db.execute("SELECT COUNT(*) FROM images").fetchone()[0]
    logger.info(f'Database status: {camera_count} cameras, {image_count} images')
    db.close()


def import_images_with_camera_detection(image_dir: Path,
                                       database_path: Path,
                                       camera_mode: pycolmap.CameraMode,
                                       image_list: Optional[List[str]] = None,
                                       options: Optional[Dict[str, Any]] = None):
    """
    ファイル名に基づいてカメラタイプを判別し、適切なカメラモデルで画像をインポート
    - ファイル名に'iphone'が含まれる → perspective (PINHOLE)
    - それ以外 → sphere (SPHERE)
    """
    logger.info(f'Importing images with camera detection, camera mode is {camera_mode}...')
    if options is None:
        options = {}
    
    images = list(image_dir.iterdir())
    if len(images) == 0:
        raise IOError(f'No images found in {image_dir}.')
    
    # 画像をカメラタイプ別に分類
    perspective_images = []
    sphere_images = []
    
    for img_path in images:
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.tiff', '.tif']:
            img_name = img_path.name.lower()
            if 'iphone' in img_name:
                perspective_images.append(img_path.name)  # ファイル名のみ
            else:
                sphere_images.append(img_path.name)  # ファイル名のみ
    
    logger.info(f'Found {len(perspective_images)} perspective images (iPhone) and {len(sphere_images)} sphere images')
    
    # per_imageモードの場合は、カメラモデルを指定せずにインポート
    if camera_mode == pycolmap.CameraMode.PER_IMAGE:
        logger.info('Using PER_IMAGE mode - importing all images without specific camera model...')
        all_images = [img.name for img in images if img.suffix.lower() in ['.jpg', '.jpeg', '.png', '.tiff', '.tif']]
        with pycolmap.ostream():
            pycolmap.import_images(database_path, image_dir, camera_mode,
                                   image_list=all_images,
                                   options=pycolmap.ImageReaderOptions())
    else:
        # データベースに接続
        db = COLMAPDatabase.connect(database_path)
        
        # perspective画像をインポート
        if perspective_images:
            logger.info('Importing perspective images with PINHOLE camera model...')
            perspective_opts = pycolmap.ImageReaderOptions(camera_model="PINHOLE")
            with pycolmap.ostream():
                pycolmap.import_images(database_path, image_dir, camera_mode,
                                       image_list=perspective_images,
                                       options=perspective_opts)
        
        # sphere画像をインポート
        if sphere_images:
            logger.info('Importing sphere images with SPHERE camera model...')
            sphere_opts = pycolmap.ImageReaderOptions(camera_model="SPHERE")
            with pycolmap.ostream():
                pycolmap.import_images(database_path, image_dir, camera_mode,
                                       image_list=sphere_images,
                                       options=sphere_opts)
        
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
        pycolmap.import_images(database_path, image_dir, camera_mode,
                               image_list=image_list or [],
                               options=options)


def get_image_ids(database_path: Path) -> Dict[str, int]:
    db = COLMAPDatabase.connect(database_path)
    images = {}
    for name, image_id in db.execute("SELECT name, image_id FROM images;"):
        images[name] = image_id
    db.close()
    return images

def run_reconstruction(sfm_dir, database_path, image_dir, colmap_configs):
    models_path = sfm_dir / 'models'

    models_path.mkdir(exist_ok=True, parents=True)
    if colmap_configs['colmap_mapper_cfgs'] is None:
        logger.info(f"Use PyCOLMAP for reconstruction...")
        if colmap_configs["use_pba"]:
            mapper_options = pycolmap.IncrementalMapperOptions(ba_global_use_pba=colmap_configs['use_pba'], ba_refine_focal_length=not colmap_configs['no_refine_intrinsics'], ba_refine_extra_params=not colmap_configs['no_refine_intrinsics'], num_threads=min(multiprocessing.cpu_count(), colmap_configs['n_threads'] if 'n_threads' in colmap_configs else 16))
        else:
            mapper_options = pycolmap.IncrementalMapperOptions(ba_refine_focal_length=not colmap_configs['no_refine_intrinsics'],
                                                               ba_refine_extra_params=not colmap_configs['no_refine_intrinsics'],
                                                               ba_refine_principal_point=not colmap_configs['no_refine_intrinsics'],
                                                               sphere_camera=colmap_configs['ImageReader_camera_model'] == 'SPHERE',
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
        cmd = [COLMAP_PATH, "mapper"]
        cmd += ["--image_path", str(image_dir)]
        cmd += ["--database_path", str(database_path)]
        cmd += ["--output_path", str(models_path)]
        if colmap_configs is not None and "min_model_size" in colmap_configs:
            cmd += ["--Mapper.min_model_size", str(colmap_configs["min_model_size"])]
        cmd += ["--Mapper.num_threads", str(min(multiprocessing.cpu_count(), colmap_configs['n_threads'] if 'n_threads' in colmap_configs else 16))]

        if colmap_configs['use_pba']:
            cmd += ["--Mapper.ba_global_use_pba", '1']

        if colmap_configs["ImageReader_camera_model"] == 'SPHERE':
            cmd += ["--Mapper.sphere_camera", '1']

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

        logger.info(' '.join(cmd))
        colmap_res = subprocess.run(cmd, capture_output=True)
        with open(osp.join(models_path, "output.txt"), "w") as f:
            f.write(colmap_res.stdout.decode())

        reconstructions = {}
        for id, model_path in enumerate(sorted(models_path.glob('*'))):
            if model_path.is_dir():
                reconstructions[id] = pycolmap.Reconstruction(model_path)

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
                if merge_reconstruction(models_path/str(big_idx), models_path/str(small_idx), models_path/str(big_idx), models_path):
                    delete_models.append(small_idx)

            # logger.info(f'Model #{index} has {rec.num_reg_images()} images.')
    for idx in delete_models:
        os.system(f"rm -rf {models_path}/{idx}")
        logger.info(f"Deleted model #{idx}.")
    logger.info("Merge reconstruction models with common images...")
    logger.info(f"省略できたモデル: {delete_models}")
    logger.info(f"biggest new model:{sorted_models[0][0]} images: {pycolmap.Reconstruction(models_path/str(sorted_models[0][0])).num_reg_images()}")

    os.system(f"mv {models_path}/* {sfm_dir}")
    os.system(f"rm -rf {models_path}")
    return reconstructions[largest_index]


def merge_reconstruction(input_path1, input_path2, output_path, log_path):
    cmd = [COLMAP_PATH, "model_merger"]
    cmd += ["--input_path1", str(input_path1)]
    cmd += ["--input_path2", str(input_path2)]
    cmd += ["--output_path", str(output_path)]

    cmd2 = [COLMAP_PATH, "bundle_adjuster"]
    cmd2 += ["--input_path", str(output_path)]
    cmd2 += ["--output_path", str(output_path)]

    colmap_res = subprocess.run(cmd, capture_output=True)
    # logger.info(' '.join(cmd))
    merge_output = colmap_res.stdout.decode()
    with open(osp.join(log_path, "output.txt"), "a") as f:
        f.write(merge_output)

    if "Merge failed" in merge_output:
        logger.error(f"Merge failed: {merge_output}")
        return False
    colmap_res = subprocess.run(cmd2, capture_output=True)
    # logger.info(' '.join(cmd2))
    with open(osp.join(log_path, "output.txt"), "a") as f:
        f.write(colmap_res.stdout.decode())
    return True

def main(sfm_dir, image_dir, pairs, features, matches, prior_intrin,
         colmap_configs: Dict[str, Any],
         camera_mode=pycolmap.CameraMode.AUTO, verbose=False,
         skip_geometric_verification=False, min_match_score=None,
         image_list: Optional[List[str]] = None,
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

    # カメラ設定方法を選択
    if colmap_configs.get('use_two_camera_prior', False):
        # 2つのカメラの事前知識を使用
        setup_two_camera_database(image_dir, database, camera_mode, image_list, None, colmap_configs)
    elif colmap_configs.get('use_filename_camera_detection', False):
        # ファイル名に基づいてカメラタイプを判別し、適切なカメラモデルでインポート
        import_images_with_camera_detection(image_dir, database, camera_mode, image_list)
    else:
        # 従来の方法
        if camera_mode == pycolmap.CameraMode.PER_IMAGE:
            # per_imageモードでは、カメラモデルを指定しない（自動推定）
            logger.info('Using PER_IMAGE mode - importing images with automatic camera model detection...')
            img_import_opts = pycolmap.ImageReaderOptions()
        else:
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
    import_features(image_ids, database, features, verbose=verbose)
    import_matches(image_ids, database, pairs, matches,
                   min_match_score, skip_geometric_verification, verbose=verbose)
    if not skip_geometric_verification:
        max_error = 4.0 if 'geometry_verify_thr' not in colmap_configs else colmap_configs['geometry_verify_thr']
        estimation_and_geometric_verification(database, pairs, verbose, max_error=max_error)

    reconstruction = run_reconstruction(sfm_dir, database, image_dir, colmap_configs=colmap_configs)
    if reconstruction is not None and verbose:
        logger.info(f'Reconstruction statistics:\n{reconstruction.summary()}'
                    + f'\n\tnum_input_images = {len(image_ids)}')
    return reconstruction
