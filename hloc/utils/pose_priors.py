"""COLMAP database への位置プライア（pose_priors）注入。

PDR 軌跡 CSV（共通書式: frame,t_sec,x,y,z,…）の位置を COLMAP 4.x の
pose_priors テーブルへ書き込む。hloc が作る DB は旧スキーマで
pose_priors テーブルを持たないため、ここで CREATE TABLE IF NOT EXISTS する
（COLMAP 4.x は開いた DB に不足テーブルがあっても同スキーマなら読める）。

画像との対応付けは images.name に含まれるフレーム番号
（例: images/frame_number_000042.jpg → 42）の正規表現解析による。
"""

import re
import sqlite3
from pathlib import Path
from typing import Tuple

import numpy as np

COORDINATE_SYSTEM_CARTESIAN = 1     # pycolmap.PosePriorCoordinateSystem.CARTESIAN
FRAME_NUMBER_RE = re.compile(r"(\d+)\.[A-Za-z]+$")

CREATE_POSE_PRIORS_TABLE = """CREATE TABLE IF NOT EXISTS pose_priors (
    corr_data_id INTEGER NOT NULL,
    corr_sensor_id INTEGER NOT NULL,
    corr_sensor_type INTEGER NOT NULL,
    position BLOB NOT NULL,
    position_covariance BLOB,
    gravity BLOB,
    coordinate_system INTEGER,
    PRIMARY KEY (corr_data_id, corr_sensor_id, corr_sensor_type))"""


def load_prior_positions_csv(csv_path: Path) -> dict:
    """共通軌跡 CSV → {frame_number: np.ndarray(3)}。"""
    d = np.genfromtxt(csv_path, delimiter=",", names=True)
    return {int(f): np.array([d["x"][k], d["y"][k], d["z"][k]])
            for k, f in enumerate(np.atleast_1d(d["frame"]))}


def inject_pose_priors_from_csv(database_path: Path, csv_path: Path, *,
                                sigma: float = 3.0) -> Tuple[int, int]:
    """CSV の位置を pose_priors へ注入する。戻り値: (注入数, 対応無しスキップ数)。

    sensor_type=0 はカメラ（COLMAP 4.x の frame_data で実測された値）。
    共分散は等方 σ²·I（PDR のスケール±20% とヨードリフトを包含する保守値）。
    """
    positions = load_prior_positions_csv(Path(csv_path))
    cov_blob = (np.eye(3) * sigma ** 2).astype(np.float64).tobytes()
    con = sqlite3.connect(str(database_path))
    inserted = skipped = 0
    try:
        con.execute(CREATE_POSE_PRIORS_TABLE)
        for image_id, camera_id, name in con.execute(
                "SELECT image_id, camera_id, name FROM images").fetchall():
            m = FRAME_NUMBER_RE.search(name)
            frame = int(m.group(1)) if m else None
            if frame is None or frame not in positions:
                skipped += 1
                continue
            con.execute(
                "INSERT OR REPLACE INTO pose_priors "
                "(corr_data_id, corr_sensor_id, corr_sensor_type, position, "
                " position_covariance, gravity, coordinate_system) "
                "VALUES (?, ?, ?, ?, ?, NULL, ?)",
                (image_id, camera_id, 0,
                 positions[frame].astype(np.float64).tobytes(), cov_blob,
                 COORDINATE_SYSTEM_CARTESIAN))
            inserted += 1
        con.commit()
    finally:
        con.close()
    return inserted, skipped
