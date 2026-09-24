from math import ceil
from typing import List, Sequence


def split_image_chunks(image_names: Sequence[str], chunk_size: int) -> List[List[str]]:
    """画像名リストを時系列順のまま等分チャンクへ分割する。

    chunk_size <= 0 または len(image_names) <= chunk_size の場合は分割せず
    単一チャンクを返す（境界 n == chunk_size は「超えた」場合ではないため
    分割しない）。分割時は ceil(n / chunk_size) 個へ等分するため、
    全チャンクは chunk_size 以下・サイズ差は 1 以下・連結すると入力と一致し、
    オーバーラップは無い。端数だけの縮退チャンクは発生せず、発動時の最小
    チャンクはおよそ chunk_size / 2 以上になるので mapper の min_model_size
    を下回ることは実用上ない。
    """
    n = len(image_names)
    if chunk_size <= 0 or n <= chunk_size:
        return [list(image_names)]

    n_chunks = ceil(n / chunk_size)
    base, rem = divmod(n, n_chunks)
    chunks = []
    start = 0
    for k in range(n_chunks):
        size = base + 1 if k < rem else base
        chunks.append(list(image_names[start:start + size]))
        start += size
    return chunks
