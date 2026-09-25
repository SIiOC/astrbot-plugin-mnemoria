"""纯函数：向量打包与余弦相似度（无第三方依赖）。"""

from __future__ import annotations

import array
import math


def pack(vec: list[float]) -> bytes:
    """list[float] -> float32 小端字节串。"""
    return array.array("f", [float(x) for x in vec]).tobytes()


def unpack(blob: bytes, dim: int) -> list[float]:
    """字节串 -> list[float]。dim 仅用于校验。"""
    arr = array.array("f")
    arr.frombytes(blob)
    if dim and len(arr) != dim:
        # 容错：以实际长度为准
        pass
    return list(arr)


def cosine(a: list[float], b: list[float]) -> float:
    """余弦相似度；长度不等或零向量返回 0.0。"""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / math.sqrt(na * nb)


def normalize_vec(vec: list[float]) -> list[float]:
    """L2 归一化（便于用点积当余弦）。零向量原样返回。"""
    norm = math.sqrt(sum(x * x for x in vec))
    if norm <= 0.0:
        return list(vec)
    return [x / norm for x in vec]
