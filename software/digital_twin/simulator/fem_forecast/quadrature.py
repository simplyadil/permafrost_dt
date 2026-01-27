"""Gaussian quadrature rules for 1D and 2D integration."""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np


def _gauss_1d(order: int) -> Tuple[np.ndarray, np.ndarray]:
    if order == 1:
        points = np.array([0.0])
        weights = np.array([2.0])
    elif order == 2:
        val = 1.0 / math.sqrt(3.0)
        points = np.array([-val, val])
        weights = np.array([1.0, 1.0])
    elif order == 3:
        val = math.sqrt(3.0 / 5.0)
        points = np.array([-val, 0.0, val])
        weights = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])
    elif order == 4:
        val1 = math.sqrt((3.0 - 2.0 * math.sqrt(6.0 / 5.0)) / 7.0)
        val2 = math.sqrt((3.0 + 2.0 * math.sqrt(6.0 / 5.0)) / 7.0)
        points = np.array([-val2, -val1, val1, val2])
        weights = np.array(
            [
                (18.0 - math.sqrt(30.0)) / 36.0,
                (18.0 + math.sqrt(30.0)) / 36.0,
                (18.0 + math.sqrt(30.0)) / 36.0,
                (18.0 - math.sqrt(30.0)) / 36.0,
            ]
        )
    else:
        raise ValueError(f"Unsupported Gauss order {order}.")
    return weights, points


def quadrature(order: int, scheme: str, dim: int):
    """Return quadrature weights and points for 1D/2D Gaussian integration."""

    if scheme.upper() != "GAUSS":
        raise ValueError(f"Unsupported quadrature scheme '{scheme}'.")

    w1, q1 = _gauss_1d(order)
    if dim == 1:
        return w1, q1

    if dim == 2:
        weights = np.outer(w1, w1).reshape(-1)
        points = np.array([[x, y] for x in q1 for y in q1])
        return weights, points

    raise ValueError(f"Unsupported quadrature dimension {dim}.")
