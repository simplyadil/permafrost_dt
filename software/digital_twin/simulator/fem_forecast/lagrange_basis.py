"""Lagrange shape functions for basic element types."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def lagrange_basis(elem_type: str, xi) -> Tuple[np.ndarray, np.ndarray]:
    """Return shape functions and derivatives for L2 and Q4 elements."""

    elem_type = elem_type.upper()
    if elem_type == "L2":
        xi_val = float(xi)
        N = np.array([0.5 * (1.0 - xi_val), 0.5 * (1.0 + xi_val)])
        dNdxi = np.array([-0.5, 0.5])
        return N, dNdxi

    if elem_type == "Q4":
        xi_val = float(xi[0])
        eta_val = float(xi[1])
        N = 0.25 * np.array(
            [
                (1.0 - xi_val) * (1.0 - eta_val),
                (1.0 + xi_val) * (1.0 - eta_val),
                (1.0 + xi_val) * (1.0 + eta_val),
                (1.0 - xi_val) * (1.0 + eta_val),
            ]
        )
        dNdxi = 0.25 * np.array(
            [
                [-(1.0 - eta_val), -(1.0 - xi_val)],
                [+(1.0 - eta_val), -(1.0 + xi_val)],
                [+(1.0 + eta_val), +(1.0 + xi_val)],
                [-(1.0 + eta_val), +(1.0 - xi_val)],
            ]
        )
        return N, dNdxi

    raise ValueError(f"Unsupported element type '{elem_type}'.")
