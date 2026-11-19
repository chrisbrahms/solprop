# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Design space of resonant dispersive wave emission in a gas-filled capillary."""

from solprop.tools.rdw_space.constraints import (
    Constraint,
    DamageConstraint,
    FixedConstraint,
    NoConstraint,
    WindowConstraint,
    details,
    distance,
    window_materials,
)
from solprop.tools.rdw_space.limits import (
    Ratios,
    fission_length,
    loss_length,
    n_max,
    n_min,
    ratios,
)
from solprop.tools.rdw_space.soliton_term import (
    DEFAULT_SOLITON_TERM,
    SolitonTerm,
    largest_constituent_power,
    novoa_pr425,
    solprop_half,
)
from solprop.tools.rdw_space.solve import Solved, solve_rho

__all__ = [
    "DEFAULT_SOLITON_TERM",
    "Constraint",
    "DamageConstraint",
    "FixedConstraint",
    "NoConstraint",
    "Ratios",
    "SolitonTerm",
    "Solved",
    "WindowConstraint",
    "details",
    "distance",
    "fission_length",
    "largest_constituent_power",
    "loss_length",
    "n_max",
    "n_min",
    "novoa_pr425",
    "ratios",
    "solprop_half",
    "solve_rho",
    "window_materials",
]
