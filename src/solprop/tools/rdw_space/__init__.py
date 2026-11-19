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
    WindowPreset,
    details,
    distance,
    window_materials,
    window_presets,
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
from solprop.tools.rdw_space.space import RdwSpace, SpaceInputs, SpaceResult

__all__ = [
    "DEFAULT_SOLITON_TERM",
    "Constraint",
    "DamageConstraint",
    "FixedConstraint",
    "NoConstraint",
    "Ratios",
    "RdwSpace",
    "SolitonTerm",
    "Solved",
    "SpaceInputs",
    "SpaceResult",
    "WindowConstraint",
    "WindowPreset",
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
    "window_presets",
]
