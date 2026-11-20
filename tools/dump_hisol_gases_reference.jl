# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

# Dump HiSol.jl's own per-gas numbers for the RDW design-space comparison report
# (tools/rdw_space_compare.py, table 3) to
# tests/data/rdw_space/hisol_gases_reference.json. Written by Claude.
#
# Run from the repository root:
#
#     julia tools/dump_hisol_gases_reference.jl
#
# It needs HiSol (branch new-constraints) loadable from the default environment, and only
# reads it. Nothing in the test suite runs this script; the JSON it writes is the
# committed artefact, which the report reads.
#
# The case: a 200 nm dispersive wave from an 800 nm, 10 fs pump in the HE11 mode, 5 m
# for the whole system, no length constraint at either end, in each noble gas. For each
# gas: Luna's gamma3, HiSol's density-area product, N_max in both of HiSol's forms (from
# rho a^2 directly, and through the zero-dispersion wavelength, as Design.Params uses
# it), N_min, and the linear corners min_energy and max_energy. HiSol's allowed region
# needs both N_max > N_min and the corners in order (max_energy's radius above
# min_energy's, which ignores N_min): region_exists is the two together, a necessary
# condition read from HiSol's own closed forms, not a grid of its ratios.

using Dates
using HiSol
import HiSol: Solitons, Limits, Focusing, Design
import Luna.PhysData: γ3_gas

const OUTPUT = joinpath(
    @__DIR__, "..", "tests", "data", "rdw_space", "hisol_gases_reference.json")

λ_target = 200e-9
λ0 = 800e-9
τfwhm = 10e-15
maxlength = 5.0
gases = [:He, :Ne, :Ar, :Kr, :Xe]
none = Focusing.NoConstraint()

function corner(f)
    try
        e, a = f()
        return (e, a, "")
    catch error
        return (nothing, nothing, sprint(showerror, error))
    end
end

rows = map(gases) do gas
    ρasq = Solitons.density_area_product(λ_target, gas, λ0)
    λzd = Solitons.RDW_to_ZDW(λ0, λ_target, gas)
    e_min, a_min, min_error = corner(() -> Design.min_energy(λ_target, λ0, gas, τfwhm))
    e_max, a_max, max_error = corner(
        () -> Design.max_energy(λ_target, λ0, gas, τfwhm, maxlength;
                                input_constraint=none, output_constraint=none))
    ordered = a_min !== nothing && a_max !== nothing && a_max > a_min
    Nmax = Limits.Nmax(λzd, gas, λ0, τfwhm)
    Nmin = Limits.Nmin(λ_target, λ0, τfwhm)
    string(gas) => [
        "gamma3" => γ3_gas(gas),
        "rho_asq" => ρasq,
        "zdw" => λzd,
        "Nmax_rhoasq" => Limits.Nmax(gas, λ0, τfwhm; ρasq),
        "Nmax_zdw" => Nmax,
        "Nmin" => Nmin,
        "min_energy" => e_min,
        "min_energy_radius" => a_min,
        "min_energy_error" => min_error,
        "max_energy" => e_max,
        "max_energy_radius" => a_max,
        "max_energy_error" => max_error,
        "corners_ordered" => ordered,
        "region_exists" => ordered && Nmax > Nmin,
    ]
end

case = [
    "gases" => join(string.(gases), ","),
    "wavelength_target" => λ_target,
    "wavelength_pump" => λ0,
    "fwhm" => τfwhm,
    "max_length" => maxlength,
    "mode" => "HE11",
    "constraints" => "none at either end",
    "s_sf" => 5.0,
    "s_ion" => 10.0,
    "s_fiss" => 1.5,
    "s_loss" => 1.0,
]

hisol_dir = pkgdir(HiSol)
provenance = [
    "written_by" => "tools/dump_hisol_gases_reference.jl (written by Claude)",
    "command" => "julia tools/dump_hisol_gases_reference.jl",
    "date" => string(Dates.today()),
    "julia_version" => string(VERSION),
    "hisol_commit" => readchomp(`git -C $hisol_dir rev-parse HEAD`),
    "hisol_branch" => readchomp(`git -C $hisol_dir rev-parse --abbrev-ref HEAD`),
    "hisol_worktree_dirty" => !isempty(readchomp(`git -C $hisol_dir status --porcelain --untracked-files=no -- src`)),
    "units" => "SI throughout (m, s, J, m^-1 for rho_asq, m^3 for gamma3)",
]

# A minimal JSON writer, as tools/dump_hisol_reference.jl's: floats in Julia's shortest
# round-trip representation, `nothing` as null.
json(::Nothing) = "null"
json(x::AbstractString) = "\"" * escape_string(x) * "\""
json(x::Bool) = x ? "true" : "false"
function json(x::Real)
    isfinite(x) || error("non-finite number $x cannot be written to JSON")
    repr(Float64(x))
end
json(pairs::Vector{<:Pair}; indent="  ") =
    "{\n" * join(["$indent  $(json(k)): $(nested(v, indent * "  "))" for (k, v) in pairs], ",\n") * "\n$indent}"
nested(v::Vector{<:Pair}, indent) = json(v; indent)
nested(v, indent) = json(v)

body = join([
    "  \"provenance\": " * json(provenance),
    "  \"case\": " * json(case),
    "  \"gases\": " * json(rows),
], ",\n")
mkpath(dirname(OUTPUT))
write(OUTPUT, "{\n" * body * "\n}\n")
println("wrote ", normpath(OUTPUT))
