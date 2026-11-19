# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

# Dump HiSol.jl's own numbers for the RDW design-space formula test
# (tests/test_rdw_space_hisol.py) to tests/data/rdw_space/hisol_reference.json.
#
# Run from the repository root:
#
#     julia tools/dump_hisol_reference.jl
#
# It needs HiSol (branch new-constraints) loadable from the default environment, and only
# reads it. The test suite never runs this script; the JSON it writes is the committed
# artefact.
#
# The canonical case is HiSol's examples/design_space_160nm_He.jl: He, a 160 nm
# dispersive wave from an 800 nm, 10 fs pump, 5 m for the whole system, a 1 mm fused
# silica window with an LIDT of 2000 J/m^2 at both ends, read at a = 125 um, E = 200 uJ.
# That example calls design_space_a_energy, which plots; this script builds the same
# numbers with the non-plotting Design.Params and Design.RadiusParams that it uses.

using Dates
using HiSol
import HiSol: Data, HCF, Solitons, Limits, Focusing, Design
import Luna.PhysData: wlfreq, c, ε_0, γ3_gas, ionisation_potential, ref_index

const OUTPUT = joinpath(@__DIR__, "..", "tests", "data", "rdw_space", "hisol_reference.json")

# The canonical case.
λ_target = 160e-9
gas = :He
λ0 = 800e-9
τfwhm = 10e-15
maxlength = 5.0
a = 125e-6
energy = 200e-6
window = WindowConstraint(λ0, :SiO2; thickness=1e-3, LIDT=2000)

# Gas- and mode-dependent constants: the fields of solprop's GasConstants.
ω0 = wlfreq(λ0)
u_nm = HCF.get_unm(; n=1, m=1, kind=:HE)
γ3 = γ3_gas(gas)
constants = [
    "unm" => u_nm,
    "aeff0" => HCF.Aeff0(),
    "dbeta_wg" => Solitons.Δβwg(λ_target, λ0),
    "dbeta_rho" => Solitons.Δβρ(λ_target, gas, λ0),
    "g2" => (2 * Data.dγ1dω(gas, λ0, 1) + ω0 * Data.dγ1dω(gas, λ0, 2)) / (2c),
    "w2" => u_nm^2 * c / ω0^3,
    "n2_0" => Data.n2_0(gas),
    "gamma3" => γ3,
    "ionisation_potential" => ionisation_potential(gas),
    "isupp" => Limits.barrier_suppression_intensity(gas),
    "alpha_bar" => HCF.αbar_a(λ0),
]

# The gas and glass data the constants above are built from, so that the test can feed
# them through solprop's own gas_constants formulas.
primitives = [
    "gamma1_target" => Data.γ1(gas, λ_target), # susceptibility per unit density, m^3
    "gamma1_pump" => Data.γ1(gas, λ0),
    "dgamma1_domega_pump" => Data.dγ1dω(gas, λ0, 1), # m^3 s
    "d2gamma1_domega2_pump" => Data.dγ1dω(gas, λ0, 2), # m^3 s^2
    "silica_index_pump" => real(ref_index(:SiO2, λ0)), # as HCF.αbar_a
]

# HiSol's design numbers at the canonical point, as the example's `params(a, energy)`.
params = Design.Params(λ_target, gas, λ0, maxlength;
                       input_constraint=window, output_constraint=window)
rp = Design.RadiusParams(a, params)
point = rp(energy, τfwhm)
window_details = Focusing.details(window, a, energy, τfwhm; pressure=rp.pressure)
T0, P0 = Solitons.T0P0(τfwhm, energy)

hisol = [
    # Params: a- and energy-independent.
    "rho_asq" => params.ρasq,
    "zdw" => params.zdw,
    "delta" => params.δ, # beta2 * a^2 through the ZDW (eq. S7), what RadiusParams uses
    "Delta" => HCF.Δ(gas, λ0, params.ρasq), # beta2 * a^2 through rho*a^2 directly
    # RadiusParams at a.
    "radius" => rp.a,
    "density" => rp.density,
    "pressure" => rp.pressure,
    "aeff" => rp.aeff,
    "n2" => rp.n2,
    "gamma" => rp.γ,
    "beta2" => rp.β2,
    "Lloss" => rp.Lloss,
    # RadiusParams at (a, energy).
    "energy" => point.energy,
    "tfwhm" => point.τfwhm,
    "T0" => T0,
    "P0" => P0,
    "intensity" => point.intensity,
    "flength" => point.flength,
    "N" => point.N,
    "Nmin" => point.Nmin,
    "Nmax" => point.Nmax,
    "Lfiss" => point.Lfiss,
    "Icrit" => point.Icrit,
    "Isupp" => point.Isupp,
    # The window at the pressure above (both ends use the same window).
    "window_distance" => window_details.distance,
    "window_thickness" => window_details.thickness,
    "window_aperture" => window_details.aperture,
]

# Two more windows at the same point, so that the pressure-dependent branch of the
# window (a variable thickness, found where the B-integral and pressure limits meet) is
# compared too: at HiSol's pressure, and at 5 bar, where the pressure limit binds harder.
variable = WindowConstraint(λ0, :SiO2)
variable_details = Focusing.details(variable, a, energy, τfwhm; pressure=rp.pressure)
variable_5bar = Focusing.details(variable, a, energy, τfwhm; pressure=5.0)
window_variants = [
    "variable_distance" => variable_details.distance,
    "variable_thickness" => variable_details.thickness,
    "variable_aperture" => variable_details.aperture,
    "variable_5bar_distance" => variable_5bar.distance,
    "variable_5bar_thickness" => variable_5bar.thickness,
    "variable_5bar_aperture" => variable_5bar.aperture,
    # The LIDT distance of the canonical window, which does not bind there.
    "lidt_distance" => Focusing.mirror_distance(a, λ0, energy, window.LIDT),
]

case = [
    "gas" => string(gas),
    "wavelength_target" => λ_target,
    "wavelength_pump" => λ0,
    "fwhm" => τfwhm,
    "max_length" => maxlength,
    "mode" => "HE11",
    "window_n2" => Data.n2_solid(:SiO2),
    "window_elastic_limit" => Data.elastic_limit(:SiO2),
    "window_thickness" => window.thickness,
    "window_lidt" => window.LIDT,
    "window_b_max" => window.Bmax,
    "window_s_fluence" => window.S_fluence,
    "window_s_break" => window.S_break,
    "window_aperture_factor" => window.aperture_factor,
    "s_sf" => params.S_sf,
    "s_ion" => params.S_ion,
    "s_fiss" => params.S_fiss,
    "s_loss" => params.S_loss,
]

hisol_dir = pkgdir(HiSol)
provenance = [
    "written_by" => "tools/dump_hisol_reference.jl (written by Claude)",
    "command" => "julia tools/dump_hisol_reference.jl",
    "date" => string(Dates.today()),
    "julia_version" => string(VERSION),
    "hisol_commit" => readchomp(`git -C $hisol_dir rev-parse HEAD`),
    "hisol_branch" => readchomp(`git -C $hisol_dir rev-parse --abbrev-ref HEAD`),
    "hisol_worktree_dirty" => !isempty(readchomp(`git -C $hisol_dir status --porcelain --untracked-files=no -- src`)),
    "luna_c" => c,
    "luna_eps0" => ε_0,
    "units" => "SI throughout (m, s, J, W, m^-3), pressure in bar",
]

# A minimal JSON writer, so the script needs nothing beyond HiSol. Floats are written
# with Julia's shortest round-trip representation.
json(x::AbstractString) = "\"" * escape_string(x) * "\""
json(x::Bool) = x ? "true" : "false"
function json(x::Real)
    isfinite(x) || error("non-finite number $x cannot be written to JSON")
    repr(Float64(x))
end
json(pairs::Vector{<:Pair}; indent="  ") =
    "{\n" * join(["$indent  $(json(k)): $(json(v))" for (k, v) in pairs], ",\n") * "\n$indent}"

body = join([
    "  \"provenance\": " * json(provenance),
    "  \"case\": " * json(case),
    "  \"constants\": " * json(constants),
    "  \"primitives\": " * json(primitives),
    "  \"hisol\": " * json(hisol),
    "  \"window_variants\": " * json(window_variants),
], ",\n")
mkpath(dirname(OUTPUT))
write(OUTPUT, "{\n" * body * "\n}\n")
println("wrote ", normpath(OUTPUT))
println("flength = ", point.flength)
