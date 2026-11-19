# RDW design-space tool: design (prototype)

*Written by Claude (AI-drafted), 2026-11-19, from a conversation with the maintainer.
Status: draft awaiting approval. The physics in "Model" must be checked by the
maintainer before it is relied on.*

## Intent

HiSol.jl's `design_space_a_energy` (branch `new-constraints`) maps the design space for
resonant-dispersive-wave (RDW) emission in a gas-filled hollow capillary fibre (HCF), in the
plane of core radius `a` and pulse energy `E`. It takes seconds to tens of seconds per
evaluation, so exploring inputs is slow. This prototype adds a design tool to the solprop
GUI that shows the same four panels (loss, fission length, minimum soliton order, maximum
soliton order) plus the allowed region, and updates **live** while the user types or drags
sliders. It uses the better soliton term from Luna.jl PR #425 in the RDW phase-matching, and
leads to a *Simulate this* action from a picked point.

The question being tested is whether this GUI idea is useful. That is why it is a prototype
on a branch in the maintainer's fork (`chrisbrahms/solprop`), without the SEP, changelog and
manual-page steps. Those come only if the idea survives.

## Goals

- All inputs of `design_space_a_energy` are editable: gas, λ_target, λ0, τ_FWHM, max length,
  safety factors (`S_sf`, `S_ion`, `S_fiss`, `S_loss`), mode (`kind`, `n`, `m`),
  log/linear axes, and the input and output length constraints.
- Length constraints are switchable per end: none, fixed distance, damage (mirror, with LIDT,
  reference wavelength, `S_fluence`, conversion factor) and window (with its material,
  thickness and aperture options). Semantics follow HiSol `Focusing.jl`.
- Live updates: a coarse grid (about 100 x 100) while an input is changing, refined to full
  resolution (default 512 x 512 as in HiSol) once it rests.
- Clicking the map picks a point. A readout shows the real parameters there: radius, energy,
  pressure, density, ρ·a², the local phase-matching shift, N, N_min, N_max, L_fiss, L_loss,
  HCF length, input and output constraint distances, total length, peak intensity against
  I_crit and I_supp, ZDW, and (where relevant) window thickness or mirror beam radius.
- *Simulate this* turns the picked point into a new simulation sheet, reusing the mechanism
  of the existing *Design a waveguide* page (`design_keywords` and `design_lines` in
  `solprop.service.explore`).
- Never block the window: all computation runs in the kernel process, latest request wins.

## Non-goals and anti-goals

- No SEP, no 100 % coverage gate, no changelog or manual page in the prototype.
- No ionisation term in the phase matching (PR #425 also adds one). HiSol's ionisation
  *limits* (`Nmax_ion`) stay as they are.
- No 3-D scans, no other design axes (HiSol's other `aplot_*` functions), no optimiser.
- Anti-goal: a fast tool that silently disagrees with the physics. A point with no
  phase-matching solution is masked and reported, never filled with a plausible number.
- Anti-goal: physics inside the GUI package. It holds none (see `gui/AGENTS.md`).

## Constraints

- Python only; HiSol (Julia) is not a runtime dependency.
- Gas properties come from solprop's optmat-backed `physdata`, which differ from HiSol's.
  HiSol is therefore a reference for the *formulas*, not for the *numbers*.
- The GUI follows `gui/AGENTS.md`: the window process imports no simulation code, nothing
  over about 50 ms in the window process, public solprop names only.

## Model

All quantities follow HiSol (`Solitons.jl`, `Limits.jl`, `Design.jl`, `HCF.jl`) unless stated.

**Linear case (HiSol).** RDW phase matching at λ_target fixes the density-area product
`ρ·a² = Δβwg / Δβρ`. Everything else (ZDW, β₂ = δ/a², N_max, N_min, L_fiss, L_loss) then
follows from `a`, `E`, τ with the pressure `P = P(ρasq/a²)`.

**With the soliton term (new).** solprop already has this physics in `solprop.tools`:
`soliton_content` (`tools/soliton.py`) gives the *largest soliton constituent* of the input
pulse, whose peak power for a sech pulse is `P0·(2η/N)²` with `η = N − 1/2`, i.e.
`((2N − 1)/N)²·P0`: the factor in PR #425. (This identification is Claude's reading of the
code; it rests on `soliton_content` citing Klaus and Shaw 2003 and Kim, Lee and Lyng.) The
two existing conventions for the soliton's nonlinear wavenumber differ:

- solprop `phase_mismatch(..., nonlinear_phase="largest")`: `γ·P_largest/2`, independent of
  the dispersive-wave frequency;
- Luna PR #425, after Novoa et al., PRL 115, 033901 (2015), Eq. (6) (`novoa2015photoionizationinduced`,
  p. 3): `γ·P_sc·ω_DW/ω_p`, with `P_sc = ((2N−1)/N)²·P0` standing in for the self-compressed
  peak power (Novoa's `P_sc`); note no factor 1/2.

Which one the tool uses is an open item (1). Write the term as `Δβ_nl(ω) = s(ω)·γ·P_largest`
with `s` the chosen convention. Since `γ ∝ ρ`, the phase-matching condition at λ_target is

```
Δβwg/a²  =  ρ · [ Δβρ  −  s(ω_t) · ((2N−1)/N)² · P0 · γ̂ ]
```

where `γ̂ = γ/ρ` is evaluated from `n2/ρ` and `Aeff0·a²` (so the bracket's last term carries
`1/a²`), `ω_t` is the target frequency and `N² = γ·P0·T0²/|β₂|`, with `γ ∝ ρ` and
`β₂(ρ, a) = ρ·β₂,gas(λ0) + β₂,wg/a²`. So `N` depends on `ρ`, and the condition is one scalar
equation in `ρ` at each `(a, E)` grid point. It is solved for the whole grid at once by a
vectorised bracketed Newton iteration started from the linear `ρ`; the root connected to the
linear one is taken. Points without such a root, and points where `β₂ ≥ 0` (no anomalous
dispersion, hence no soliton), are masked and shown as such.

**One β₂ everywhere.** `N`, `L_D`, `L_fiss`, the `N_max` formulas and the ratios all use the
same local `β₂(ρ, a)`. In the linear limit this equals HiSol's `δ/a²`.

Pulse shape is sech, as in HiSol (`P0 = E/2T0`), which equals PR #425's `0.88·E/τ`. PR #425's
default `:gauss` factor of 0.936 is not used.

**Limits at each point**, using the local `ρ`: `N_max = min(N_max,ion, N_max,sf)` with
HiSol's formulas written in terms of the local `ρ·a²` (the `ρasq` method variants);
`N_min` from HiSol's `Nmin`; `L_loss = a³/ᾱ`; `L_fiss = sqrt(L_D·L_NL)`; available HCF length
`= max_length − d_in − d_out` from the constraints at the local pressure; and the four ratios
exactly as in `maxlength_limitratios`. The allowed region is where all four ratios are below 1.

**Dependency tiers** decide what is recomputed when an input changes:

1. gas, λ_target, λ0, mode: spectral and waveguide constants (`Δβwg`, `Δβρ`, `β₂,gas`,
   `Aeff0`, `ᾱ`, `n2/ρ`, `γ3`, `I_supp`) and a per-gas table of `P(ρ)`, built once.
2. τ and the (a, E) grid: the per-point `ρ` solve and everything derived from it.
3. max length, safety factors, constraints: the ratios and the allowed region only, from tier-2
   fields. These inputs change no `ρ`, so they update almost instantly.

The grid axes are chosen as in HiSol from the closed-form corner estimates, evaluated in the
linear limit as a starting range and widened if the soliton term moves the allowed region.

## Approaches considered

1. **Vectorised NumPy module in solprop, tiered cache (chosen).** Python only; fits the
   existing kernel; the per-point solve is natural; the allowed-region outline comes from
   contours of the ratio grid.
2. *Run HiSol from Python (PythonCall or a Julia server).* Rejected: adds Julia to a Python
   tool, HiSol has no per-point `ρ` solve, and every slider move crosses a process boundary.
3. *Precomputed interpolation or a surrogate.* Not chosen now. Kept as a fallback layer on top
   of approach 1 if full resolution turns out too slow.

## Architecture

- **Numerics:** a new module under `src/solprop/tools/` (name: builder's call; `tools/design.py`
  already exists and holds `capillary_design`, so check for overlap first). Pure functions on
  arrays; a small `GasConstants` record holds tier-1 values so tests can inject numbers.
- **Kernel:** a new handler in `solprop.service` that takes the inputs, runs tiers as needed
  (cached by input hash), and replies with plain arrays and scalars. Latest request wins.
- **GUI:** a new workflow page beside *Design a waveguide* (`design_stage.py` is the model for
  debouncing, the bridge to the kernel and *Simulate this*). Four coloured panels with shared
  axes, the allowed region outline (pyqtgraph isocurve items), a picked-point marker and
  readout, an input column with constraint selectors. While an input changes, ask for the
  coarse grid; on rest, ask for the full grid.
- **Simulate this:** from the picked point, a recipe with the gas, radius, pressure, HCF
  length (the length the readout shows), pump wavelength, τ and energy; opened as a new sheet.

## Testing

- The soliton term switched off must give HiSol's linear design **exactly in structure**: the
  per-point `ρ` equals `ρasq/a²`, and the contours of the ratio grid agree with HiSol's closed
  form boundaries (`boundaries`, `min_energy`, `max_energy`) within grid resolution.
- Formula tests with HiSol's gas numbers injected through `GasConstants` (dumped from a HiSol
  session for a fixed case, with the file recording the HiSol commit and the commands). These
  compare the **formulas**, including the canonical He, 160 nm, 800 nm / 10 fs, 5 m case.
- Properties of the nonlinear solve: residual below tolerance at every unmasked point, root
  continuity as the soliton coefficient is ramped from 0, masking where no root exists.
- An end-to-end comparison against HiSol with each side's own gas data is *reported* (a script
  that prints the differences), not asserted, until the gas data are unified.
- GUI: a smoke test and a picker/readout test, as for the existing pages.
- Timing target: tier 3 under 20 ms; coarse tier 2 under about 100 ms; full grid under about
  1 s. Measured and reported, not asserted.

## Open items

1. **Convention for the soliton term**: solprop's existing `γ·P_largest/2`, or PR #425's
   Novoa-style `γ·P_sc·ω/ω_p`. Both use the same `P_largest`. The maintainer decides; the code
   keeps the choice as one clearly marked parameter, so the other can be compared.

## Calls made by Claude (builder-check answers, to be checked)

- **Reuse, not duplicate.** The vectorised module shares the formulas of `soliton_content` and
  `phase_mismatch` rather than copying them, and a test checks that at every unmasked grid point,
  a real `MarcatiliMode` at the solved density has `phase_mismatch(...) = 0` at λ_target using
  solprop's own `phase_mismatch`. This is also the check that the map and a later simulation agree.
- **Pressure for the recipe** comes from solprop's own `P(ρ)`. The readout also shows the
  phase-matched wavelength recomputed with `lambda_rdw` at the picked point, as a visible check;
  agreement is reported, not asserted.
- **Constraints in the first prototype:** all four types, with every parameter HiSol exposes. The
  window material list is whatever solprop's `physdata.glasses` offers; a material missing there is
  not offered.
- **Cancellation** is cooperative, between tiers and between grid chunks: a new request marks the
  running one stale and its result is dropped; a solve is not interrupted midway.
- **Names:** a vectorised module beside `tools/design.py`, with `GasConstants` as the injectable
  tier-1 record.

## Left to the builder

Module and function names, grid sizes for the coarse and full passes, the Newton bracket and
tolerance, how the closed-form variable-window distance (`WindowConstraint`) is vectorised
(HiSol's performance notes, item R8b, give a quadratic in d²), the page layout details, and
the cache key.
