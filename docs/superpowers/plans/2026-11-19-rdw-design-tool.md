# RDW design-space tool Implementation Plan

*Written by Claude (AI-drafted), 2026-11-19. Prototype on branch `prototype/rdw-design-tool`.*

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A GUI tool that shows HiSol's `design_space_a_energy` panels for RDW emission live as inputs change, with the soliton term in the phase matching, a point picker with full parameters, and *Simulate this*.

**Architecture:** A vectorised NumPy package `solprop.tools.rdw_space` computes in three dependency tiers (gas and wavelengths and mode; τ and the (a, E) grid with a per-point ρ solve; limits and constraints). A kernel handler returns arrays; a new GUI stage draws four panels, the allowed region, a picker and a readout.

**Tech Stack:** Python 3.14, NumPy, SciPy, solprop `physdata` (optmat), PySide6 and pyqtgraph, pytest (`uv run pytest`).

**Spec:** `docs/superpowers/specs/2026-11-19-rdw-design-tool-design.md`. HiSol reference (read-only, `git show new-constraints:src/<file>` in `~/.julia/dev/HiSol`): `Design.jl`, `Limits.jl`, `Solitons.jl`, `HCF.jl`, `Focusing.jl`.

## Global Constraints

- Prototype: no SEP, changelog fragment, manual page or 100 % coverage gate. Ruff, ruff format and basedpyright must still pass on touched files; NumPy-style docstrings on public names.
- Python only; HiSol is never a runtime dependency. Gas data comes from solprop `physdata` (optmat) and differs from HiSol's.
- GUI rules (`gui/AGENTS.md`): no physics in `gui/`, public solprop names only, nothing over about 50 ms in the window process, `ui`/`views` import Qt and `model`/`classes` do not.
- Soliton term: solprop's convention `γ·P_largest/2` (`nonlinear_phase="largest"`), `P_largest = P0·(2η/N)²`, `η = N − ½` for sech. It lives in one replaceable function; the PR #425 form `γ·P_largest·ω/ω_p` is a documented alternative.
- Pulse shape sech (`P0 = E/(2·T0)`, `T0 = τ/(2·acosh(√2))`). Reduced Marcatili model for the guide (`β = ω/c·(1 + ρχ/2 − u²c²/(2a²ω²))`, as HiSol).
- HiSol defaults: `S_sf=5`, `S_ion=10`, `S_fiss=1.5`, `S_loss=1`, full grid 512 × 512, mode HE₁₁.
- Commit after each task; messages end with the attribution lines from the session. Do not push.

## Review Focus

1. A point where `β₂ ≥ 0` (normal dispersion, e.g. high pressure or small `a`): masked, flagged `normal_dispersion`, never `NaN` in a colour panel without a mask.
2. No root of the ρ equation (soliton term cancels the linear mismatch): masked, flagged `no_root`, and the picker says so instead of reporting numbers.
3. `N ≤ ½` (no soliton, so no self-compression): maintainer-confirmed that such points are never solutions. The soliton term is zero there, the point is flagged `no_soliton`, and it never enters the allowed region. (HiSol's `N_min = 1.15 + (0.04 + 0.3·log10(λ0/λt))·c·τ/λ0` is above 1.15 whenever λt < λ0, so the N_min ratio excludes these points anyway; the flag makes it explicit.)
4. Grid axes when the region is empty or tiny (λ_target close to λ0, `maxlength` below the constraint distances): the page shows "no allowed region" and does not raise.
5. A fast burst of edits: only the newest request's reply is drawn (kernel supersedes by kind and key; the stage also ignores a reply whose request id is not current).

---

## File structure

Create `src/solprop/tools/rdw_space/`:

| File | Responsibility |
|---|---|
| `__init__.py` | Public names: `GasConstants`, `gas_constants`, `RdwSpace`, `SpaceInputs`, `Constraint` types, `SolitonTerm` |
| `constants.py` | Tier 1: `GasConstants`, `gas_constants(...)`, pressure table |
| `soliton_term.py` | The replaceable nonlinear-mismatch term and `largest_constituent_power` |
| `solve.py` | Tier 2: per-point ρ solve, masks |
| `limits.py` | `n_min`, `n_max`, loss and fission lengths, ratios |
| `constraints.py` | `NoConstraint`, `FixedConstraint`, `DamageConstraint`, `WindowConstraint`, `distance(...)`, `details(...)` |
| `space.py` | `SpaceInputs`, `RdwSpace` (tiers, grid axes, region, `point(a, E)`) |

Create `src/solprop/service/rdw_space.py` (kernel handlers). Modify `src/solprop/service/handlers.py` (register them).
Create `gui/src/solprop_gui/classes/rdw_inputs.py` (no Qt: inputs model and the constraint field table), `gui/src/solprop_gui/ui/rdw_stage.py` (the page), `gui/src/solprop_gui/views/rdw_plots.py` (four panels, picker). Modify `gui/src/solprop_gui/classes/workflow.py`, `ui/workflow_window.py`, `ui/home.py`.
Tests: `tests/tools/test_rdw_space_*.py`, `tests/service/test_rdw_space.py`, `gui/tests/test_rdw_stage.py`; data `tests/data/rdw_space/hisol_reference.json`; scripts `tools/rdw_space_compare.py`, `tools/rdw_space_timing.py`; Julia dump `tools/dump_hisol_reference.jl`.

---

### Task 1: Tier 1, gas and guide constants

**Files:** Create `src/solprop/tools/rdw_space/__init__.py`, `constants.py`; Test `tests/tools/test_rdw_space_constants.py`.

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) class GasConstants` with float fields (SI): `gas: str`, `wavelength_target`, `wavelength_pump`, `unm` (Bessel zero of the mode), `aeff0` (A_eff/a²), `dbeta_wg` (HiSol `Δβwg`), `dbeta_rho` (HiSol `Δβρ`), `g2` (gas GVD per density, `(2γ1' + ωγ1'')/(2c)` at the pump), `w2` (waveguide GVD, `u²c/ω³`), `n2_0` (`n2/ρ`), `gamma3`, `ionisation_potential`, `isupp` (barrier-suppression intensity), `alpha_bar` (loss `ᾱ`, α = ᾱ/a³).
  - `gas_constants(gas: str, wavelength_target: float, wavelength_pump: float, *, kind: str = "HE", n: int = 1, m: int = 1) -> GasConstants`.
  - `pressure_table(gas: str, rho_max: float, points: int = 1024) -> Callable[[np.ndarray], np.ndarray]`: a spline of `log10 P` against `log10 ρ`, returns bar.

- [ ] **Step 1: Failing tests** in `test_rdw_space_constants.py`: (a) `gas_constants("He", 160e-9, 800e-9)` has `dbeta_wg > 0`, `dbeta_rho > 0`, `w2 > 0`, `g2 > 0`; (b) `dbeta_wg/dbeta_rho` equals ρ·a² of a root-found phase match: build `MarcatiliMode(125e-6, "He", pressure, model="reduced")` at `pressure = pressure(gas, (dbeta_wg/dbeta_rho)/a²)` and assert `abs(phase_mismatch(mode, [ω_t], 800e-9)) < 1e-3` (1/m) with `nonlinear_phase="linear"`; (c) `pressure_table` reproduces `physdata.pressure` to rtol 2e-4 over ρ from 1e22 to 1e26.
- [ ] **Step 2: Run, expect failure** (`uv run pytest tests/tools/test_rdw_space_constants.py -v`: ImportError).
- [ ] **Step 3: Implement** `constants.py` using `physdata.sellmeier.chi1`/`dchi1_domega` (check their ω-derivative orders; take the second derivative by differentiating `dchi1_domega` if only the first is exposed), `physdata.gamma3(gas, wavelength=λ0)`, `physdata.ionisation_potential`, `physdata.glasses.fused_silica()` for `ν` in `alpha_bar` with HiSol's HE/TE/TM formulas, `MarcatiliMode`'s Bessel-zero helper for `unm`, and the existing effective-area routine for `aeff0` (find it via `MarcatiliMode.aeff`). `isupp` from the barrier-suppression field and `field_to_intensity` conventions (find the equivalents in `physdata`/`nonlinear`; if absent, the standard `E_supp = Ip²·ε0·... ` form from HiSol `Limits.barrier_suppression_intensity` with Z=1, documented).
- [ ] **Step 4: Run, expect pass.**
- [ ] **Step 5: Commit** (`feat: tier-1 constants for the RDW design space`).

### Task 2: The replaceable soliton term

**Files:** Create `soliton_term.py`; Test `tests/tools/test_rdw_space_soliton_term.py`.

**Interfaces:**
- Produces:
  - `largest_constituent_power(order: np.ndarray, p0: np.ndarray) -> np.ndarray`: `P0·((2N−1)/N)²` where `N > ½`, else `0`.
  - `class SolitonTerm(Protocol)`: `__call__(self, *, gamma: np.ndarray, p_largest: np.ndarray, omega_target: float, omega_pump: float) -> np.ndarray` returning the soliton's nonlinear wavenumber the mismatch subtracts, in 1/m.
  - `solprop_half(...)`: `0.5·gamma·p_largest` (the default, named `DEFAULT_SOLITON_TERM`).
  - `novoa_pr425(...)`: `gamma·p_largest·omega_target/omega_pump`.

- [ ] **Step 1: Failing tests:** `largest_constituent_power` equals `soliton_content(...).power` from `solprop.tools.soliton` for N = 0.7, 1.0, 3.2, 7.5 (sech, same `P0`, `gamma`, `beta2`); it is `0` at N = 0.5 and 0.3; `solprop_half(gamma=g, p_largest=p, ...) == 0.5*g*p`; `novoa_pr425` at `omega_target == omega_pump` is exactly twice `solprop_half`.
- [ ] **Step 2: Run, expect failure.**
- [ ] **Step 3: Implement** the three functions and the Protocol. Each has a docstring naming its convention and source (solprop `phase_mismatch(nonlinear_phase="largest")`; Luna PR #425 after Novoa 2015, Eq. 6).
- [ ] **Step 4: Run, expect pass.**
- [ ] **Step 5: Commit.**

### Task 3: The per-point ρ solve

**Files:** Create `solve.py`; Test `tests/tools/test_rdw_space_solve.py`.

**Interfaces:**
- Consumes: Task 1 `GasConstants`; Task 2 `SolitonTerm`, `largest_constituent_power`.
- Produces:
  - `@dataclass(frozen=True) class Solved` with arrays of the grid shape: `rho`, `beta2`, `gamma`, `order` (N), `p0`, `t0`, `dbeta_nl` (soliton term value), and boolean masks `normal_dispersion`, `no_root`, `no_soliton`; `ok` is `~(normal_dispersion | no_root)`.
  - `solve_rho(c: GasConstants, a: np.ndarray, energy: np.ndarray, fwhm: float, *, term: SolitonTerm = DEFAULT_SOLITON_TERM, tol: float = 1e-10, max_iter: int = 60) -> Solved`. `a` has shape `(na,)`, `energy` `(ne,)`; outputs have shape `(ne, na)` (rows energy, columns radius, as HiSol's `pcolormesh(a, energy, ·)`).

  The equation per point: `F(ρ) = ρ·Δβρ − Δβwg/a² − term(γ = ρ·n2_0·ω0/(c·aeff0·a²), p_largest = largest_constituent_power(N(ρ), P0))` with `β₂(ρ) = ρ·g2 − w2/a²` and `N(ρ)² = γ·P0·T0²/|β₂|`.

- [ ] **Step 1: Failing tests:** (a) with a term returning zeros, `solve_rho(...).rho == (dbeta_wg/dbeta_rho)/a²` to rtol 1e-12 everywhere; (b) with the default term, at every point where `ok & ~no_soliton`, building `MarcatiliMode(a, gas, pressure(gas, rho), model="reduced")` and calling solprop `phase_mismatch(mode, [ω_t], λ0, peak_power=p0, fwhm=τ, nonlinear_phase="largest")` gives `|Δβ| < 1e-6·|Δβwg/a²|` (use a 5 × 5 subgrid for speed); (c) a ramp of the term's coefficient from 0 to 1 moves `rho` continuously (max step between consecutive ramp values bounded); (d) a point constructed with a huge `P0` where the term overwhelms the gas term has `no_root` set; (e) a point with `a` so small that `β₂ > 0` has `normal_dispersion` set and no exception; (f) `N ≤ ½` points have `no_soliton` set and `rho` equal to the linear one.
- [ ] **Step 2: Run, expect failure.**
- [ ] **Step 3: Implement** a vectorised safeguarded Newton (bisection fallback) in `ρ`, bracket `[1e-3·ρ_lin, 3·ρ_lin]`, starting at `ρ_lin`, taking the root connected to `ρ_lin` (continuation along the energy axis from the lowest energy row, where the term is smallest, is acceptable and preferred); no Python loop over grid points, only over iterations.
- [ ] **Step 4: Run, expect pass.**
- [ ] **Step 5: Commit.**

### Task 4: Limits and ratios

**Files:** Create `limits.py`; Test `tests/tools/test_rdw_space_limits.py`.

**Interfaces:**
- Consumes: `GasConstants`, `Solved`.
- Produces: `n_min(wavelength_target, wavelength_pump, fwhm) -> float` (`1.15 + (0.04 + 0.3·log10(λ0/λt))·c·τ/λ0`); `n_max(c: GasConstants, solved: Solved, fwhm: float, *, s_sf: float, s_ion: float) -> np.ndarray` (min of ion and self-focusing limits, HiSol `Nmax_ion`/`Nmax_sf` ρasq variants with the local `Δ = β₂·a²`: `N_sf = sqrt(1.86225·T0²·λ0/(2·aeff0·S_sf·|Δ|))`, `N_ion = sqrt(2π·n2_0·ρa²·I_supp·T0²/(λ0·|Δ|·S_ion))`); `loss_length(c, a) -> np.ndarray` (`a³/ᾱ`); `fission_length(solved, fwhm) -> np.ndarray` (`sqrt(L_D·L_NL)`, `L_D = T0²/|β₂|`, `L_NL = 1/(γ·P0)`); `ratios(...) -> Ratios` with fields `loss`, `fiss`, `nmin`, `nmax`, `allowed` as in HiSol `maxlength_limitratios`: `loss = S_fiss·L_fiss/(L_loss/S_loss)`, `fiss = S_fiss·L_fiss/flength`, `nmin = N_min/N`, `nmax = N/N_max`, `allowed = all < 1 & solved.ok & ~solved.no_soliton`.

- [ ] **Step 1: Failing tests:** the four formulas on a hand-computed single point (numbers written into the test from a plain-Python calculation of HiSol's formulas using the same `GasConstants`); `ratios.allowed` is `False` wherever `normal_dispersion`, `no_root` or `no_soliton`; a mask never yields `allowed=True` with `NaN` ratios.
- [ ] **Step 2: Run, expect failure.** **Step 3: Implement.** **Step 4: Run, expect pass.** **Step 5: Commit.**

### Task 5: Length constraints

**Files:** Create `constraints.py`; Test `tests/tools/test_rdw_space_constraints.py`.

**Interfaces:**
- Produces (frozen dataclasses): `NoConstraint()`, `FixedConstraint(distance: float)`, `DamageConstraint(wavelength_ref: float, lidt: float, s_fluence: float = 5.0, conversion: float = 1.0)` (lidt in J/m²), `WindowConstraint(wavelength_ref: float, material: str = "SiO2", b_max: float = 0.2, thickness: float | None = None, aperture_radius: float | None = None, wavelength_max: float | None = None, s_break: float = 4.0)` (`None` thickness or aperture means variable); a type `Constraint`; `distance(constraint, a, energy, fwhm, pressure) -> np.ndarray` and `details(constraint, a, energy, fwhm, pressure) -> dict[str, np.ndarray]` (always `distance`; damage adds `beam_radius`; window adds `thickness`, `aperture_radius`, `beam_radius`).
- Semantics follow `~/.julia/dev/HiSol` `Focusing.jl` (`mirror_distance`, `window_distance`, `window_thickness_nonlinear`, `window_distance_thickness_aperture`, `window_thickness_breaking`); read them with `git show new-constraints:src/Focusing.jl`. Window material nonlinear index and elastic limit come from `solprop.physdata.glasses`; a material without them is not offered (`window_materials() -> tuple[str, ...]`).

- [ ] **Step 1: Failing tests:** `NoConstraint` gives zeros; `FixedConstraint(0.1)` gives `0.1` broadcast to the grid shape; `DamageConstraint` distance scales as `√E` and equals `sqrt(2E·z_R²/(π·w0²·LIDT/S))` with `w0 = 0.64a`, `z_R = π w0²/λ`; fixed-thickness `WindowConstraint` equals `window_distance` of HiSol for one hand-computed point; a variable window's distance and thickness satisfy `t_NL(d) = t_P(d)` to rtol 1e-9 (closed form from the quadratic in `d²` described in HiSol's `performance_plan.md`, item R8b; keep an internal `scipy.optimize.brentq` cross-check in the test over a 6 × 6 grid); points where thickness limits never cross return `inf` and never raise.
- [ ] **Step 2: Run, expect failure.** **Step 3: Implement** (vectorised; `window_materials` from the glasses module). **Step 4: Run, expect pass.** **Step 5: Commit.**

### Task 6: Orchestration, grid and region

**Files:** Create `space.py`; Test `tests/tools/test_rdw_space_space.py`.

**Interfaces:**
- Consumes: Tasks 1 to 5.
- Produces:
  - `@dataclass(frozen=True) class SpaceInputs`: `gas`, `wavelength_target`, `wavelength_pump`, `fwhm`, `max_length`, `input_constraint: Constraint`, `output_constraint: Constraint`, `s_sf=5.0`, `s_ion=10.0`, `s_fiss=1.5`, `s_loss=1.0`, `kind="HE"`, `n=1`, `m=1`, `log_a=False`, `log_e=False`.
  - `class RdwSpace`: `__init__(self, term: SolitonTerm = DEFAULT_SOLITON_TERM)`; `compute(self, inputs: SpaceInputs, points: int) -> SpaceResult` (keeps the last tier-1 result and tier-2 result and reuses them when only later-tier inputs changed; cache keys are the frozen inputs of each tier); `point(self, inputs: SpaceInputs, a: float, energy: float) -> dict[str, float | bool | str]`.
  - `SpaceResult`: `a`, `energy` (1-D axes, m and J), `ratios: Ratios`, `solved: Solved`, `flength` (available HCF length grid), `masks` (the three flags), `region_empty: bool`, `computed_tiers: tuple[int, ...]` (which tiers ran, for tests).
  - `point(...)` keys: `radius`, `energy`, `pressure`, `density`, `rho_a2`, `dbeta_nl`, `n`, `n_min`, `n_max`, `l_fiss`, `l_loss`, `hcf_length`, `d_in`, `d_out`, `total_length`, `intensity`, `i_crit`, `i_supp`, `zdw`, `beta2`, `window_thickness`/`mirror_beam_radius` where relevant, `flags` (list of mask names).
  - Grid axes as HiSol's `maxlength_limitratios`: energy `0.1·E_min…1.1·E_max`, radius `0.9·a_min…1.3·a_max`, where `E_min, a_min` and `E_max, a_max` come from the linear closed forms (HiSol `min_energy`, `max_energy`, using `GasConstants`); if the linear estimate is empty, fall back to a documented default span and set `region_empty`.

- [ ] **Step 1: Failing tests:** (a) changing only `max_length`, a safety factor or a constraint recomputes only tier 3 (`computed_tiers == (3,)`); changing `fwhm` recomputes tiers 2 and 3; changing `gas` recomputes 1, 2, 3; (b) **linear-limit identity:** with a zero soliton term, the zero contour of `ratios.allowed`-defining ratios agrees with HiSol's closed-form boundaries (`energy_Nmin(a)`, `energy_Nmax(a)`, the loss minimum energy) to within one grid cell, with the closed forms written in the test from `GasConstants` and the formulas in `Limits.jl`; (c) `point()` at a grid point agrees with `compute()`'s arrays at that point to rtol 1e-12; (d) `region_empty` is `True` and nothing raises for `max_length` below the constraint distances; (e) swapping in `novoa_pr425` through `RdwSpace(term=...)` changes `solved.rho` and nothing else in the module's code path.
- [ ] **Step 2: Run, expect failure.** **Step 3: Implement.** **Step 4: Run, expect pass.** **Step 5: Commit.**

### Task 7: HiSol formula reference data

**Files:** Create `tools/dump_hisol_reference.jl`, `tests/data/rdw_space/hisol_reference.json`, Test `tests/tools/test_rdw_space_hisol.py`.

**Interfaces:** the Julia script writes JSON with HiSol's own gas-dependent numbers (`dbeta_wg`, `dbeta_rho`, `g2`, `w2`, `n2_0`, `gamma3`, `isupp`, `alpha_bar`, `aeff0`) and, for the canonical case (He, 160 nm from 800 nm/10 fs, 5 m, `WindowConstraint` fixed-thickness SiO₂ at the pump, `FixedConstraint` input as in `examples/design_space_160nm_He.jl`), HiSol's outputs `RadiusParams`-style at `(a, E) = (125e-6, 200e-6)` including `flength = 2.8737700334469096` (HiSol `performance_plan.md`), `Lfiss`, `N`, `Nmin`, `Nmax`, `ρasq`. The JSON records the HiSol commit, branch, and the exact command.

- [ ] **Step 1: Write the Julia script.** Julia 1.13.3 is installed and HiSol (branch `new-constraints`, checked out in `~/.julia/dev/HiSol`) loads in the default environment, so run it as `julia tools/dump_hisol_reference.jl`. Leave HiSol's working tree as it is (it has an unrelated local `.gitignore` change).
- [ ] **Step 2: Run it and inspect the JSON.** Expected: the `flength` above.
- [ ] **Step 3: Failing test:** build `GasConstants` from the JSON (a `GasConstants(...)` constructed directly), run `RdwSpace` with a zero soliton term, and assert `point(...)` values equal HiSol's to rtol 1e-6 for `n`, `n_min`, `n_max`, `l_fiss`, `hcf_length` (the formulas must match exactly when the gas numbers match).
- [ ] **Step 4: Fix any formula difference found in Tasks 3 to 6** (the test is the oracle), run to pass.
- [ ] **Step 5: Commit** script, JSON and test.

### Task 8: Kernel handlers

**Files:** Create `src/solprop/service/rdw_space.py`; Modify `src/solprop/service/handlers.py` (add `**RDW_SPACE_HANDLERS` beside `**DESIGN_HANDLERS` near line 635 and its import near line 40); Test `tests/service/test_rdw_space.py`.

**Interfaces:**
- Produces: `RDW_SPACE_HANDLERS = {"rdw_space": rdw_space, "rdw_space_point": rdw_space_point}`.
  - `rdw_space(payload) -> dict`: payload `inputs` (plain mapping with the `SpaceInputs` fields; constraints as `{"type": "none"|"fixed"|"damage"|"window", ...fields}`), `points: int`. Returns `a`, `energy` (m, J), `loss`, `fiss`, `nmin`, `nmax` (ratio grids), `allowed` (0/1 grid), `masks` (`normal_dispersion`, `no_root`, `no_soliton` grids), `region_empty`, `tiers`.
  - `rdw_space_point(payload) -> dict`: payload `inputs`, `radius`, `energy`. Returns `RdwSpace.point(...)` fields, plus `rdw_check` (the phase-matched wavelength from `solprop.tools.lambda_rdw` for a real `MarcatiliMode` at the point's pressure with the point's `peak_power` and `fwhm`, `None` if it has no root), and `recipe_text` (from `solprop.service.waveguide.design_recipe({"guide": "capillary", "values": {radius, gas, pressure, wavelength, fwhm, energy, length}}).to_text()`, where `length` is the point's `hcf_length`).
  - A module-level `RdwSpace` instance per kernel process holds the tier cache.

- [ ] **Step 1: Failing tests:** `rdw_space` on a tiny grid returns same-shape lists; two calls differing only in `s_fiss` report `tiers == [3]` the second time; `rdw_space_point` returns a `recipe_text` that `Recipe.loads(...)` accepts and whose `radius`, `pressure`, `energy` and `length` keywords equal the point's; `rdw_check` is within 1 % of `wavelength_target` at a linear-limit point (soliton term set to zero through an optional payload field `soliton_term: "none"|"solprop"|"novoa"`).
- [ ] **Step 2: Run, expect failure.** **Step 3: Implement**; measure the reply's serialisation time for a 512 × 512 result (`tools/rdw_space_timing.py` from Task 11 reports it); if over about 100 ms, send the ratio grids as float32 and masks as one packed integer grid (`bit0..2`), and say so in the docstring. **Step 4: Run, expect pass.** **Step 5: Commit.**

### Task 9: GUI inputs model (no Qt)

**Files:** Create `gui/src/solprop_gui/classes/rdw_inputs.py`; Test `gui/tests/test_rdw_inputs.py`.

**Interfaces:**
- Produces: `RDW_DEFAULTS: Final[dict[str, object]]` (He, 160 nm, 800 nm, 10 fs, 5 m, window input constraint as in the HiSol example, damage output); `CONSTRAINT_TYPES = ("none", "fixed", "damage", "window")`; `constraint_fields(type: str) -> tuple[FieldSpec, ...]` (name, label, unit, default) for each type; `to_payload(values: Mapping[str, object]) -> dict[str, object]` (the `inputs` mapping for the kernel, units to SI via `solprop.recipe.quantity.parse_quantity`); `validate(values) -> str | None` (message for the first bad value, `None` if fine); `coarse_points = 96`, `full_points = 512`.

- [ ] **Step 1: Failing tests:** every constraint type's fields round-trip through `to_payload`; `validate` rejects `λ_target >= λ0`, non-positive `max_length`, an unknown gas; defaults validate.
- [ ] **Step 2: Run, expect failure.** **Step 3: Implement.** **Step 4: Run, expect pass.** **Step 5: Commit.**

### Task 10: GUI page, plots, picker, Simulate this

**Files:** Create `gui/src/solprop_gui/views/rdw_plots.py`, `gui/src/solprop_gui/ui/rdw_stage.py`; Modify `gui/src/solprop_gui/classes/workflow.py` (add `Stage.RDW = "rdw"`), `ui/workflow_window.py` (create the stage next to `self.design`, add it to the stack and to the Simulate-this primary action, as `Stage.DESIGN` is handled at lines ~171-203, 476-489, 522, 842), `ui/home.py` (a second design tile and a `rdw_chosen` signal); Test `gui/tests/test_rdw_stage.py`.

**Interfaces:**
- Consumes: kernel kinds `rdw_space` and `rdw_space_point` (Task 8); `rdw_inputs` (Task 9).
- Produces: `class RdwPlots(QWidget)` with four `pg.ImageItem` panels sharing axes (radius µm, energy µJ), colour range 0 to 2 (`Spectral_r`-like), masked cells drawn grey with a legend entry, the allowed-region outline from `pg.IsocurveItem` on `allowed`, a `point_picked = Signal(float, float)` on click and a marker; `class RdwStage(QWidget)` with `text: str | None` (recipe text of the picked point, as `DesignStage.text`), `bridge`, a `QTimer` debounce (`_DEBOUNCE_MS = 150` as in `design_stage`), a coarse request while editing and a full request after 400 ms of rest, both with key `"rdw-space"` (kernel supersedes by kind and key), reply handling that ignores any request id not current, and a readout table filled from `rdw_space_point`.

- [ ] **Step 1: Failing tests** (pytest-qt, offscreen, a stub bridge): editing a field sends one `rdw_space` request with `points == 96` then, after the rest timer, one with `512`; a reply for a stale request id changes nothing; a reply with `region_empty` shows "no allowed region" and the panels still draw; clicking a cell sends `rdw_space_point` and fills the readout (including `total_length` and `rdw_check`); `RdwStage.text` becomes the reply's `recipe_text`; the workflow window's *Simulate this* in `Stage.RDW` opens it through `document.open_text`.
- [ ] **Step 2: Run, expect failure.** **Step 3: Implement.** The stage follows `design_stage.py` for the form layout, `QuantityField` use and debounce; constraint selectors are a `Segments`/combo that swaps in `constraint_fields(type)` rows. **Step 4: Run, expect pass**, then launch with `uv run solprop-gui`, open the new tile, drag a slider and a field, pick a point, press *Simulate this*, and report what you saw (screenshots optional). **Step 5: Commit.**

### Task 11: Timing and end-to-end comparison reports

**Files:** Create `tools/rdw_space_timing.py`, `tools/rdw_space_compare.py`.

- [ ] **Step 1: `rdw_space_timing.py`** prints, for the canonical case, the time of tier 1, tier 2 at 96² and 512², tier 3, the kernel round-trip including serialisation, against the targets in the spec (tier 3 under 20 ms, coarse tier 2 under about 100 ms, full grid under about 1 s). It asserts nothing.
- [ ] **Step 2: `rdw_space_compare.py`** prints HiSol's end-to-end numbers (from `hisol_reference.json`) beside solprop's for the canonical case, with relative differences and a note that the gas data differ. It asserts nothing.
- [ ] **Step 3: Run both and put the output in the final report.** If a target is missed, say by how much and where the time goes (profile before proposing a remedy).
- [ ] **Step 4: Commit.**

---

## Self-review

- **Spec coverage:** inputs and constraints (T5, T9); live updates and coarse/full (T10); ρ solve and masks (T3); limits and ratios (T4); tiers (T6); picker readout and *Simulate this* (T8, T10); testing items (T3, T6, T7, T11); replaceable term (T2, T6e); non-goals respected (no ionisation term, no SEP).
- **Types:** `GasConstants` (T1) used in T3 to T7; `Solved` (T3) in T4 and T6; `Ratios` (T4) in T6; `Constraint` types (T5) in T6 and T8; payload shapes agree between T8 and T9.
- **Proportion:** the plan states signatures, equations the spec pins and tests; bodies are the builder's.
