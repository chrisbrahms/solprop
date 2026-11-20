# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The RDW design stage, offscreen: live requests, stale replies, the picker, Simulate.

The kernel's replies are real: the service's handler functions are called directly, as
the kernel calls them, and the stage receives them through a stub bridge. One test
starts a real kernel.
"""

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PySide6.QtCore import (
    QObject,
    QPoint,
    QPointF,
    QSettings,
    QStandardPaths,
    Qt,
    Signal,
)
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QComboBox
from pytestqt.qtbot import QtBot
from solprop.service.messages import Reply
from solprop.service.rdw_space import rdw_space, rdw_space_options, rdw_space_point

from solprop_gui.classes.rdw_inputs import RdwInputs
from solprop_gui.classes.workflow import Stage
from solprop_gui.ui import rdw_stage
from solprop_gui.ui.bridge import ServiceBridge
from solprop_gui.ui.quantity import QuantityField
from solprop_gui.ui.rdw_stage import COARSE, FULL, RdwStage
from solprop_gui.ui.workflow_window import WorkflowWindow
from solprop_gui.views.rdw_plots import MASKED, cell_at, colours

WAIT_MS = 120_000


class StubBridge(QObject):
    """A bridge that records requests and answers only when a test says so."""

    replied = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self.sent: list[tuple[int, str, dict[str, Any], str]] = []

    def request(
        self,
        kind: str,
        payload: Mapping[str, object] | None = None,
        *,
        revision: int = 0,
        key: str = "",
    ) -> int:
        del revision
        request_id = len(self.sent) + 1
        self.sent.append((request_id, kind, dict(payload or {}), key))
        return request_id

    def of(self, kind: str) -> list[tuple[int, str, dict[str, Any], str]]:
        return [entry for entry in self.sent if entry[1] == kind]

    def answer(self, request_id: int, value: object = None, error: str | None = None):
        kind = self.sent[request_id - 1][1]
        self.replied.emit(Reply(kind, request_id, 0, value=value, error=error))


@pytest.fixture(scope="module")
def options() -> Mapping[str, Any]:
    return rdw_space_options({})


A_PINNED = np.linspace(60e-6, 160e-6, 7)
E_PINNED = np.linspace(50e-6, 400e-6, 5)


@pytest.fixture(scope="module")
def pinned(options: Mapping[str, Any]) -> dict[str, Any]:
    """A real reply on a non-square grid: 5 energies (rows) by 7 radii (columns)."""
    model = RdwInputs(options)
    model.axes = (A_PINNED, E_PINNED)
    return rdw_space(model.to_payload())


@pytest.fixture(scope="module")
def coarse(options: Mapping[str, Any]) -> dict[str, Any]:
    """A real reply of the starting design on 12 points, the kernel's own axes."""
    model = RdwInputs(options)
    model.points = 12
    return rdw_space(model.to_payload())


@pytest.fixture
def bridge() -> StubBridge:
    return StubBridge()


@pytest.fixture
def stage(
    qtbot: QtBot, bridge: StubBridge, options: Mapping[str, Any]
) -> Iterator[RdwStage]:
    made = RdwStage(bridge)  # pyright: ignore[reportArgumentType]
    qtbot.addWidget(made)
    made.set_options(options)
    made._coarse.stop()
    made._rest.stop()
    bridge.sent.clear()
    yield made


def edit(field: QuantityField, text: str) -> None:
    """Type into a field as a person does."""
    field.number.setText(text)
    field.number.textEdited.emit(text)


def drawn(stage: RdwStage, bridge: StubBridge, reply: Mapping[str, Any]) -> None:
    """Ask for the full grid and answer it with `reply`."""
    stage._ask_space(FULL)
    bridge.answer(bridge.of("rdw_space")[-1][0], reply)


# ------------------------------------------------------------------ live requests


def test_an_edit_asks_coarse_then_full(
    qtbot: QtBot, stage: RdwStage, bridge: StubBridge
) -> None:
    edit(stage.fields["fwhm"], "12")
    qtbot.waitUntil(lambda: len(bridge.of("rdw_space")) >= 1, timeout=2000)
    first = bridge.of("rdw_space")
    assert len(first) == 1
    assert first[0][2]["points"] == COARSE == 96
    assert first[0][3] == "rdw-space"
    assert first[0][2]["inputs"]["fwhm"] == pytest.approx(12e-15)
    assert stage.busy.isVisibleTo(stage)
    qtbot.waitUntil(lambda: len(bridge.of("rdw_space")) == 2, timeout=2000)
    full = bridge.of("rdw_space")[1]
    assert full[2]["points"] == FULL == 512
    assert full[3] == "rdw-space"
    qtbot.wait(500)
    assert len(bridge.of("rdw_space")) == 2  # nothing more without an edit
    assert not bridge.of("rdw_space_point")  # nothing picked


def test_edits_in_quick_succession_send_one_coarse_request(
    qtbot: QtBot, stage: RdwStage, bridge: StubBridge
) -> None:
    for text in ("1", "12", "120"):
        edit(stage.fields["fwhm"], text)
        qtbot.wait(40)
    qtbot.waitUntil(lambda: len(bridge.of("rdw_space")) == 2, timeout=2000)
    sent = bridge.of("rdw_space")
    assert [s[2]["points"] for s in sent] == [96, 512]
    assert sent[0][2]["inputs"]["fwhm"] == pytest.approx(120e-15)


def test_a_typing_error_sends_nothing_and_says_why(
    qtbot: QtBot, stage: RdwStage, bridge: StubBridge
) -> None:
    stage.model.set("fwhm", "ten fs")  # pyright: ignore[reportOptionalMemberAccess]
    stage._edited()
    qtbot.waitUntil(lambda: "fwhm" in stage.status.text(), timeout=2000)
    qtbot.wait(500)
    assert not bridge.of("rdw_space")


def test_a_stale_reply_changes_nothing(
    stage: RdwStage, bridge: StubBridge, coarse: dict[str, Any], pinned: dict[str, Any]
) -> None:
    stage._ask_space(COARSE)
    stage._ask_space(FULL)
    old, new = (s[0] for s in bridge.of("rdw_space"))
    # The coarse reply arrives after the full grid was asked for: ignored.
    bridge.answer(old, coarse)
    assert stage.space is None
    assert stage.plots.space is None
    bridge.answer(new, pinned)
    assert stage.space is not None
    assert stage.space.shape == (5, 7)
    assert not stage.busy.isVisibleTo(stage)
    # A late reply to the old request again: still nothing.
    bridge.answer(old, coarse)
    assert stage.space.shape == (5, 7)


def test_an_empty_region_says_so_and_still_draws(
    stage: RdwStage, bridge: StubBridge, coarse: dict[str, Any]
) -> None:
    empty = dict(coarse)
    empty["flags"] = np.asarray(coarse["flags"]) & ~np.uint8(1)
    empty["region_empty"] = True
    drawn(stage, bridge, empty)
    assert stage.plots.message.text() == "No allowed region for these inputs."
    assert not stage.plots.message.isHidden()
    for image in stage.plots.images:
        assert image.image is not None and image.image.shape == (12, 12, 4)
    x, _y = stage.plots.outlines[0].getData()
    assert x is None or np.size(x) == 0  # no outline without a region
    drawn(stage, bridge, coarse)
    assert stage.plots.message.isHidden()
    x, _y = stage.plots.outlines[0].getData()
    assert x is not None and np.size(x) > 0


def test_a_failed_reply_says_why_and_keeps_the_plot(
    stage: RdwStage, bridge: StubBridge, coarse: dict[str, Any]
) -> None:
    drawn(stage, bridge, coarse)
    image = stage.plots.images[0].image
    assert image is not None
    before = image.copy()
    stage._ask_space(FULL)
    bridge.answer(bridge.of("rdw_space")[-1][0], error="inputs.fwhm: must be positive")
    assert "must be positive" in stage.status.text()
    assert "Traceback" not in stage.status.text()
    assert stage.plots.space is stage.space
    after = stage.plots.images[0].image
    assert after is not None and np.array_equal(after, before)
    # The next good reply clears the error.
    drawn(stage, bridge, coarse)
    assert stage.status.text() == ""


def test_masked_cells_are_grey_and_the_colours_fixed() -> None:
    grid = np.array([[0.0, 1.0, 2.0, 5.0, np.nan, np.inf]])
    pixels = colours(grid)
    assert tuple(pixels[0, 4]) == MASKED
    assert np.array_equal(pixels[0, 2], pixels[0, 3])  # clipped at 2
    assert np.array_equal(pixels[0, 2], pixels[0, 5])  # infinity is the top
    assert tuple(pixels[0, 0][:3]) == (0x5E, 0x4F, 0xA2)


# ------------------------------------------------------------------ the picker


def click_cell(qtbot: QtBot, stage: RdwStage, panel: int, row: int, column: int,
               offset: tuple[float, float] = (0.5, 0.5)) -> None:  # fmt: skip
    """Click a panel where it draws a cell, at `offset` within the cell (0..1)."""
    plots = stage.plots
    image = plots.images[panel]
    view_point = image.mapToParent(QPointF(column + offset[0], row + offset[1]))
    box = plots.plots[panel].getViewBox()
    scene_point = box.mapViewToScene(view_point)
    widget_point = plots.view.mapFromScene(scene_point)
    QTest.mouseClick(
        plots.view.viewport(),
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
        QPoint(widget_point.x(), widget_point.y()),
    )
    qtbot.wait(20)


def test_a_click_picks_the_cell_it_lands_in_on_a_non_square_grid(
    qtbot: QtBot,
    stage: RdwStage,
    bridge: StubBridge,
    pinned: dict[str, Any],
    options: Mapping[str, Any],
) -> None:
    stage.resize(1400, 850)
    stage.show()
    qtbot.waitExposed(stage)
    drawn(stage, bridge, pinned)
    space = stage.space
    assert space is not None and space.shape == (5, 7)
    # Pixel centres lie on the grid values: rows are energies, columns radii.
    centre = stage.plots.images[0].mapToParent(QPointF(5.5, 3.5))
    assert centre.x() == pytest.approx(A_PINNED[5] * 1e6)
    assert centre.y() == pytest.approx(E_PINNED[3] * 1e6)
    for panel, row, column, offset in (
        (2, 3, 5, (0.5, 0.5)),
        (0, 1, 6, (0.9, 0.1)),  # near the cell's corner, still that cell
        (3, 4, 0, (0.1, 0.9)),
    ):
        bridge.sent.clear()
        click_cell(qtbot, stage, panel, row, column, offset)
        points = bridge.of("rdw_space_point")
        assert len(points) == 1, (panel, row, column)
        payload = points[0][2]
        assert payload["radius"] == A_PINNED[column]
        assert payload["energy"] == E_PINNED[row]
        assert points[0][3] == "rdw-point"
    # The readout of a real point reply.
    model = RdwInputs(options)
    reply = rdw_space_point(model.point_payload(125e-6, 200e-6))
    bridge.answer(bridge.of("rdw_space_point")[-1][0], reply)
    assert stage.readout_text("Total length") == f"{reply['total_length']:.4g} m"
    assert stage.readout_text("Phase-matched wavelength (check)") == "160 nm"
    assert stage.simulate_text() == reply["recipe_text"] is not None
    assert "Simulate this" in stage.recipe_note.text()
    for halo, cross in stage.plots.markers:
        assert np.size(cross.getData()[0]) == 1
        assert np.size(halo.getData()[0]) == 1


def test_a_logarithmic_axis_maps_clicks_to_its_cells(
    options: Mapping[str, Any],
) -> None:
    from solprop_gui.classes.rdw_inputs import decode_space_reply  # noqa: PLC0415

    model = RdwInputs(options)
    model.points = 8
    model.set_flag("log_a", True)
    space = decode_space_reply(rdw_space(model.to_payload()))
    logs = np.log10(space.a_um)
    assert not np.allclose(np.diff(space.a_um), np.diff(space.a_um)[0])
    step = logs[1] - logs[0]
    y = float(space.energy_uj[2])
    assert cell_at(space, float(logs[4]) + 0.45 * step, y) == (2, 4)
    assert cell_at(space, float(logs[4]) + 0.55 * step, y) == (2, 5)
    assert cell_at(space, float(logs[0]) - 0.6 * step, y) is None


def test_inputs_changed_read_the_picked_point_again(
    qtbot: QtBot, stage: RdwStage, bridge: StubBridge, coarse: dict[str, Any]
) -> None:
    drawn(stage, bridge, coarse)
    stage.pick(125e-6, 200e-6)
    bridge.answer(
        bridge.of("rdw_space_point")[-1][0], {"flags": [], "recipe_text": "x"}
    )
    assert stage.text == "x"
    bridge.sent.clear()
    edit(stage.fields["max_length"], "4")
    assert stage.text is None  # the recipe no longer matches the inputs
    qtbot.waitUntil(lambda: len(bridge.of("rdw_space_point")) == 1, timeout=2000)
    payload = bridge.of("rdw_space_point")[0][2]
    assert (payload["radius"], payload["energy"]) == (125e-6, 200e-6)
    assert payload["inputs"]["max_length"] == 4.0


def fire(stage: RdwStage) -> None:
    """Fire both request timers now, as the rest after an edit does."""
    stage._coarse.stop()
    stage._rest.stop()
    stage._ask_space(COARSE)
    stage._ask_full()


def picked(stage: RdwStage, bridge: StubBridge, coarse: dict[str, Any]) -> int:
    """Draw a space, pick a point and return the point request's id, unanswered."""
    drawn(stage, bridge, coarse)
    stage.pick(125e-6, 200e-6)
    return bridge.of("rdw_space_point")[-1][0]


def test_a_point_reply_for_inputs_since_edited_is_ignored(
    stage: RdwStage, bridge: StubBridge, coarse: dict[str, Any]
) -> None:
    old = picked(stage, bridge, coarse)
    edit(stage.fields["max_length"], "4")
    bridge.answer(old, {"flags": [], "recipe_text": "OLD"})
    assert stage.text is None
    assert stage.readout.rowCount() == 0  # the readout is not filled from it
    # The point asked for at the new inputs is taken.
    fire(stage)
    new = bridge.of("rdw_space_point")[-1]
    assert new[0] != old and new[2]["inputs"]["max_length"] == 4.0
    bridge.answer(new[0], {"flags": [], "recipe_text": "NEW"})
    assert stage.text == "NEW"
    assert stage.readout_text("Masks set") == "none"


def test_a_point_reply_for_inputs_since_made_invalid_is_ignored(
    qtbot: QtBot, stage: RdwStage, bridge: StubBridge, coarse: dict[str, Any]
) -> None:
    window_states: list[bool] = []
    stage.recipe_changed.connect(lambda: window_states.append(stage.text is not None))
    old = picked(stage, bridge, coarse)
    stage.model.set("fwhm", "ten fs")  # pyright: ignore[reportOptionalMemberAccess]
    stage._edited()
    bridge.answer(old, {"flags": [], "recipe_text": "OLD"})
    fire(stage)  # the new inputs do not read: nothing is asked
    assert "fwhm" in stage.status.text()
    assert len(bridge.of("rdw_space_point")) == 1
    assert stage.text is None and stage.simulate_text() is None
    assert True not in window_states  # Simulate this never came back on


def test_holding_the_axes_before_any_full_plot_takes_the_first(
    stage: RdwStage, bridge: StubBridge, pinned: dict[str, Any]
) -> None:
    stage.hold.setChecked(True)  # no full plot yet
    fire(stage)
    assert "axes" not in bridge.of("rdw_space")[-1][2]
    bridge.answer(bridge.of("rdw_space")[-1][0], pinned)
    bridge.sent.clear()
    edit(stage.fields["fwhm"], "12")
    fire(stage)
    for _id, _kind, payload, _key in bridge.of("rdw_space"):
        assert payload["axes"] == {"a": list(A_PINNED), "energy": list(E_PINNED)}
    # Unticked: solprop's own axes again.
    bridge.sent.clear()
    stage.hold.setChecked(False)
    fire(stage)
    assert all("axes" not in s[2] for s in bridge.of("rdw_space"))


def test_holding_again_takes_the_latest_full_plot_and_keeps_a_copy(
    stage: RdwStage, bridge: StubBridge, pinned: dict[str, Any], coarse: dict[str, Any]
) -> None:
    drawn(stage, bridge, pinned)
    stage.hold.setChecked(True)
    stage.hold.setChecked(False)
    drawn(stage, bridge, coarse)  # a newer full plot, on solprop's axes
    stage.hold.setChecked(True)
    assert stage.space is not None
    held = stage._held
    assert held is not None
    assert np.array_equal(held[0], stage.space.a)
    assert held[0] is not stage.space.a  # a copy, not the drawn space's array
    bridge.sent.clear()
    fire(stage)
    assert bridge.of("rdw_space")[-1][2]["axes"]["a"] == list(stage.space.a)


# ------------------------------------------------------------------ the form


def test_a_preset_disables_the_rows_it_supplies(
    stage: RdwStage, bridge: StubBridge
) -> None:
    rows = stage.end_fields["in"]
    assert not rows["material"].isEnabled()
    assert not rows["elastic_limit"].isEnabled()
    material = rows["material"]
    assert isinstance(material, QuantityField)
    assert "SiO2" in material.number.placeholderText()
    preset = rows["preset"]
    assert isinstance(preset, QComboBox)
    preset.setCurrentIndex(preset.findData(""))
    assert rows["material"].isEnabled()
    assert rows["elastic_limit"].isEnabled()
    # The other end keeps its own preset.
    assert not stage.end_fields["out"]["material"].isEnabled()
    # Optional fields say what empty means; the defaults show in grey.
    thickness = rows["thickness"]
    assert isinstance(thickness, QuantityField)
    assert thickness.number.text() == "1"
    lidt = rows["lidt"]
    assert isinstance(lidt, QuantityField)
    assert lidt.unit() == "J/cm^2"
    assert lidt.number.text() == "0.2"
    aperture = rows["aperture_radius"]
    assert isinstance(aperture, QuantityField)
    assert aperture.number.placeholderText() == "variable"
    assert stage.fields["s_sf"].number.placeholderText() == "5"


def test_a_constraint_type_swaps_its_rows(
    qtbot: QtBot, stage: RdwStage, bridge: StubBridge
) -> None:
    choice: QComboBox = stage._out_choice  # pyright: ignore[reportAttributeAccessIssue]
    choice.setCurrentIndex(choice.findData("fixed"))
    assert set(stage.end_fields["out"]) == {"distance"}
    distance = stage.end_fields["out"]["distance"]
    assert isinstance(distance, QuantityField)
    edit(distance, "0.5")
    qtbot.waitUntil(lambda: len(bridge.of("rdw_space")) >= 1, timeout=2000)
    constraint = bridge.of("rdw_space")[-1][2]["inputs"]["output_constraint"]
    assert constraint == {"type": "fixed", "distance": 0.5}
    choice.setCurrentIndex(choice.findData("none"))
    assert stage.end_fields["out"] == {}


def test_holding_the_axes_sends_those_of_the_last_full_plot(
    qtbot: QtBot, stage: RdwStage, bridge: StubBridge, pinned: dict[str, Any]
) -> None:
    drawn(stage, bridge, pinned)
    assert "axes" not in bridge.of("rdw_space")[-1][2]
    bridge.sent.clear()
    stage.hold.setChecked(True)
    qtbot.waitUntil(lambda: len(bridge.of("rdw_space")) == 2, timeout=2000)
    for _id, _kind, payload, _key in bridge.of("rdw_space"):
        assert payload["axes"] == {"a": list(A_PINNED), "energy": list(E_PINNED)}
    bridge.sent.clear()
    stage.hold.setChecked(False)
    qtbot.waitUntil(lambda: len(bridge.of("rdw_space")) == 2, timeout=2000)
    assert "axes" not in bridge.of("rdw_space")[-1][2]


def test_held_axes_are_thinned_for_the_coarse_grid(stage: RdwStage) -> None:
    stage._held = (np.linspace(1e-5, 2e-4, 512), np.linspace(1e-6, 4e-4, 512))
    a, energy = stage._axes_for(COARSE)  # pyright: ignore[reportGeneralTypeIssues]
    assert a.size <= COARSE and energy.size <= COARSE
    assert np.allclose(np.diff(a), np.diff(a)[0])  # still evenly spaced
    full = stage._axes_for(FULL)
    assert full is not None and full[0].size == 512


def test_the_options_are_asked_for_as_the_stage_comes_up(
    qtbot: QtBot, bridge: StubBridge, options: Mapping[str, Any]
) -> None:
    made = RdwStage(bridge)  # pyright: ignore[reportArgumentType]
    qtbot.addWidget(made)
    made.show()
    asked = bridge.of("rdw_space_options")
    assert len(asked) == 1
    made.set_gases(["He", "Ar"])
    bridge.answer(asked[0][0], dict(options))
    assert made.model is not None
    assert made.gas.currentData() == "He"
    made.hide()
    made.show()
    assert len(bridge.of("rdw_space_options")) == 1


# ------------------------------------------------------------------ the window


@pytest.fixture
def private_settings(tmp_path: Path) -> Iterator[None]:
    QStandardPaths.setTestModeEnabled(True)
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(
        QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path / "s")
    )
    yield
    QStandardPaths.setTestModeEnabled(False)


def test_home_leads_to_the_stage_and_simulate_this_opens_the_point(
    qtbot: QtBot, tmp_path: Path, private_settings: None
) -> None:
    del private_settings
    window = WorkflowWindow(workspace=tmp_path / "runs")
    window.confirm = lambda _title, _text: "discard"
    qtbot.addWidget(window)
    window.show()
    try:
        window.home.tiles["rdw"].clicked.emit()
        assert window.stage is Stage.RDW
        assert window.stack.currentWidget() is window.rdw
        assert window.primary.text() == "Simulate this"
        assert not window.primary.isEnabled()  # nothing picked
        assert window.context.text() == "Design an RDW system"
        assert not window.stage_buttons[Stage.SETUP].isVisible()
        opened: list[tuple[str, object]] = []
        window.document.open_text = lambda text, base: opened.append((text, base))  # type: ignore[method-assign]
        window.rdw.text = "[recipe]\nformat = 1\n"
        window.rdw.recipe_changed.emit()
        assert window.primary.isEnabled()
        window.primary_clicked()
        assert opened == [("[recipe]\nformat = 1\n", None)]
        window.show_stage(Stage.HOME)
        assert not window.stage_buttons[Stage.SETUP].isHidden()
    finally:
        window.close()


# ------------------------------------------------------------------ a real kernel


def test_a_real_kernel_draws_the_space_and_reads_a_point(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(rdw_stage, "COARSE", 8)
    monkeypatch.setattr(rdw_stage, "FULL", 12)
    bridge = ServiceBridge()
    lost: list[bool] = []
    bridge.kernel_lost.connect(lambda: lost.append(True))
    try:
        with qtbot.waitSignal(bridge.replied, timeout=60_000):
            bridge.request("gases", {}, key="probe")
    except Exception as error:
        bridge.close()
        pytest.skip(f"the kernel cannot start in this environment: {error}")
    if lost:
        bridge.close()
        pytest.skip("the kernel cannot start in this environment: it was lost")
    made = RdwStage(bridge)
    qtbot.addWidget(made)
    try:
        made.show()
        qtbot.waitUntil(lambda: made.model is not None, timeout=WAIT_MS)
        qtbot.waitUntil(
            lambda: made.space is not None and made.space.points == 12,
            timeout=WAIT_MS,
        )
        assert made.space is not None and made.space.shape == (12, 12)
        assert made.plots.images[0].image is not None
        assert made.status.text() == ""
        made.pick(125e-6, 200e-6)
        qtbot.waitUntil(lambda: made.text is not None, timeout=WAIT_MS)
        assert made.readout_text("Total length") is not None
        assert made.readout_text("Phase-matched wavelength (check)") == "160 nm"
        assert "[recipe]" in (made.simulate_text() or "")
    finally:
        made.close()
        bridge.close()
