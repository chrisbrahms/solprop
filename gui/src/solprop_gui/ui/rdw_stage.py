# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Design an RDW system: the design space of a capillary, its region, and one point.

The inputs (gas, target and pump wavelengths, duration, length, the constraint at each
end, the safety factors) are on the left; the four ratio panels of the design space,
with the allowed region outlined, are on the right, and below them the numbers of a
picked point. The form is built from the kernel's ``rdw_space_options`` reply, so its
fields, units and defaults are solprop's.

Every edit asks again, live. After :data:`_DEBOUNCE_MS` of rest the stage asks for a
coarse grid (:data:`COARSE` points a side), and after :data:`_REST_MS` of rest for the
full one (:data:`FULL`), both under one latest-wins key, so the kernel drops a request
a newer one overtakes. Only the reply to the latest request is drawn; a failed reply
says why and leaves the last good plot. A picked point is read again, at the same
radius and energy, with each full request. *Simulate this* opens the picked point's
capillary as a new setup.
"""

import math
import time
from collections.abc import Mapping, Sequence
from typing import Final

import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from solprop.service.messages import Reply

from solprop_gui.classes.rdw_inputs import (
    DecodedSpace,
    RdwInputs,
    decode_point_reply,
    decode_space_reply,
)
from solprop_gui.registry.units import pretty
from solprop_gui.ui.bridge import ServiceBridge
from solprop_gui.ui.quantity import QuantityField
from solprop_gui.ui.rdw_form import RdwForm
from solprop_gui.ui.theme import named_label
from solprop_gui.views.rdw_plots import RdwPlots

__all__ = ["COARSE", "FULL", "RdwStage"]

_DEBOUNCE_MS: Final = 150
"""How long the editing rests before the coarse grid is asked for."""

_REST_MS: Final = 400
"""How long the editing rests before the full grid is asked for."""

COARSE: Final = 96
"""Grid points a side while editing."""

FULL: Final = 512
"""Grid points a side at rest."""

_SPACE_KEY: Final = "rdw-space"
_POINT_KEY: Final = "rdw-point"

_READOUT_COLUMNS: Final = 3
"""How many (quantity, value) column pairs the readout has."""


class RdwStage(QWidget):
    """Design an RDW system: inputs, the design space's four panels, a picked point.

    Parameters
    ----------
    bridge : solprop_gui.ui.bridge.ServiceBridge
        The kernel.
    parent : PySide6.QtWidgets.QWidget or None, optional
        The owner.

    Attributes
    ----------
    recipe_changed : Signal
        Emitted when :attr:`text` changes, so that *Simulate this* follows it.
    text : str or None
        The picked point's capillary recipe; ``None`` with no point, a masked one, or
        while it is being read again.
    form : solprop_gui.ui.rdw_form.RdwForm
        The inputs.
    plots : solprop_gui.views.rdw_plots.RdwPlots
        The four panels.
    readout : PySide6.QtWidgets.QTableWidget
        The picked point's numbers.
    status : PySide6.QtWidgets.QLabel
        Why the inputs or the kernel's answer failed; empty when nothing did.
    notes : PySide6.QtWidgets.QLabel
        Under the panels: the kernel's notes on the gas data (a gas law used outside
        the range of its source) for the space and the picked point; hidden when
        there are none. Not an error: the map is still drawn.
    busy : PySide6.QtWidgets.QLabel
        Shown while a request is unanswered.
    draw_ms : float
        How long the last space took to decode and draw, ms.
    """

    recipe_changed = Signal()

    def __init__(self, bridge: ServiceBridge, parent: QWidget | None = None) -> None:
        """Make the stage; the form is built when the kernel's options arrive."""
        super().__init__(parent)
        self.setObjectName("page")
        self.bridge = bridge
        self.text: str | None = None
        self.draw_ms = 0.0
        self.space: DecodedSpace | None = None
        self._options_id: int | None = None
        self._space_id: int | None = None
        self._space_points = 0
        self._point_id: int | None = None
        self._pick: tuple[float, float] | None = None
        self._full_axes: tuple[np.ndarray, np.ndarray] | None = None
        self._held: tuple[np.ndarray, np.ndarray] | None = None
        self._space_notes: tuple[str, ...] = ()
        self._point_notes: tuple[str, ...] = ()
        self._coarse = QTimer(self)
        self._coarse.setSingleShot(True)
        self._coarse.setInterval(_DEBOUNCE_MS)
        self._coarse.timeout.connect(lambda: self._ask_space(COARSE))
        self._rest = QTimer(self)
        self._rest.setSingleShot(True)
        self._rest.setInterval(_REST_MS)
        self._rest.timeout.connect(self._ask_full)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)
        self.form = RdwForm()
        self.form.edited.connect(self._edited)
        self.form.said.connect(self._say)
        self.form.hold_toggled.connect(self._hold_toggled)
        scroll = QScrollArea()
        scroll.setObjectName("sheetScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(self.form)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFixedWidth(440)
        row.addWidget(scroll)
        row.addWidget(self._make_right(), 1)
        bridge.replied.connect(self._on_reply)

    # ------------------------------------------------------------------ the layout

    def _make_right(self) -> QWidget:
        """Make the panels, the status line and the readout."""
        right = QWidget()
        right.setObjectName("page")
        column = QVBoxLayout(right)
        column.setContentsMargins(16, 12, 16, 12)
        column.setSpacing(6)
        top = QHBoxLayout()
        self.status = named_label("", "ready")
        self.status.setProperty("error", True)
        self.status.setWordWrap(True)
        top.addWidget(self.status, 1)
        self.busy = named_label("Updating…", "note")
        self.busy.hide()
        top.addWidget(self.busy)
        column.addLayout(top)
        self.plots = RdwPlots()
        self.plots.point_picked.connect(self.pick)
        column.addWidget(self.plots, 3)
        self.notes = named_label("", "pageNote")
        self.notes.setWordWrap(True)
        self.notes.hide()
        column.addWidget(self.notes)
        self.readout = QTableWidget(0, 2 * _READOUT_COLUMNS)
        self.readout.setObjectName("rdw-readout")
        self.readout.verticalHeader().hide()
        self.readout.setHorizontalHeaderLabels(
            ["Picked point", "Value"] * _READOUT_COLUMNS
        )
        self.readout.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.readout.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.readout.setShowGrid(False)
        self.readout.verticalHeader().setDefaultSectionSize(20)
        header = self.readout.horizontalHeader()
        for index in range(2 * _READOUT_COLUMNS):
            header.setSectionResizeMode(
                index,
                QHeaderView.ResizeMode.ResizeToContents
                if index % 2 == 0
                else QHeaderView.ResizeMode.Stretch,
            )
        column.addWidget(self.readout, 2)
        self.recipe_note = named_label(
            "Click a cell of a panel to read its design.", "pageNote"
        )
        self.recipe_note.setWordWrap(True)
        column.addWidget(self.recipe_note)
        return right

    # ------------------------------------------------------------------ the inputs

    @property
    def model(self) -> RdwInputs | None:  # numpydoc ignore=RT01
        """The typed inputs; ``None`` until the kernel's options arrive."""
        return self.form.model

    @property
    def fields(self) -> dict[str, QuantityField]:  # numpydoc ignore=RT01
        """The form's number boxes, by input name."""
        return self.form.fields

    @property
    def end_fields(self) -> dict[str, dict[str, QWidget]]:  # numpydoc ignore=RT01
        """The widgets of each end's constraint rows."""
        return self.form.end_fields

    @property
    def gas(self) -> QComboBox:  # numpydoc ignore=RT01
        """The gas."""
        return self.form.gas

    @property
    def hold(self) -> QCheckBox:  # numpydoc ignore=RT01
        """*Hold axes while editing*."""
        return self.form.hold

    @property
    def _out_choice(self) -> QComboBox:
        """The output end's constraint type."""
        return self.form.out_choice

    def set_options(self, options: Mapping[str, object]) -> None:
        """Build the form from the kernel's ``rdw_space_options`` reply, and ask.

        Parameters
        ----------
        options : Mapping
            The reply.
        """
        self.form.set_options(options)
        self._edited()

    def set_gases(self, gases: Sequence[str]) -> None:
        """Take the gases solprop knows.

        Parameters
        ----------
        gases : Sequence of str
            Their names, from the kernel's ``gases`` reply.
        """
        self.form.set_gases(gases)

    def _say(self, text: str) -> None:
        """Show a problem in the status line (empty clears it)."""
        self.status.setText(text)

    def _hold_toggled(self, on: bool) -> None:
        """Hold the axes of the last (or next) full plot, or let solprop choose."""
        self._held = self._full_axes if on else None
        self._edited()

    def _edited(self) -> None:
        """Ask again: coarse once the editing rests briefly, full once it rests."""
        # A point asked for at the old inputs no longer describes the form: its reply
        # is dropped, and the point is asked for again at rest.
        self._point_id = None
        self._show_busy()
        if self.text is not None:
            self.text = None
            self.recipe_changed.emit()
        self._coarse.start()
        self._rest.start()

    # ------------------------------------------------------------------ the kernel

    def _ask_space(self, points: int) -> None:
        """Ask for the design space at `points` a side (or on the held axes)."""
        model = self.model
        if model is None:
            return
        model.points = min(points, int(model.options["limits"]["points"]["max"]))  # pyright: ignore[reportIndexIssue, reportArgumentType, reportCallIssue]
        model.axes = self._axes_for(points)
        problem = model.validate()
        if problem is not None:
            self._say(problem)
            return
        self._say("")
        self._space_id = self.bridge.request(
            "rdw_space", model.to_payload(), key=_SPACE_KEY
        )
        self._space_points = points
        self._show_busy()

    def _axes_for(self, points: int) -> tuple[np.ndarray, np.ndarray] | None:
        """Return the held axes, every n-th value of them for a coarse grid."""
        if self._held is None:
            return None
        a, energy = self._held
        stride = max(1, math.ceil(max(a.size, energy.size) / points))
        return a[::stride], energy[::stride]

    def _ask_full(self) -> None:
        """Ask for the full grid, and the picked point at the new inputs."""
        self._ask_space(FULL)
        if self._pick is not None:
            self._ask_point()

    def pick(self, radius: float, energy: float) -> None:
        """Read the design at a point of the space.

        Parameters
        ----------
        radius : float
            Core radius, m.
        energy : float
            Pulse energy, J.
        """
        self._pick = (radius, energy)
        self.plots.set_point(radius, energy)
        self._ask_point()

    def _ask_point(self) -> None:
        """Ask for the picked point's numbers."""
        model = self.model
        if model is None or self._pick is None or model.validate() is not None:
            return
        self._point_id = self.bridge.request(
            "rdw_space_point", model.point_payload(*self._pick), key=_POINT_KEY
        )
        if self.text is not None:
            self.text = None
            self.recipe_changed.emit()
        self.recipe_note.setText("Reading the point…")
        self._show_busy()

    def _show_busy(self) -> None:
        """Show the busy mark while a request is unanswered."""
        self.busy.setVisible(self._space_id is not None or self._point_id is not None)

    def _on_reply(self, reply: Reply) -> None:
        """Take a reply, if it answers the stage's latest request of its kind."""
        if reply.superseded:
            return
        if reply.request_id == self._options_id:
            self._options_id = None
            if reply.ok:
                self.set_options(reply.value)
            else:
                self._say(f"solprop could not describe the inputs: {reply.error}")
        elif reply.request_id == self._space_id:
            self._space_id = None
            if reply.ok:
                self._show_space(reply.value)
            else:
                self._say(f"solprop could not map the design space: {reply.error}")
        elif reply.request_id == self._point_id:
            self._point_id = None
            if reply.ok:
                self._show_point(reply.value)
            else:
                self._say(f"solprop could not read the point: {reply.error}")
                self.recipe_note.setText("No design at the picked point.")
        self._show_busy()

    def request_options(self) -> None:
        """Ask the kernel what the design space accepts, once."""
        if self.model is None and self._options_id is None:
            self._options_id = self.bridge.request(
                "rdw_space_options", {}, key="rdw-options"
            )

    def showEvent(self, event: object) -> None:  # noqa: N802  # pyright: ignore[reportIncompatibleMethodOverride]
        """Ask for the inputs as the stage first comes up.

        Parameters
        ----------
        event : object
            The event.
        """
        super().showEvent(event)  # pyright: ignore[reportArgumentType]
        self.request_options()

    # ------------------------------------------------------------------ the replies

    def _show_space(self, value: Mapping[str, object]) -> None:
        """Decode and draw a design space, timing it."""
        started = time.perf_counter()
        space = decode_space_reply(value)
        self.space = space
        if self._space_points == FULL and self._held is None:
            self._full_axes = (np.array(space.a), np.array(space.energy))
            if self.model is not None and self.hold.isChecked():
                # Held before any full plot had come: hold this one's.
                self._held = self._full_axes
        self._space_notes = space.notes
        self._show_notes()
        self.plots.set_space(space)
        self.plots.set_message(
            "No allowed region for these inputs." if space.region_empty else None
        )
        self.draw_ms = (time.perf_counter() - started) * 1e3

    def _show_point(self, value: Mapping[str, object]) -> None:
        """Fill the readout with a point's numbers, and take its recipe."""
        decoded = decode_point_reply(value)
        rows: Sequence[Mapping[str, object]] = decoded["rows"]  # pyright: ignore[reportAssignmentType]
        per_column = math.ceil(len(rows) / _READOUT_COLUMNS)
        self.readout.clearContents()
        self.readout.setRowCount(per_column)
        for index, row in enumerate(rows):
            unit = pretty(str(row["unit"])) if row["unit"] else ""
            text = (
                f"{row['text']} {unit}".strip()
                if row["value"] is not None
                else (str(row["text"]))
            )
            column = 2 * (index // per_column)
            label = QTableWidgetItem(str(row["label"]))
            label.setToolTip(str(row["label"]))
            shown = QTableWidgetItem(text)
            shown.setToolTip(text)
            self.readout.setItem(index % per_column, column, label)
            self.readout.setItem(index % per_column, column + 1, shown)
        # Every row in view: the readout takes the height its rows need.
        frame = 2 * self.readout.frameWidth()
        self.readout.setMinimumHeight(
            self.readout.horizontalHeader().height()
            + per_column * self.readout.verticalHeader().defaultSectionSize()
            + frame
        )
        recipe = decoded["recipe_text"]
        self.text = None if recipe is None else str(recipe)
        self.recipe_changed.emit()
        self._point_notes = tuple(decoded["notes"])  # pyright: ignore[reportArgumentType]
        self._show_notes()
        self.recipe_note.setText(_point_note(decoded, self.text is not None))

    def _show_notes(self) -> None:
        """Show the gas-data notes of the space and the point, each once."""
        notes = list(dict.fromkeys((*self._space_notes, *self._point_notes)))
        self.notes.setText("Notes: " + " ".join(notes) if notes else "")
        self.notes.setToolTip("\n\n".join(notes))
        self.notes.setVisible(bool(notes))

    def readout_text(self, key: str) -> str | None:
        """Return what the readout shows for a row of the point reply.

        Parameters
        ----------
        key : str
            The row's label, as
            :func:`~solprop_gui.classes.rdw_inputs.decode_point_reply` gives it.

        Returns
        -------
        str or None
            The value shown beside it, ``None`` if the readout has no such row.
        """
        for row in range(self.readout.rowCount()):
            for column in range(0, self.readout.columnCount(), 2):
                label = self.readout.item(row, column)
                if label is not None and label.text() == key:
                    value = self.readout.item(row, column + 1)
                    return None if value is None else value.text()
        return None

    def simulate_text(self) -> str | None:
        """Return the recipe *Simulate this* opens: the picked point's capillary.

        Returns
        -------
        str or None
            The recipe text, ``None`` with nothing to simulate.
        """
        return self.text


def _point_note(decoded: Mapping[str, object], simulate: bool) -> str:
    """Return the sentence under the readout: whether the point is a design.

    - ``decoded``: :func:`~solprop_gui.classes.rdw_inputs.decode_point_reply`'s.
    - ``simulate``: whether there is a recipe to simulate.
    """
    if not simulate:
        return "No design at the picked point (a masked cell): nothing to simulate."
    parts: list[str] = []
    if decoded["allowed"] is False:
        outside: Sequence[str] = decoded["outside"]  # pyright: ignore[reportAssignmentType]
        why = ", ".join(outside) or "it is not a design (see the masks set)"
        parts.append(
            f"This point is outside the allowed region: {why}. Simulate this still "
            "opens it as a new setup, but the design's limits are not met."
        )
    else:
        parts.append("Simulate this opens this capillary as a new setup.")
    if decoded["recipe_note"] is not None:
        parts.append(str(decoded["recipe_note"]))
    return " ".join(parts)
