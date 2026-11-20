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
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLineEdit,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from solprop.service.messages import Reply

from solprop_gui.classes.rdw_inputs import (
    STARTS,
    DecodedSpace,
    RdwInputs,
    decode_point_reply,
    decode_space_reply,
)
from solprop_gui.registry.pages import GAS_TITLES
from solprop_gui.registry.units import pretty
from solprop_gui.ui.bridge import ServiceBridge
from solprop_gui.ui.quantity import QuantityField
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

_MAIN: Final = (
    ("wavelength_target", "Target RDW wavelength", "m", "nm"),
    ("wavelength_pump", "Pump wavelength", "m", "nm"),
    ("fwhm", "Pulse duration (FWHM)", "s", "fs"),
    ("max_length", "Largest system length", "m", "m"),
)
"""The main inputs: name, label, payload unit, unit shown first."""

_FACTORS: Final = (
    ("s_sf", "Self-focusing, S_sf"),
    ("s_ion", "Ionisation, S_ion"),
    ("s_fiss", "Fission length, S_fiss"),
    ("s_loss", "Loss length, S_loss"),
)
"""The safety factors: name and label."""

_SHOWN: Final = {
    "wavelength": "nm",
    "time": "fs",
    "energy_density": "J/cm^2",
    "pressure": "MPa",
    "index": "cm^2/W",
}
"""The unit a constraint field of each kind is shown in first (lengths: by name)."""

_ENDS: Final = (("in", "Input end"), ("out", "Output end"))

_READOUT_COLUMNS: Final = 3
"""How many (quantity, value) column pairs the readout has."""


def _empty_means(field: Mapping[str, object]) -> str:
    """Return what an empty optional field means, from the kernel's help text."""
    help_text = str(field["help"])
    if "empty:" in help_text:
        said = help_text.split("empty:", 1)[1].strip()
        for stop in (",", ".", ";"):
            said = said.split(stop, 1)[0]
        return said.strip()
    if field["alternative"] is not None:
        return f"none: give a {field['alternative']}"
    return "none"


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
    model : solprop_gui.classes.rdw_inputs.RdwInputs or None
        The typed inputs; ``None`` until the kernel's options arrive.
    plots : solprop_gui.views.rdw_plots.RdwPlots
        The four panels.
    readout : PySide6.QtWidgets.QTableWidget
        The picked point's numbers.
    status : PySide6.QtWidgets.QLabel
        Why the inputs or the kernel's answer failed; empty when nothing did.
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
        self.model: RdwInputs | None = None
        self.draw_ms = 0.0
        self.space: DecodedSpace | None = None
        self._options_id: int | None = None
        self._space_id: int | None = None
        self._space_points = 0
        self._point_id: int | None = None
        self._pick: tuple[float, float] | None = None
        self._full_axes: tuple[np.ndarray, np.ndarray] | None = None
        self._held: tuple[np.ndarray, np.ndarray] | None = None
        self._gases: list[str] = []
        self.fields: dict[str, QuantityField] = {}
        self.end_fields: dict[str, dict[str, QWidget]] = {"in": {}, "out": {}}
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
        self._panel = QWidget()
        self._panel.setObjectName("page")
        self._column = QVBoxLayout(self._panel)
        self._column.setContentsMargins(20, 16, 16, 16)
        self._column.setSpacing(6)
        self._column.addWidget(named_label("Asking solprop for the inputs…", "note"))
        scroll = QScrollArea()
        scroll.setObjectName("sheetScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(self._panel)
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

    def set_options(self, options: Mapping[str, object]) -> None:
        """Build the form from the kernel's ``rdw_space_options`` reply, and ask.

        Parameters
        ----------
        options : Mapping
            The reply.
        """
        self.model = RdwInputs(options)
        while self._column.count():
            item = self._column.takeAt(0)
            if item is not None and item.widget() is not None:
                item.widget().deleteLater()  # pyright: ignore[reportOptionalMemberAccess]
        self._build_form(options)
        self._edited()

    def _build_form(self, options: Mapping[str, object]) -> None:  # noqa: PLR0915 - a layout
        """Make the inputs' rows."""
        model = self.model
        assert model is not None  # set_options made it
        defaults: Mapping[str, object] = options["defaults"]  # pyright: ignore[reportAssignmentType]
        column = self._column
        column.addWidget(named_label("Design", "h3"))
        form = QFormLayout()
        form.setSpacing(6)
        self.gas = QComboBox()
        self.gas.setObjectName("rdw-gas")
        self.gas.currentIndexChanged.connect(self._gas_chosen)
        form.addRow("Gas", self.gas)
        self.set_gases(self._gases)
        for name, label, unit, shown in _MAIN:
            field = QuantityField(unit, shown, live=True)
            field.set_value(model.get(name))
            field.changed.connect(lambda v, n=name: self._set(n, v))
            field.refused.connect(self._say)
            form.addRow(label, field)
            self.fields[name] = field
        column.addLayout(form)
        for end, title in _ENDS:
            column.addSpacing(6)
            column.addWidget(named_label(title, "h3"))
            choice = QComboBox()
            choice.setObjectName(f"rdw-{end}-type")
            for entry in options["constraint_types"]:  # pyright: ignore[reportGeneralTypeIssues]
                choice.addItem(str(entry["label"]), str(entry["name"]))
            choice.setCurrentIndex(choice.findData(model.constraint_type(end)))
            choice.currentIndexChanged.connect(
                lambda _i, e=end, c=choice: self._type_chosen(e, str(c.currentData()))
            )
            column.addWidget(choice)
            holder = QWidget()
            holder.setObjectName("page")
            rows = QFormLayout(holder)
            rows.setContentsMargins(0, 0, 0, 0)
            rows.setSpacing(6)
            column.addWidget(holder)
            setattr(self, f"_{end}_choice", choice)
            setattr(self, f"_{end}_holder", holder)
            self._fill_end(end)
        column.addSpacing(6)
        column.addWidget(named_label("Safety factors", "h3"))
        factors = QFormLayout()
        factors.setSpacing(6)
        for name, label in _FACTORS:
            field = QuantityField(None, live=True)
            field.set_value(None, _plain(defaults.get(name)))
            field.changed.connect(lambda v, n=name: self._set(n, v))
            field.refused.connect(self._say)
            factors.addRow(label, field)
            self.fields[name] = field
        column.addLayout(factors)
        column.addSpacing(6)
        column.addWidget(named_label("Mode and axes", "h3"))
        mode = QFormLayout()
        mode.setSpacing(6)
        self.kind = QLineEdit()
        self.kind.setObjectName("rdw-kind")
        self.kind.setPlaceholderText(str(defaults.get("kind", "")))
        self.kind.textEdited.connect(lambda t: self._set("kind", t))
        mode.addRow("Mode kind", self.kind)
        for name in ("n", "m"):
            field = QuantityField(None, integer=True, live=True)
            field.set_value(None, _plain(defaults.get(name)))
            field.changed.connect(lambda v, n=name: self._set(n, v))
            field.refused.connect(self._say)
            mode.addRow(f"Mode {name}", field)
            self.fields[name] = field
        self.term = QComboBox()
        self.term.setObjectName("rdw-term")
        for entry in options["soliton_terms"]:  # pyright: ignore[reportGeneralTypeIssues]
            self.term.addItem(str(entry["label"]), str(entry["name"]))
        self.term.setCurrentIndex(self.term.findData(model.soliton_term))
        self.term.currentIndexChanged.connect(self._term_chosen)
        mode.addRow("Soliton term", self.term)
        column.addLayout(mode)
        self.log_a = QCheckBox("Logarithmic radius axis")
        self.log_e = QCheckBox("Logarithmic energy axis")
        self.hold = QCheckBox("Hold axes while editing")
        self.hold.setToolTip(
            "Keep the radius and energy axes of the last full plot, rather than "
            "letting solprop choose them for each design."
        )
        for flag, box in (("log_a", self.log_a), ("log_e", self.log_e)):
            box.toggled.connect(lambda on, f=flag: self._flag(f, on))
            column.addWidget(box)
        self.hold.toggled.connect(self._hold_toggled)
        column.addWidget(self.hold)
        column.addStretch(1)
        for box in self._panel.findChildren(QComboBox):
            _narrow(box)

    def _fill_end(self, end: str) -> None:
        """Put the rows of the constraint type chosen at an end in its form."""
        model = self.model
        assert model is not None  # the form exists only with a model
        holder: QWidget = getattr(self, f"_{end}_holder")
        form = holder.layout()
        assert isinstance(form, QFormLayout)
        while form.rowCount():
            form.removeRow(0)
        widgets: dict[str, QWidget] = {}
        presets = [str(p["name"]) for p in model.options["window_presets"]]  # pyright: ignore[reportGeneralTypeIssues]
        for field in model.constraint_fields(end):
            name = str(field["name"])
            text = model.constraint_value(end, name)
            widget = self._field_widget(end, field, text, presets)
            widget.setToolTip(
                str(field["help"])
                + (
                    " Type off, on, or a step such as 0.5 mm."
                    if field["kind"] == "rounding"
                    else ""
                )
            )
            form.addRow(str(field["label"]), widget)
            widgets[name] = widget
        self.end_fields[end] = widgets
        self._show_preset(end)

    def _field_widget(
        self, end: str, field: Mapping[str, object], text: str, presets: Sequence[str]
    ) -> QWidget:
        """Return the widget of one constraint field, showing `text`."""
        name, kind = str(field["name"]), str(field["kind"])
        if kind == "choice":
            box = QComboBox()
            box.addItem("No preset", "")
            for preset in presets:
                box.addItem(preset, preset)
            box.setCurrentIndex(max(box.findData(text), 0))
            _narrow(box)
            box.currentIndexChanged.connect(
                lambda _i, b=box: self._end_set(end, name, str(b.currentData()))
            )
            return box
        if kind == "rounding":
            line = QLineEdit(text)
            default = "on" if field["default"] is True else "off"
            line.setPlaceholderText(default)
            step_unit = str(field["unit"])
            line.textEdited.connect(
                lambda t: self._end_set(end, name, _rounding(t, step_unit))
            )
            return line
        unit = str(field["unit"])
        shown = _SHOWN.get(kind, "m" if name == "distance" else "mm")
        number = kind == "number" or unit == "1"
        quantity = QuantityField(
            None if number else unit, None if number else shown, live=True
        )
        quantity.set_value(
            text or None, None if field["optional_none"] else _plain(field["default"])
        )
        if field["optional_none"]:
            quantity.number.setPlaceholderText(_empty_means(field))
        elif field["required"]:
            quantity.number.setPlaceholderText("required")
        quantity.changed.connect(
            lambda v: self._end_set(end, name, "" if v is None else str(v))
        )
        quantity.refused.connect(self._say)
        return quantity

    def _show_preset(self, end: str) -> None:
        """Disable the rows a chosen preset supplies, and say so."""
        model = self.model
        assert model is not None
        preset = (
            model.constraint_value(end, "preset")
            if "preset" in self.end_fields[end]
            else ""
        )
        for field in model.constraint_fields(end):
            if field["alternative"] is None:
                continue
            widget = self.end_fields[end][str(field["name"])]
            widget.setEnabled(not preset)
            if isinstance(widget, QuantityField):
                widget.number.setPlaceholderText(
                    f"{preset} preset" if preset else _empty_means(field)
                )

    def set_gases(self, gases: Sequence[str]) -> None:
        """Take the gases solprop knows.

        Parameters
        ----------
        gases : Sequence of str
            Their names, from the kernel's ``gases`` reply.
        """
        self._gases = list(gases)
        if self.model is None:
            return
        self.gas.blockSignals(True)
        self.gas.clear()
        for gas in self._gases:
            self.gas.addItem(GAS_TITLES.get(gas, gas), gas)
        self.gas.setCurrentIndex(self.gas.findData(self.model.get("gas")))
        self.gas.blockSignals(False)

    # ------------------------------------------------------------------ the edits

    def _say(self, text: str) -> None:
        """Show a problem in the status line (empty clears it)."""
        self.status.setText(text)

    def _set(self, name: str, value: object) -> None:
        """Take a main input's new value."""
        if self.model is not None:
            self.model.set(name, "" if value is None else str(value))
            self._edited()

    def _end_set(self, end: str, name: str, text: str) -> None:
        """Take a constraint field's new text."""
        if self.model is not None:
            self.model.set_constraint_value(end, name, text)
            if name == "preset":
                self._show_preset(end)
            self._edited()

    def _type_chosen(self, end: str, kind: str) -> None:
        """Swap in the rows of a constraint type."""
        if self.model is not None:
            self.model.set_constraint_type(end, kind)
            self._fill_end(end)
            self._edited()

    def _gas_chosen(self, _index: int) -> None:
        """Take the gas chosen."""
        self._set("gas", self.gas.currentData() or STARTS["gas"])

    def _term_chosen(self, _index: int) -> None:
        """Take the soliton term chosen."""
        if self.model is not None:
            self.model.soliton_term = str(self.term.currentData())
            self._edited()

    def _flag(self, name: str, on: bool) -> None:
        """Take a logarithmic-axis flag."""
        if self.model is not None:
            self.model.set_flag(name, on)
            self._edited()

    def _hold_toggled(self, on: bool) -> None:
        """Hold the axes of the last full plot, or let solprop choose them again."""
        self._held = self._full_axes if on else None
        self._edited()

    def _edited(self) -> None:
        """Ask again: coarse once the editing rests briefly, full once it rests."""
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
            self._full_axes = (space.a, space.energy)
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
        recipe = decoded["recipe_text"]
        self.text = None if recipe is None else str(recipe)
        self.recipe_changed.emit()
        self.recipe_note.setText(
            "Simulate this opens this capillary as a new setup."
            if self.text is not None
            else "No design at the picked point (a masked cell): nothing to simulate."
        )

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


def _narrow(box: QComboBox) -> None:
    """Let a choice be narrower than its longest entry, which the list still shows."""
    box.setSizeAdjustPolicy(
        QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
    )
    box.setMinimumContentsLength(
        3 if box.parent() is not None and isinstance(box.parent(), QuantityField) else 8
    )


def _plain(value: object) -> object:
    """Return a default number as a person writes it (5, not 5.0)."""
    return f"{value:g}" if isinstance(value, float) else value


def _rounding(text: str, unit: str) -> str:
    """Return a rounding's text, a bare step taken in the field's own `unit`."""
    stripped = text.strip()
    try:
        float(stripped)
    except ValueError:
        return text
    return f"{stripped} {unit}"
