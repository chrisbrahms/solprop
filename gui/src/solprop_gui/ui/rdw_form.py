# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The RDW design stage's inputs: a form built from the kernel's options.

The rows (the gas, the target and pump wavelengths, the duration, the largest length,
the constraint at each end, the safety factors, the mode and the axes) come from the
kernel's ``rdw_space_options`` reply, so their fields, units and defaults are solprop's.
Each edit goes into an :class:`~solprop_gui.classes.rdw_inputs.RdwInputs` model, as
text, and is announced; the stage decides what to ask the kernel.
"""

from collections.abc import Mapping, Sequence
from typing import Final

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

from solprop_gui.classes.rdw_inputs import STARTS, RdwInputs
from solprop_gui.registry.pages import GAS_TITLES
from solprop_gui.ui.quantity import QuantityField
from solprop_gui.ui.theme import named_label

__all__ = ["RdwForm"]

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


class RdwForm(QWidget):
    """The inputs of an RDW design, as rows built from the kernel's options.

    Parameters
    ----------
    parent : PySide6.QtWidgets.QWidget or None, optional
        The owner.

    Attributes
    ----------
    edited : Signal
        Emitted after each edit that changes the inputs.
    said : Signal
        Emitted with a problem to show (a value that does not read).
    hold_toggled : Signal
        Emitted with the state of *Hold axes while editing*.
    model : solprop_gui.classes.rdw_inputs.RdwInputs or None
        The typed inputs; ``None`` until the options arrive.
    fields : dict
        The number boxes of the main inputs, safety factors and mode, by name.
    end_fields : dict
        The widgets of each end's constraint rows: ``{"in": {...}, "out": {...}}``.
    gas : PySide6.QtWidgets.QComboBox
        The gas (made with the rows).
    hold : PySide6.QtWidgets.QCheckBox
        *Hold axes while editing* (made with the rows).
    out_choice : PySide6.QtWidgets.QComboBox
        The output end's constraint type (made with the rows).
    """

    edited = Signal()
    said = Signal(str)
    hold_toggled = Signal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Make the form, empty until the options arrive."""
        super().__init__(parent)
        self.setObjectName("page")
        self.model: RdwInputs | None = None
        self._gases: list[str] = []
        self.fields: dict[str, QuantityField] = {}
        self.end_fields: dict[str, dict[str, QWidget]] = {"in": {}, "out": {}}
        self._column = QVBoxLayout(self)
        self._column.setContentsMargins(20, 16, 16, 16)
        self._column.setSpacing(6)
        self._column.addWidget(named_label("Asking solprop for the inputs…", "note"))

    def set_options(self, options: Mapping[str, object]) -> None:
        """Build the rows from the kernel's ``rdw_space_options`` reply.

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
            if end == "out":
                self.out_choice = choice
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
        self.hold.toggled.connect(self.hold_toggled)
        column.addWidget(self.hold)
        column.addStretch(1)
        for box in self.findChildren(QComboBox):
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

    def _say(self, text: str) -> None:
        """Say a problem (empty clears it)."""
        self.said.emit(text)

    def _set(self, name: str, value: object) -> None:
        """Take a main input's new value."""
        if self.model is not None:
            self.model.set(name, "" if value is None else str(value))
            self.edited.emit()

    def _end_set(self, end: str, name: str, text: str) -> None:
        """Take a constraint field's new text."""
        if self.model is not None:
            self.model.set_constraint_value(end, name, text)
            if name == "preset":
                self._show_preset(end)
            self.edited.emit()

    def _type_chosen(self, end: str, kind: str) -> None:
        """Swap in the rows of a constraint type."""
        if self.model is not None:
            self.model.set_constraint_type(end, kind)
            self._fill_end(end)
            self.edited.emit()

    def _gas_chosen(self, _index: int) -> None:
        """Take the gas chosen."""
        self._set("gas", self.gas.currentData() or STARTS["gas"])

    def _term_chosen(self, _index: int) -> None:
        """Take the soliton term chosen."""
        if self.model is not None:
            self.model.soliton_term = str(self.term.currentData())
            self.edited.emit()

    def _flag(self, name: str, on: bool) -> None:
        """Take a logarithmic-axis flag."""
        if self.model is not None:
            self.model.set_flag(name, on)
            self.edited.emit()


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
