# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The RDW design space's four panels: ratio maps on shared axes, and a picker.

Each panel is one design ratio (a value below 1 is inside its limit) on the grid of
core radius against pulse energy the kernel computed, coloured on a fixed range of 0
to 2. A masked cell (no solution: normal dispersion, no root, no soliton, beyond the
equation of state) has no ratio and is drawn in grey. The allowed region is outlined
in black on every panel, along the edges of its cells, so the outline shows exactly
which cells are allowed. A click on any panel picks the cell under it.

The grid's values are the cells' centres, as in HiSol. Every cell is drawn as it is
(nearest, no smoothing), so the cell a click lands in is the cell drawn there.

Nothing here computes physics: the panels colour, outline and place what the kernel
sent; the only arithmetic is display scaling (m to µm, J to µJ) and, on a logarithmic
axis, the base-10 logarithm the axis draws in.
"""

from typing import Final

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from solprop_gui.classes.rdw_inputs import DecodedSpace
from solprop_gui.ui.theme import AXIS, FONT, GREY, INK, named_label

__all__ = [
    "LEVELS",
    "MASKED",
    "PANELS",
    "SPECTRAL_R",
    "RdwPlots",
    "cell_at",
    "colours",
]

PANELS: Final = (
    ("loss", "Loss"),
    ("fiss", "Fission length"),
    ("nmin", "Minimum soliton order"),
    ("nmax", "Maximum soliton order"),
)
"""The panels, left to right: the decoded grid each draws, and its title."""

SPECTRAL_R: Final = (
    "#5e4fa2",
    "#3288bd",
    "#66c2a5",
    "#abdda4",
    "#e6f598",
    "#ffffbf",
    "#fee08b",
    "#fdae61",
    "#f46d43",
    "#d53e4f",
    "#9e0142",
)
"""ColorBrewer's 11-class Spectral scheme (Cynthia Brewer), reversed: HiSol's
``Spectral_r``. Blue is far inside a limit, pale yellow at it (a ratio of 1), dark red
at twice it."""

LEVELS: Final = (0.0, 2.0)
"""The fixed colour range of every panel."""

MASKED: Final = (150, 150, 150, 255)
"""The grey of a cell with no solution: no hue of the colour map comes near it."""

_LEFT_AXIS_WIDTH: Final = 56
"""Pixels of left margin on every panel: room for the first one's numbers and label."""

_LABEL: Final = {"color": GREY, "font-size": "11px", "font-family": FONT}


def _font(size: int = 11) -> QFont:
    """Return the panels' font at `size` pixels."""
    font = QFont(FONT)
    font.setPixelSize(size)
    return font


def _lookup() -> np.ndarray:
    """Return the 256-entry RGBA table of :data:`SPECTRAL_R`."""
    stops = np.linspace(0.0, 1.0, len(SPECTRAL_R))
    colour_map = pg.ColorMap(stops, [pg.mkColor(c) for c in SPECTRAL_R])
    return np.asarray(colour_map.getLookupTable(nPts=256, alpha=True), dtype=np.uint8)


_TABLE: Final = _lookup()


def colours(grid: np.ndarray) -> np.ndarray:
    """Return a ratio grid as RGBA pixels: the colour map on 0..2, grey where NaN.

    Parameters
    ----------
    grid : numpy.ndarray
        The ratios, shape ``(ne, na)``; NaN where masked, ``inf`` where no length is
        left (drawn as the top of the range).

    Returns
    -------
    numpy.ndarray
        ``uint8``, shape ``(ne, na, 4)``.
    """
    masked = np.isnan(grid)
    scaled = np.nan_to_num(grid, nan=0.0, posinf=LEVELS[1], neginf=LEVELS[0])
    index = np.clip(scaled * (255.0 / LEVELS[1]), 0.0, 255.0).astype(np.uint8)
    out = _TABLE[index]
    out[masked] = MASKED
    return out


class _Axis:
    """Where a grid axis's cells lie, in the coordinates a panel draws in.

    Linear axes draw in the display unit; a logarithmic one (the kernel spaced the
    values evenly in their logarithm) in its base-10 logarithm, which the axis labels
    as powers of ten.
    """

    def __init__(self, values: np.ndarray) -> None:
        self.values = values
        self.log = _evenly_logarithmic(values)
        drawn = np.log10(values) if self.log else values
        self.start = float(drawn[0])
        self.step = float(drawn[1] - drawn[0]) if drawn.size > 1 else 1.0

    def position(self, value: float) -> float:
        """Return where `value` (display unit) is drawn."""
        return float(np.log10(value)) if self.log else value

    def edges(self) -> tuple[float, float]:
        """Return the outer edges of the first and last cells, as drawn."""
        return (
            self.start - 0.5 * self.step,
            self.start + (self.values.size - 0.5) * self.step,
        )

    def index(self, drawn: float) -> int | None:
        """Return the cell whose extent holds a drawn position, ``None`` outside."""
        found = int(np.floor((drawn - self.start) / self.step + 0.5))
        return found if 0 <= found < self.values.size else None


def _evenly_logarithmic(values: np.ndarray) -> bool:
    """Return whether `values` are evenly spaced in their logarithm but not linearly."""
    if values.size < 3 or np.any(values <= 0):
        return False
    steps = np.diff(values)
    if np.allclose(steps, steps[0], rtol=1e-6, atol=0.0):
        return False
    log_steps = np.diff(np.log10(values))
    return bool(np.allclose(log_steps, log_steps[0], rtol=1e-6, atol=0.0))


def cell_at(space: DecodedSpace, x: float, y: float) -> tuple[int, int] | None:
    """Return the cell (row, column) drawn at a panel position, ``None`` outside.

    Parameters
    ----------
    space : DecodedSpace
        The space drawn.
    x, y : float
        The position, in the panels' coordinates (µm and µJ, or their base-10
        logarithm on a logarithmic axis).

    Returns
    -------
    tuple of int or None
        ``(row, column)``: row = energy index, column = radius index.
    """
    column = _Axis(space.a_um).index(x)
    row = _Axis(space.energy_uj).index(y)
    if row is None or column is None:
        return None
    return row, column


def _outline(allowed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the edges between allowed and other cells, as pairs of index points.

    Cell ``(i, j)`` spans ``j ± 0.5`` across and ``i ± 0.5`` up, in index units; the
    grid's border counts as not allowed, so a region touching it is closed.
    """
    padded = np.pad(allowed, 1)
    rows, columns = np.nonzero(padded[:, 1:] != padded[:, :-1])
    x_v = np.repeat(columns - 0.5, 2)
    y_v = np.column_stack((rows - 1.5, rows - 0.5)).ravel()
    rows, columns = np.nonzero(padded[1:, :] != padded[:-1, :])
    x_h = np.column_stack((columns - 1.5, columns - 0.5)).ravel()
    y_h = np.repeat(rows - 0.5, 2)
    return np.concatenate((x_v, x_h)), np.concatenate((y_v, y_h))


class RdwPlots(QWidget):
    """The four ratio panels, on shared radius and energy axes, with a colour bar.

    Parameters
    ----------
    parent : PySide6.QtWidgets.QWidget or None, optional
        The owner.

    Attributes
    ----------
    point_picked : Signal
        Emitted with the picked cell's radius (m) and energy (J) when a panel is
        clicked.
    space : DecodedSpace or None
        The space drawn.
    plots : list of pyqtgraph.PlotItem
        The panels, in the order of :data:`PANELS`.
    images : list of pyqtgraph.ImageItem
        Their images.
    outlines : list of pyqtgraph.PlotDataItem
        The allowed region's outline on each.
    message : PySide6.QtWidgets.QLabel
        What the panels say over themselves (no region, an error), hidden when empty.
    hover : PySide6.QtWidgets.QLabel
        The values under the pointer.
    """

    point_picked = Signal(float, float)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Make the panels, empty."""
        super().__init__(parent)
        self.setObjectName("page")
        self.space: DecodedSpace | None = None
        self._point: tuple[float, float] | None = None
        self._axes: tuple[_Axis, _Axis] | None = None
        self._extent: tuple[float, ...] | None = None
        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(4)
        self.message = named_label("", "ready")
        self.message.setProperty("warn", True)
        self.message.setWordWrap(True)
        self.message.hide()
        column.addWidget(self.message)
        self.view = pg.GraphicsLayoutWidget()
        self.view.setBackground("w")
        self.view.ci.layout.setContentsMargins(0, 4, 0, 0)  # pyright: ignore[reportAttributeAccessIssue]
        self.view.ci.layout.setHorizontalSpacing(6)  # pyright: ignore[reportAttributeAccessIssue]
        self.view.setMinimumHeight(240)
        column.addWidget(self.view, 1)
        for column_index in range(len(PANELS)):
            # Equal stretch: otherwise a column's share follows its title's width.
            self.view.ci.layout.setColumnStretchFactor(column_index, 1)  # pyright: ignore[reportAttributeAccessIssue]
        self.plots: list[pg.PlotItem] = []
        self.images: list[pg.ImageItem] = []
        self.outlines: list[pg.PlotDataItem] = []
        self.markers: list[tuple[pg.ScatterPlotItem, pg.ScatterPlotItem]] = []
        for index, (_key, title) in enumerate(PANELS):
            self._add_panel(index, title)
        self.bar = pg.ColorBarItem(
            values=LEVELS,
            colorMap=pg.ColorMap(
                np.linspace(0.0, 1.0, len(SPECTRAL_R)),
                [pg.mkColor(c) for c in SPECTRAL_R],
            ),
            width=12,
            interactive=False,
            colorMapMenu=False,
        )
        axis = self.bar.axis
        axis.setPen(pg.mkPen(AXIS))
        axis.setTextPen(pg.mkPen(GREY))
        axis.setTickFont(_font())
        axis.setLabel("Ratio to limit", **_LABEL)
        self.view.ci.addItem(self.bar, row=0, col=len(PANELS))
        self.caption = QLabel(
            "Each panel: the design's value over its limit; below 1 is inside it. "
            f'<span style="color: rgb{MASKED[:3]};">■</span> Grey: no solution '
            "(normal dispersion, no root, no soliton, or beyond the equation of "
            "state). <b>Black outline</b>: the allowed region, where all four are "
            "below 1. Click a cell to read it."
        )
        self.caption.setObjectName("pageNote")
        self.caption.setWordWrap(True)
        column.addWidget(self.caption)
        self.hover = named_label("", "pageNote")
        column.addWidget(self.hover)
        scene = self.view.scene()
        scene.sigMouseClicked.connect(self._clicked)  # pyright: ignore[reportAttributeAccessIssue, reportOptionalMemberAccess]
        scene.sigMouseMoved.connect(self._moved)  # pyright: ignore[reportAttributeAccessIssue, reportOptionalMemberAccess]

    def _add_panel(self, index: int, title: str) -> None:
        """Make one panel, linked to the first."""
        plot = self.view.ci.addPlot(row=0, col=index)
        plot.setTitle(title, color=INK, size="10pt")
        plot.setPreferredWidth(300.0)  # the same for all, so the stretch shares equally
        plot.hideButtons()
        plot.setMenuEnabled(False)
        box = plot.getViewBox()
        box.setDefaultPadding(0.0)
        for side in ("left", "bottom", "top", "right"):
            plot.showAxis(side)
            axis = plot.getAxis(side)
            axis.setPen(pg.mkPen(AXIS))
            axis.setTextPen(pg.mkPen(GREY))
            axis.setTickFont(_font())
            axis.enableAutoSIPrefix(False)
            if side in ("top", "right"):
                axis.setStyle(showValues=False, tickLength=0)
            else:
                axis.setStyle(tickLength=-4, tickTextOffset=3)
        plot.getAxis("bottom").setLabel("Core radius (µm)", **_LABEL)
        if index == 0:
            plot.getAxis("left").setLabel("Pulse energy (µJ)", **_LABEL)
        else:
            plot.getAxis("left").setStyle(showValues=False)
            box.setXLink(self.plots[0].getViewBox())
            box.setYLink(self.plots[0].getViewBox())
        # The same left margin on every panel, so the four plot areas are equally wide:
        # only the first shows numbers and a label, the others keep the room blank.
        plot.getAxis("left").setWidth(_LEFT_AXIS_WIDTH)
        image = pg.ImageItem(axisOrder="row-major")
        plot.addItem(image)
        outline = pg.PlotDataItem(connect="pairs", pen=pg.mkPen("k", width=1.6))
        outline.setZValue(20)
        plot.addItem(outline)
        halo = pg.ScatterPlotItem(symbol="+", size=17, pen=pg.mkPen("w", width=4.5))
        cross = pg.ScatterPlotItem(symbol="+", size=15, pen=pg.mkPen("k", width=2))
        for item, z in ((halo, 30), (cross, 31)):
            item.setZValue(z)
            plot.addItem(item)
        self.plots.append(plot)
        self.images.append(image)
        self.outlines.append(outline)
        self.markers.append((halo, cross))

    # ------------------------------------------------------------------ drawing

    def set_space(self, space: DecodedSpace) -> None:
        """Draw a design space.

        Parameters
        ----------
        space : DecodedSpace
            The decoded ``rdw_space`` reply.
        """
        self.space = space
        x_axis, y_axis = _Axis(space.a_um), _Axis(space.energy_uj)
        self._axes = (x_axis, y_axis)
        (x0, x1), (y0, y1) = x_axis.edges(), y_axis.edges()
        rect = QRectF(x0, y0, x1 - x0, y1 - y0)
        index_x, index_y = _outline(space.allowed)
        outline_x = x_axis.start + index_x * x_axis.step
        outline_y = y_axis.start + index_y * y_axis.step
        for (key, _title), image, outline in zip(
            PANELS, self.images, self.outlines, strict=True
        ):
            image.setImage(colours(getattr(space, key)), autoLevels=False)
            image.setRect(rect)
            outline.setData(outline_x, outline_y)
        for plot in self.plots:
            plot.getAxis("bottom").setLogMode(x_axis.log)
            plot.getAxis("left").setLogMode(y_axis.log)
        first = self.plots[0].getViewBox()
        first.setLimits(xMin=x0, xMax=x1, yMin=y0, yMax=y1)
        # A new view only for new ends of the grid: the coarse and the full grid of
        # one design share them, so a zoom survives the edits that keep them.
        ends = (
            float(space.a[0]),
            float(space.a[-1]),
            float(space.energy[0]),
            float(space.energy[-1]),
            float(x_axis.log),
            float(y_axis.log),
        )
        if self._extent is None or not np.allclose(ends, self._extent, rtol=0.02):
            self._extent = ends
            first.setRange(xRange=(x0, x1), yRange=(y0, y1), padding=0.0)
        self._place_marker()

    def set_point(self, radius: float | None, energy: float | None = None) -> None:
        """Mark the picked point on every panel, or clear the mark.

        Parameters
        ----------
        radius : float or None
            Core radius in m; ``None`` clears the mark.
        energy : float or None, optional
            Pulse energy in J.
        """
        self._point = (
            None if radius is None or energy is None else (radius * 1e6, energy * 1e6)
        )
        self._place_marker()

    def _place_marker(self) -> None:
        """Put the cross at the picked point, as each axis draws it."""
        if self._point is None or self._axes is None:
            for halo, cross in self.markers:
                halo.setData([], [])
                cross.setData([], [])
            return
        x = self._axes[0].position(self._point[0])
        y = self._axes[1].position(self._point[1])
        for halo, cross in self.markers:
            halo.setData([x], [y])
            cross.setData([x], [y])

    def set_message(self, text: str | None) -> None:
        """Say something over the panels, or nothing.

        Parameters
        ----------
        text : str or None
            What; ``None`` or empty hides the line.
        """
        self.message.setText(text or "")
        self.message.setVisible(bool(text))

    # ------------------------------------------------------------------ picking

    def _panel_point(self, scene_position: QPointF) -> QPointF | None:
        """Return a scene point in the panels' coordinates, ``None`` off them."""
        for plot in self.plots:
            box = plot.getViewBox()
            if box.sceneBoundingRect().contains(scene_position):
                return box.mapSceneToView(scene_position)
        return None

    def pick_at(self, x: float, y: float) -> bool:
        """Pick the cell drawn at a panel position, as a click there does.

        Parameters
        ----------
        x, y : float
            The position, in the panels' coordinates.

        Returns
        -------
        bool
            Whether a cell is there; if so :attr:`point_picked` was emitted with its
            radius (m) and energy (J).
        """
        if self.space is None:
            return False
        found = cell_at(self.space, x, y)
        if found is None:
            return False
        row, column = found
        self.point_picked.emit(
            float(self.space.a[column]), float(self.space.energy[row])
        )
        return True

    def _clicked(self, event: object) -> None:
        """Pick the cell under a left click."""
        if event.double() or event.button() != Qt.MouseButton.LeftButton:  # pyright: ignore[reportAttributeAccessIssue]
            return
        point = self._panel_point(event.scenePos())  # pyright: ignore[reportAttributeAccessIssue]
        if point is not None:
            self.pick_at(point.x(), point.y())

    def _moved(self, scene_position: QPointF) -> None:
        """Say the four ratios of the cell under the pointer."""
        point = self._panel_point(scene_position)
        space = self.space
        found = (
            None
            if point is None or space is None
            else cell_at(space, point.x(), point.y())
        )
        if space is None or found is None:
            self.hover.setText("")
            return
        row, column = found
        ratios = ", ".join(
            f"{title.lower()} {_ratio(float(getattr(space, key)[row, column]))}"
            for key, title in PANELS
        )
        allowed = "allowed" if space.allowed[row, column] else "not allowed"
        self.hover.setText(
            f"{space.a_um[column]:.4g} µm, {space.energy_uj[row]:.4g} µJ: "
            f"{ratios} ({allowed})"
        )


def _ratio(value: float) -> str:
    """Return a ratio as the hover line says it."""
    if np.isnan(value):
        return "—"
    return "∞" if np.isinf(value) else f"{value:.3g}"
