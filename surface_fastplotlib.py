"""XBT implied-volatility surface, drawn with fastplotlib.

Same surface, terminal colors, and floating sheets as main.py. Orbit and the
sheets move to the far side of the box so they stay behind the surface.

    uv run surface_fastplotlib.py
"""

import warnings

import numpy as np
import pylinalg as la
import pygfx

import fastplotlib as fpl

from main import (
    build_panes,
    build_surface,
    colormap,
    fmt_num,
    map_axis,
    nice_ticks,
    _point,
)

BG = "#0d1117"
PANE = (10 / 255, 15 / 255, 22 / 255, 1.0)
TICK = "#c9d1d9"
TITLE = "#ffffff"
GRID = (1.0, 1.0, 1.0, 0.45)
BORDER = (1.0, 1.0, 1.0, 1.0)
WIRE = (0.07, 0.07, 0.07, 1.0)

AZIM = -60.0
ELEV = 22.0


def eye_direction(azim: float, elev: float) -> np.ndarray:
    azim = np.deg2rad(azim)
    elev = np.deg2rad(elev)
    return np.array(
        [
            np.cos(azim) * np.cos(elev),
            np.sin(azim) * np.cos(elev),
            np.sin(elev),
        ],
        dtype=np.float64,
    )


def far_sides(eye, sticky, xlim, ylim, zlim):
    """Sheet on the box face farthest from the camera."""
    limits = (xlim, ylim, zlim)
    sticky = sticky or (xlim[0], ylim[1], zlim[0])
    sides = []
    for axis, (lo, hi) in enumerate(limits):
        if eye[axis] > 0.55:
            sides.append(lo)
        elif eye[axis] < -0.55:
            sides.append(hi)
        else:
            sides.append(sticky[axis])
    return tuple(sides)


def layout_labels(scene, x_at, y_at, z_at):
    """Tick labels on the open edge, opposite the sheet that would cover them."""
    xlim, ylim, zlim = scene["xlim"], scene["ylim"], scene["zlim"]
    y_near = ylim[0] if y_at == ylim[1] else ylim[1]
    x_near = xlim[0] if x_at == xlim[1] else xlim[1]
    z_out = -1.0 if z_at == zlim[0] else 1.0
    x_far_out = -1.0 if x_at == xlim[0] else 1.0
    y_near_out = -1.0 if y_near == ylim[0] else 1.0
    x_near_out = -1.0 if x_near == xlim[0] else 1.0
    labels = []

    def add(text, anchor, size):
        labels.append((text, np.asarray(anchor, dtype=np.float32), size))

    for value in (-1.0, -0.5, 0.0, 0.5, 1.0):
        anchor = _point(value, y_near, z_at, xlim, ylim, zlim)
        anchor += np.array([0.0, y_near_out * 0.05, z_out * 0.07], dtype=np.float32)
        add(fmt_num(value), anchor, 18)
    title = _point(0.0, y_near, z_at, xlim, ylim, zlim)
    title += np.array([0.0, y_near_out * 0.05, z_out * 0.26], dtype=np.float32)
    add("Delta (Put \u2190  \u2192 Call)", title, 20)

    last_y = None
    for years, name in zip(scene["tenor_years"][::2], scene["tenor_labels"][::2]):
        y = float(map_axis(years, ylim[0], ylim[1], 1.6))
        if last_y is not None and abs(y - last_y) < 0.16:
            continue
        last_y = y
        anchor = _point(x_near, years, z_at, xlim, ylim, zlim)
        anchor += np.array([x_near_out * 0.13, 0.0, z_out * 0.06], dtype=np.float32)
        add(name, anchor, 18)
    tenor = _point(x_near, (ylim[0] + ylim[1]) * 0.5, z_at, xlim, ylim, zlim)
    tenor += np.array([x_near_out * 0.24, 0.0, z_out * 0.06], dtype=np.float32)
    add("Tenor", tenor, 20)

    for value in nice_ticks(scene["iv_min"], scene["iv_max"], 5):
        anchor = _point(x_at, y_near, value, xlim, ylim, zlim)
        anchor += np.array([x_far_out * 0.11, y_near_out * 0.05, 0.0], dtype=np.float32)
        add(fmt_num(value), anchor, 18)
    z_title = _point(x_at, y_near, (scene["iv_min"] + scene["iv_max"]) * 0.5, xlim, ylim, zlim)
    z_title += np.array([x_far_out * 0.22, y_near_out * 0.02, 0.0], dtype=np.float32)
    add("Implied Vol (%)", z_title, 20)
    return labels


def colorbar_geometry(height=0.9, width=0.05, steps=48):
    vertices = []
    colors = []
    indices = []
    for step in range(steps):
        t0 = step / steps
        t1 = (step + 1) / steps
        y0 = -height / 2 + t0 * height
        y1 = -height / 2 + t1 * height
        c0 = (*colormap(t0), 1.0)
        c1 = (*colormap(t1), 1.0)
        base = len(vertices)
        vertices.extend(
            [
                [-width / 2, y0, 0.0],
                [width / 2, y0, 0.0],
                [width / 2, y1, 0.0],
                [-width / 2, y1, 0.0],
            ]
        )
        colors.extend([c0, c0, c1, c1])
        indices.extend([base, base + 1, base + 2, base, base + 2, base + 3])
    return (
        np.asarray(vertices, dtype=np.float32),
        np.asarray(indices, dtype=np.int32).reshape(-1, 3),
        np.asarray(colors, dtype=np.float32),
    )


def _basis(camera):
    forward = np.asarray(camera.world.forward, dtype=np.float64)
    forward = forward / np.linalg.norm(forward)
    up = np.asarray(camera.world.reference_up, dtype=np.float64)
    right = np.cross(forward, up)
    norm = np.linalg.norm(right)
    if norm < 1e-6:
        right = np.array([1.0, 0.0, 0.0])
    else:
        right = right / norm
    cam_up = np.cross(right, forward)
    cam_up = cam_up / np.linalg.norm(cam_up)
    return forward, right, cam_up


def _face_camera(forward, right, cam_up):
    # Local +z points back at the camera so the strip is face-on.
    rotation = np.column_stack([right, cam_up, -forward])
    return la.quat_from_mat(rotation)


class VolSurface:
    def __init__(self):
        self.scene = build_surface()
        self.xlim = self.scene["xlim"]
        self.ylim = self.scene["ylim"]
        self.zlim = self.scene["zlim"]
        self.sides = far_sides(eye_direction(AZIM, ELEV), None, self.xlim, self.ylim, self.zlim)

        with warnings.catch_warnings():
            # The default axes sample the camera before it has a size.
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            self.figure = fpl.Figure(size=(1430, 1040), cameras="3d", controller_types="orbit")
        self.plot = self.figure[0, 0]
        self.plot.background_color = (BG,)
        self.plot.title = ""
        self.plot.axes.auto_grid = False
        self.plot.axes.visible = False
        self.plot.tooltip.enabled = False

        self._add_surface()
        self._add_panes()
        self._add_labels()
        self._add_colorbar()
        self._keep_one_light()

        self.plot.camera.show_object(
            self.surface.world_object,
            tuple(-eye_direction(AZIM, ELEV)),
            up=(0, 0, 1),
            scale=1.7,
        )
        self.plot.controller.target = (0, 0, 0)
        self.plot.add_animations(self._on_frame)
        self._place_colorbar()

    def _add_surface(self):
        rows = self.scene["tenor_years"].size
        cols = self.scene["delta"].size
        xyz = self.scene["position"].reshape(rows, cols, 3)
        colors = np.concatenate(
            [
                self.scene["color"].reshape(rows, cols, 3),
                np.ones((rows, cols, 1), dtype=np.float32),
            ],
            axis=-1,
        )
        self.surface = self.plot.add_surface(
            xyz, mode="phong", colors=colors.reshape(-1, 4), center=False
        )
        self.surface.world_object.material.side = "both"

        normals = self.scene["normal"].reshape(rows, cols, 3)
        lifted = xyz + normals * 0.004
        wires = [lifted[j] for j in range(rows)] + [lifted[:, i] for i in range(cols)]
        self.plot.add_line_collection(wires, colors=WIRE, thickness=1.0, center=False)

    def _add_panes(self):
        fill, grids, borders = build_panes(self.xlim, self.ylim, self.zlim, *self.sides)
        self.panes = self.plot.add_mesh(
            fill,
            np.arange(fill.shape[0], dtype=np.int32).reshape(-1, 3),
            mode="basic",
            colors=PANE,
            center=False,
        )
        self.panes.world_object.material.side = "both"
        self.grids = self.plot.add_line_collection(
            [segment for segment in grids],
            colors=GRID,
            thickness=1.3,
            alpha_mode="blend",
            center=False,
        )
        self.borders = self.plot.add_line_collection(
            [segment for segment in borders],
            colors=BORDER,
            thickness=2.2,
            center=False,
        )

    def _add_labels(self):
        spec = layout_labels(self.scene, *self.sides)
        self.labels = [
            self.plot.add_text(
                text,
                font_size=size,
                face_color=TICK,
                offset=tuple(anchor),
                screen_space=True,
                anchor="middle-center",
                center=False,
            )
            for text, anchor, size in spec
        ]
        self.plot.add_text(
            "XBT Implied Volatility Surface",
            font_size=28,
            face_color=TITLE,
            offset=(0.0, 0.0, 0.62),
            screen_space=True,
            anchor="middle-center",
            center=False,
        )
        for graphic in self.labels:
            if graphic.text == "Implied Vol (%)":
                graphic.rotation = la.quat_from_axis_angle((0.0, 0.0, 1.0), np.pi / 2)

    def _add_colorbar(self):
        vertices, indices, colors = colorbar_geometry()
        self.bar = self.plot.add_mesh(
            vertices, indices, mode="basic", colors=colors, center=False
        )
        self.bar.world_object.material.side = "both"
        self.bar_ticks = []
        for value in nice_ticks(self.scene["iv_min"], self.scene["iv_max"], 5):
            self.bar_ticks.append(
                (
                    value,
                    self.plot.add_text(
                        fmt_num(value),
                        font_size=16,
                        face_color=TICK,
                        screen_space=True,
                        anchor="middle-left",
                        center=False,
                    ),
                )
            )
        self.bar_label = self.plot.add_text(
            "Implied Vol (%)",
            font_size=16,
            face_color=TICK,
            screen_space=True,
            anchor="middle-left",
            center=False,
        )

    def _keep_one_light(self):
        ambients = [obj for obj in self.plot.scene.children if isinstance(obj, pygfx.AmbientLight)]
        for extra in ambients[1:]:
            self.plot.scene.remove(extra)
        if ambients:
            ambients[0].intensity = 0.9
            self.plot._ambient_light = ambients[0]
        directions = [
            obj for obj in self.plot.camera.children if isinstance(obj, pygfx.DirectionalLight)
        ]
        for extra in directions[1:]:
            self.plot.camera.remove(extra)
        if directions:
            directions[0].intensity = 0.45
            self.plot._directional_light = directions[0]

    def _sync_panes(self, eye):
        sides = far_sides(eye, self.sides, self.xlim, self.ylim, self.zlim)
        if sides == self.sides:
            return
        self.sides = sides
        fill, grids, borders = build_panes(self.xlim, self.ylim, self.zlim, *sides)
        self.panes.positions[:] = fill
        for line, segment in zip(self.grids, grids):
            line.data = segment
        for line, segment in zip(self.borders, borders):
            line.data = segment
        for graphic, (_, anchor, _) in zip(self.labels, layout_labels(self.scene, *sides)):
            graphic.offset = tuple(float(v) for v in anchor)

    def _place_colorbar(self):
        forward, right, cam_up = _basis(self.plot.camera)
        target = np.asarray(self.plot.controller.target, dtype=np.float64)
        center = target + right * (self.plot.camera.width * 0.46)
        self.bar.world_object.local.position = center
        self.bar.world_object.local.rotation = _face_camera(forward, right, cam_up)
        height = 0.9
        for value, graphic in self.bar_ticks:
            t = (value - self.scene["iv_min"]) / (self.scene["iv_max"] - self.scene["iv_min"])
            along = -height / 2 + t * height
            graphic.offset = tuple(center + cam_up * along + right * 0.06)
        self.bar_label.offset = tuple(center + right * 0.16)

    def _on_frame(self):
        eye = np.asarray(self.plot.camera.world.position, dtype=np.float64)
        self._sync_panes(eye)
        self._place_colorbar()

    def show(self):
        self.figure.show(autoscale=False, axes_visible=False)
        fpl.loop.run()


if __name__ == "__main__":
    VolSurface().show()
