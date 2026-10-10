import numpy as np
from PIL import Image, ImageDraw, ImageFont

import moderngl
import moderngl_window as mglw

# Terminal palette from the matplotlib chart.
BG = (13 / 255, 17 / 255, 23 / 255, 1.0)
PANE = np.array([10, 15, 22], dtype=np.float32) / 255.0
TICK_RGBA = (201, 209, 217, 255)
TITLE_RGBA = (255, 255, 255, 255)
HINT_RGBA = (139, 148, 158, 255)
GRID_RGBA = (1.0, 1.0, 1.0, 0.45)
BORDER_RGBA = (1.0, 1.0, 1.0, 1.0)
WIRE_RGBA = (0.07, 0.07, 0.07, 1.0)

BOX = np.array([1.3, 1.6, 0.8], dtype=np.float64)
STOPS = np.array(
    [
        [0x2F, 0xAE, 0x4A],
        [0xA8, 0xE6, 0x3A],
        [0xF7, 0xE0, 0x17],
        [0xF2, 0x8C, 0x1C],
        [0xE6, 0x22, 0x1F],
    ],
    dtype=np.float64,
) / 255.0

FONT = "/usr/share/fonts/liberation-sans-fonts/LiberationSans-Regular.ttf"
FONT_BOLD = "/usr/share/fonts/liberation-sans-fonts/LiberationSans-Bold.ttf"
_FONTS: dict[tuple[int, bool], ImageFont.FreeTypeFont] = {}


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    key = (size, bold)
    if key not in _FONTS:
        _FONTS[key] = ImageFont.truetype(FONT_BOLD if bold else FONT, size)
    return _FONTS[key]


def colormap(t: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(t, dtype=np.float64), 0.0, 1.0) * (len(STOPS) - 1)
    i = np.minimum(np.floor(x).astype(np.int32), len(STOPS) - 2)
    f = (x - i)[..., None]
    return STOPS[i] * (1.0 - f) + STOPS[i + 1] * f


def nice_ticks(lo: float, hi: float, count: int = 5) -> list[float]:
    span = hi - lo
    if span <= 0:
        return [lo]
    raw = span / max(count - 1, 1)
    mag = 10 ** np.floor(np.log10(raw))
    step = mag
    for mult in (1, 2, 2.5, 5, 10):
        if raw <= mult * mag + 1e-9:
            step = mult * mag
            break
    start = np.ceil((lo - 1e-9) / step) * step
    ticks = []
    value = float(start)
    while value <= hi + step * 1e-6:
        if lo - step * 1e-6 <= value <= hi + step * 1e-6:
            ticks.append(value)
        value += float(step)
    return ticks


def fmt_num(value: float) -> str:
    rounded = round(value)
    if abs(value - rounded) < 1e-6:
        return str(int(rounded))
    return f"{value:.1f}"


def map_axis(values, lo, hi, aspect):
    return ((np.asarray(values, dtype=np.float64) - lo) / (hi - lo) - 0.5) * aspect


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    cam_up = np.cross(right, forward)
    view = np.eye(4, dtype=np.float64)
    view[0, :3] = right
    view[1, :3] = cam_up
    view[2, :3] = -forward
    view[0, 3] = -np.dot(right, eye)
    view[1, 3] = -np.dot(cam_up, eye)
    view[2, 3] = np.dot(forward, eye)
    return view


def ortho(left, right, bottom, top, near, far) -> np.ndarray:
    proj = np.zeros((4, 4), dtype=np.float64)
    proj[0, 0] = 2.0 / (right - left)
    proj[1, 1] = 2.0 / (top - bottom)
    proj[2, 2] = -2.0 / (far - near)
    proj[0, 3] = -(right + left) / (right - left)
    proj[1, 3] = -(top + bottom) / (top - bottom)
    proj[2, 3] = -(far + near) / (far - near)
    proj[3, 3] = 1.0
    return proj


def build_surface():
    delta = np.linspace(-1.0, 1.0, 41)
    tenor_labels = ["1W", "2W", "1M", "2M", "3M", "6M", "9M", "1Y", "18M", "2Y", "3Y", "5Y", "10Y"]
    tenor_years = np.array([7, 14, 30, 60, 91, 182, 273, 365, 547, 730, 1095, 1825, 3650]) / 365.0

    delta_grid, tenor_grid = np.meshgrid(delta, tenor_years)
    smile = 8.0 * delta_grid**2 - 3.0 * delta_grid
    base_term = 42 + 10 * np.exp(-tenor_grid * 6) + 4 * np.log1p(tenor_grid)
    flatten = np.exp(-tenor_grid * 0.8)
    iv = base_term + smile * (0.35 + 0.65 * flatten)
    iv = iv + np.random.default_rng(7).normal(0, 0.15, iv.shape)

    x_pad = (delta.max() - delta.min()) * 0.35
    y_pad = (tenor_years.max() - tenor_years.min()) * 0.35
    z_pad_low = (iv.max() - iv.min()) * 0.45
    z_pad_high = (iv.max() - iv.min()) * 0.20
    xlim = (float(delta.min() - x_pad), float(delta.max() + x_pad))
    ylim = (float(tenor_years.min() - y_pad), float(tenor_years.max() + y_pad))
    zlim = (float(iv.min() - z_pad_low), float(iv.max() + z_pad_high))

    x = map_axis(delta_grid, xlim[0], xlim[1], BOX[0])
    y = map_axis(tenor_grid, ylim[0], ylim[1], BOX[1])
    z = map_axis(iv, zlim[0], zlim[1], BOX[2])

    dx, dy, dz = np.gradient(x, axis=1), np.gradient(y, axis=1), np.gradient(z, axis=1)
    ex, ey, ez = np.gradient(x, axis=0), np.gradient(y, axis=0), np.gradient(z, axis=0)
    normals = np.cross(np.stack([dx, dy, dz], axis=-1), np.stack([ex, ey, ez], axis=-1))
    length = np.linalg.norm(normals, axis=-1, keepdims=True)
    length[length < 1e-8] = 1.0
    normals = normals / length

    colors = colormap((iv - iv.min()) / (iv.max() - iv.min()))
    position = np.stack([x, y, z], axis=-1).reshape(-1, 3).astype(np.float32)
    rows, cols = iv.shape
    jj, ii = np.mgrid[0 : rows - 1, 0 : cols - 1]
    a = (jj * cols + ii).ravel()
    b = a + 1
    c = a + cols
    d = c + 1
    indices = np.stack([a, b, d, a, d, c], axis=1).astype(np.int32).ravel()

    along_x = np.stack(
        [
            np.stack([x[:, :-1], y[:, :-1], z[:, :-1]], axis=-1),
            np.stack([x[:, 1:], y[:, 1:], z[:, 1:]], axis=-1),
        ],
        axis=2,
    ).reshape(-1, 2, 3)
    along_y = np.stack(
        [
            np.stack([x[:-1, :], y[:-1, :], z[:-1, :]], axis=-1),
            np.stack([x[1:, :], y[1:, :], z[1:, :]], axis=-1),
        ],
        axis=2,
    ).reshape(-1, 2, 3)
    wires = np.concatenate([along_x, along_y], axis=0)

    return {
        "position": position,
        "normal": normals.reshape(-1, 3).astype(np.float32),
        "color": colors.reshape(-1, 3).astype(np.float32),
        "indices": indices,
        "wires": wires.astype(np.float32),
        "xlim": xlim,
        "ylim": ylim,
        "zlim": zlim,
        "iv_min": float(iv.min()),
        "iv_max": float(iv.max()),
        "delta": delta,
        "tenor_years": tenor_years,
        "tenor_labels": tenor_labels,
    }


def _point(x, y, z, xlim, ylim, zlim) -> np.ndarray:
    return np.array(
        [
            map_axis(x, xlim[0], xlim[1], BOX[0]),
            map_axis(y, ylim[0], ylim[1], BOX[1]),
            map_axis(z, zlim[0], zlim[1], BOX[2]),
        ],
        dtype=np.float32,
    )


def build_panes(xlim, ylim, zlim, x_at, y_at, z_at):
    mx = (xlim[1] - xlim[0]) * 0.12
    my = (ylim[1] - ylim[0]) * 0.12
    mz = (zlim[1] - zlim[0]) * 0.12
    n_grid = 7
    fills = []
    grids = []
    borders = []

    # Each sheet sits on one far face of the box. The face is chosen per frame.
    specs = (
        ("y", y_at, xlim, zlim, mx, mz),
        ("x", x_at, ylim, zlim, my, mz),
        ("z", z_at, xlim, ylim, mx, my),
    )
    for const_axis, const_val, u_lim, v_lim, u_margin, v_margin in specs:
        us = np.linspace(u_lim[0] + u_margin, u_lim[1] - u_margin, n_grid)
        vs = np.linspace(v_lim[0] + v_margin, v_lim[1] - v_margin, n_grid)

        def at(u, v, _axis=const_axis, _value=const_val):
            coords = {"x": _value, "y": _value, "z": _value}
            order = {"y": ("x", "z"), "x": ("y", "z"), "z": ("x", "y")}[_axis]
            coords[order[0]] = u
            coords[order[1]] = v
            return _point(coords["x"], coords["y"], coords["z"], xlim, ylim, zlim)

        corners = [at(us[0], vs[0]), at(us[-1], vs[0]), at(us[-1], vs[-1]), at(us[0], vs[-1])]
        fills.extend([corners[0], corners[1], corners[2], corners[0], corners[2], corners[3]])
        for u in us:
            grids.append((at(u, vs[0]), at(u, vs[-1])))
        for v in vs:
            grids.append((at(us[0], v), at(us[-1], v)))
        loop = corners + [corners[0]]
        borders.extend(zip(loop[:-1], loop[1:]))

    fill = np.asarray(fills, dtype=np.float32)
    return fill, np.asarray(grids, dtype=np.float32), np.asarray(borders, dtype=np.float32)


def paint_lines(segments: np.ndarray, color) -> np.ndarray:
    count = segments.shape[0] * 2
    colors = np.tile(np.asarray(color, dtype=np.float32), (count, 1))
    return np.concatenate([segments.reshape(-1, 3), colors], axis=1).astype(np.float32)


def text_image(text: str, size: int, color, bold: bool = False, rotate: int = 0) -> Image.Image:
    font = _font(size, bold)
    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    box = probe.textbbox((0, 0), text, font=font)
    image = Image.new("RGBA", (box[2] - box[0] + 4, box[3] - box[1] + 4), (0, 0, 0, 0))
    ImageDraw.Draw(image).text((2 - box[0], 2 - box[1]), text, font=font, fill=color)
    if rotate:
        image = image.rotate(rotate, expand=True, resample=Image.Resampling.BICUBIC)
    return image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)


SURFACE_VS = """
#version 330
uniform mat4 u_mvp;
in vec3 in_pos;
in vec3 in_normal;
in vec3 in_color;
out vec3 v_color;
out vec3 v_normal;
out vec3 v_world;
void main() {
    v_color = in_color;
    v_normal = in_normal;
    v_world = in_pos;
    gl_Position = u_mvp * vec4(in_pos, 1.0);
}
"""

SURFACE_FS = """
#version 330
uniform vec3 u_light;
uniform vec3 u_eye;
in vec3 v_color;
in vec3 v_normal;
in vec3 v_world;
out vec4 f_color;
void main() {
    vec3 n = normalize(v_normal);
    vec3 view_dir = normalize(u_eye - v_world);
    if (dot(n, view_dir) < 0.0) n = -n;
    float lambert = clamp(dot(n, normalize(u_light)), 0.0, 1.0);
    float shade = 0.64 + 0.36 * lambert;
    f_color = vec4(v_color * shade, 1.0);
}
"""

LINE_VS = """
#version 330
uniform mat4 u_mvp;
uniform float u_bias;
in vec3 in_pos;
in vec4 in_color;
out vec4 v_color;
void main() {
    v_color = in_color;
    gl_Position = u_mvp * vec4(in_pos, 1.0);
    gl_Position.z -= u_bias * gl_Position.w;
}
"""

LINE_GS = """
#version 330
layout(lines) in;
layout(triangle_strip, max_vertices = 4) out;
uniform vec2 u_viewport;
uniform float u_thickness;
in vec4 v_color[];
out vec4 g_color;
void main() {
    vec4 p0 = gl_in[0].gl_Position;
    vec4 p1 = gl_in[1].gl_Position;
    vec2 s0 = p0.xy / p0.w;
    vec2 s1 = p1.xy / p1.w;
    vec2 dir = s1 - s0;
    float len = length(dir);
    if (len < 1e-5) return;
    dir /= len;
    vec2 offset = vec2(-dir.y, dir.x) * (u_thickness / u_viewport);
    g_color = v_color[0];
    gl_Position = vec4(p0.xy + offset * p0.w, p0.z, p0.w);
    EmitVertex();
    gl_Position = vec4(p0.xy - offset * p0.w, p0.z, p0.w);
    EmitVertex();
    g_color = v_color[1];
    gl_Position = vec4(p1.xy + offset * p1.w, p1.z, p1.w);
    EmitVertex();
    gl_Position = vec4(p1.xy - offset * p1.w, p1.z, p1.w);
    EmitVertex();
    EndPrimitive();
}
"""

LINE_FS = """
#version 330
in vec4 g_color;
out vec4 f_color;
void main() {
    f_color = g_color;
}
"""

TEXT_VS = """
#version 330
uniform mat4 u_mvp;
in vec3 in_pos;
in vec2 in_uv;
out vec2 v_uv;
void main() {
    v_uv = in_uv;
    gl_Position = u_mvp * vec4(in_pos, 1.0);
}
"""

TEXT_FS = """
#version 330
uniform sampler2D u_tex;
in vec2 v_uv;
out vec4 f_color;
void main() {
    f_color = texture(u_tex, v_uv);
}
"""

PANE_VS = """
#version 330
uniform mat4 u_mvp;
in vec3 in_pos;
void main() {
    gl_Position = u_mvp * vec4(in_pos, 1.0);
}
"""

PANE_FS = """
#version 330
uniform vec3 u_color;
out vec4 f_color;
void main() {
    f_color = vec4(u_color, 1.0);
}
"""


def _quad(center, right, up, width, height) -> np.ndarray:
    center = np.asarray(center, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)
    r = right * (width * 0.5)
    u = up * (height * 0.5)
    corners = (
        (center - r - u, (0.0, 0.0)),
        (center + r - u, (1.0, 0.0)),
        (center + r + u, (1.0, 1.0)),
        (center - r - u, (0.0, 0.0)),
        (center + r + u, (1.0, 1.0)),
        (center - r + u, (0.0, 1.0)),
    )
    data = []
    for pos, uv in corners:
        data.extend((pos[0], pos[1], pos[2], uv[0], uv[1]))
    return np.asarray(data, dtype=np.float32)


class App(mglw.WindowConfig):
    gl_version = (3, 3)
    title = "XBT Implied Volatility Surface"
    window_size = (1430, 1040)
    aspect_ratio = None
    resizable = True
    samples = 4
    clear_color = BG

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        self.elev = 22.0
        self.azim = -60.0
        self.view_height = 1.12
        self.eye_distance = 8.0
        self.target = np.zeros(3, dtype=np.float64)
        self._buttons: set[int] = set()

        scene = build_surface()
        self.xlim, self.ylim, self.zlim = scene["xlim"], scene["ylim"], scene["zlim"]
        self.iv_min, self.iv_max = scene["iv_min"], scene["iv_max"]

        self.surf_prog = self.ctx.program(vertex_shader=SURFACE_VS, fragment_shader=SURFACE_FS)
        self.pane_prog = self.ctx.program(vertex_shader=PANE_VS, fragment_shader=PANE_FS)
        self.line_prog = self.ctx.program(
            vertex_shader=LINE_VS, geometry_shader=LINE_GS, fragment_shader=LINE_FS
        )
        self.text_prog = self.ctx.program(vertex_shader=TEXT_VS, fragment_shader=TEXT_FS)
        self.text_prog["u_tex"].value = 0

        surf = np.concatenate([scene["position"], scene["normal"], scene["color"]], axis=1)
        vbo = self.ctx.buffer(surf.astype(np.float32).tobytes())
        ibo = self.ctx.buffer(scene["indices"].tobytes())
        self.surf_vao = self.ctx.vertex_array(
            self.surf_prog,
            [(vbo, "3f 3f 3f", "in_pos", "in_normal", "in_color")],
            ibo,
        )

        self._pane_sides = None
        self._pane_sides = self._far_sides(self._eye())
        fill, grids, borders = build_panes(self.xlim, self.ylim, self.zlim, *self._pane_sides)
        self.pane_vbo = self.ctx.buffer(np.ascontiguousarray(fill).tobytes(), dynamic=True)
        self.pane_vao = self.ctx.vertex_array(self.pane_prog, [(self.pane_vbo, "3f", "in_pos")])
        self.pane_vertices = fill.shape[0]

        self.wire_vao = self._line_vao(paint_lines(scene["wires"], WIRE_RGBA))
        self.grid_vao, self.grid_vbo = self._line_vao(paint_lines(grids, GRID_RGBA), dynamic=True)
        self.border_vao, self.border_vbo = self._line_vao(paint_lines(borders, BORDER_RGBA), dynamic=True)
        self.wire_count = scene["wires"].shape[0] * 2
        self.grid_count = grids.shape[0] * 2
        self.border_count = borders.shape[0] * 2

        self.labels = self._make_labels(scene)
        self._layout_labels(*self._pane_sides)
        self.hud = self._make_hud()
        self.text_vbo = self.ctx.buffer(reserve=65536, dynamic=True)
        self.text_vao = self.ctx.vertex_array(
            self.text_prog, [(self.text_vbo, "3f 2f", "in_pos", "in_uv")]
        )
        bar_x, bar_w = 0.80, 0.028
        bar_y0, bar_y1 = -0.50, 0.58
        x0, x1 = bar_x - bar_w * 0.5, bar_x + bar_w * 0.5
        border = []
        loop = [(x0, bar_y0), (x1, bar_y0), (x1, bar_y1), (x0, bar_y1), (x0, bar_y0)]
        for (ax, ay), (bx, by) in zip(loop[:-1], loop[1:]):
            border.extend((ax, ay, 0.0, *BORDER_RGBA, bx, by, 0.0, *BORDER_RGBA))
        self.cbar_border_vao = self._line_vao(np.asarray(border, dtype=np.float32).reshape(-1, 7))
        self.bar_x = bar_x
        self.bar_w = bar_w
        self.bar_y0 = bar_y0
        self.bar_y1 = bar_y1

    def _line_vao(self, packed: np.ndarray, dynamic: bool = False):
        buffer = self.ctx.buffer(np.ascontiguousarray(packed).tobytes(), dynamic=dynamic)
        vao = self.ctx.vertex_array(self.line_prog, [(buffer, "3f 4f", "in_pos", "in_color")])
        return (vao, buffer) if dynamic else vao

    def _far_sides(self, eye: np.ndarray):
        """Put each sheet on the box face farthest from the camera."""
        limits = (self.xlim, self.ylim, self.zlim)
        # Sticky across the edge-on view so the sheets don't flicker.
        sticky = self._pane_sides or (self.xlim[0], self.ylim[1], self.zlim[0])
        sides = []
        for axis, (lo, hi) in enumerate(limits):
            if eye[axis] > 0.55:
                sides.append(lo)
            elif eye[axis] < -0.55:
                sides.append(hi)
            else:
                sides.append(sticky[axis])
        return tuple(sides)

    def _sync_panes(self, eye: np.ndarray):
        sides = self._far_sides(eye)
        if sides == self._pane_sides:
            return
        self._pane_sides = sides
        fill, grids, borders = build_panes(self.xlim, self.ylim, self.zlim, *sides)
        self.pane_vbo.write(np.ascontiguousarray(fill).tobytes())
        self.grid_vbo.write(np.ascontiguousarray(paint_lines(grids, GRID_RGBA)).tobytes())
        self.border_vbo.write(np.ascontiguousarray(paint_lines(borders, BORDER_RGBA)).tobytes())
        self._layout_labels(*sides)

    def _texture(self, image: Image.Image):
        texture = self.ctx.texture(image.size, 4, image.tobytes())
        texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        return texture

    def _make_labels(self, scene):
        labels = []

        def add(kind, text, size=20, rotate=0, **extra):
            image = text_image(text, size, TICK_RGBA, bold=False, rotate=rotate)
            labels.append(
                {
                    "tex": self._texture(image),
                    "px": image.size,
                    "kind": kind,
                    "anchor": np.zeros(3, dtype=np.float32),
                    **extra,
                }
            )

        for value in (-1.0, -0.5, 0.0, 0.5, 1.0):
            add("x", fmt_num(value), value=value)
        add("x_title", "Delta (Put \u2190  \u2192 Call)", size=22, value=0.0)

        last_y = None
        for years, name in zip(scene["tenor_years"][::2], scene["tenor_labels"][::2]):
            y = float(map_axis(years, self.ylim[0], self.ylim[1], BOX[1]))
            if last_y is not None and abs(y - last_y) < 0.16:
                continue
            last_y = y
            add("y", name, value=float(years))
        add("y_title", "Tenor", size=22, value=(self.ylim[0] + self.ylim[1]) * 0.5)

        for value in nice_ticks(self.iv_min, self.iv_max, 5):
            add("z", fmt_num(value), value=value)
        add("z_title", "Implied Vol (%)", size=22, rotate=90, value=(self.iv_min + self.iv_max) * 0.5)
        return labels

    def _layout_labels(self, x_at, y_at, z_at):
        """Keep tick labels on the outer lip, opposite the sheet that would cover them."""
        xlim, ylim, zlim = self.xlim, self.ylim, self.zlim
        # Near edge is the face the camera is on; the sheet occupies the far face.
        y_near = ylim[0] if y_at == ylim[1] else ylim[1]
        x_near = xlim[0] if x_at == xlim[1] else xlim[1]
        z_out = -1.0 if z_at == zlim[0] else 1.0
        x_far_out = -1.0 if x_at == xlim[0] else 1.0
        y_near_out = -1.0 if y_near == ylim[0] else 1.0
        x_near_out = -1.0 if x_near == xlim[0] else 1.0

        for label in self.labels:
            kind = label["kind"]
            value = label["value"]
            if kind == "x":
                anchor = _point(value, y_near, z_at, xlim, ylim, zlim)
                anchor += np.array([0.0, y_near_out * 0.05, z_out * 0.07], dtype=np.float32)
            elif kind == "x_title":
                anchor = _point(value, y_near, z_at, xlim, ylim, zlim)
                anchor += np.array([0.0, y_near_out * 0.05, z_out * 0.18], dtype=np.float32)
            elif kind == "y":
                anchor = _point(x_near, value, z_at, xlim, ylim, zlim)
                anchor += np.array([x_near_out * 0.13, 0.0, z_out * 0.06], dtype=np.float32)
            elif kind == "y_title":
                anchor = _point(x_near, value, z_at, xlim, ylim, zlim)
                anchor += np.array([x_near_out * 0.24, 0.0, z_out * 0.06], dtype=np.float32)
            elif kind == "z":
                anchor = _point(x_at, y_near, value, xlim, ylim, zlim)
                anchor += np.array([x_far_out * 0.11, y_near_out * 0.05, 0.0], dtype=np.float32)
            else:
                anchor = _point(x_at, y_near, value, xlim, ylim, zlim)
                anchor += np.array([x_far_out * 0.24, y_near_out * 0.05, 0.0], dtype=np.float32)
            label["anchor"] = anchor

    def _make_hud(self):
        gradient = (colormap(np.linspace(0.0, 1.0, 256)) * 255).astype(np.uint8)
        pixels = np.concatenate([gradient, np.full((256, 1), 255, dtype=np.uint8)], axis=1)
        image = Image.frombytes("RGBA", (1, 256), pixels.tobytes())
        bar = self._texture(image)
        title = text_image("XBT Implied Volatility Surface", 36, TITLE_RGBA, bold=True)
        hint = text_image("drag to rotate    scroll to zoom", 18, HINT_RGBA)
        cbar_ticks = []
        for value in nice_ticks(self.iv_min, self.iv_max, 5):
            tick = text_image(fmt_num(value), 20, TICK_RGBA)
            cbar_ticks.append((value, self._texture(tick), tick.size))
        cbar_label = text_image("Implied Vol (%)", 20, TICK_RGBA, rotate=90)
        return {
            "bar": bar,
            "title": (self._texture(title), title.size),
            "hint": (self._texture(hint), hint.size),
            "ticks": cbar_ticks,
            "label": (self._texture(cbar_label), cbar_label.size),
        }

    def on_mouse_press_event(self, x, y, button):
        self._buttons.add(button)

    def on_mouse_release_event(self, x, y, button):
        self._buttons.discard(button)

    def on_mouse_drag_event(self, x, y, dx, dy):
        if 2 in self._buttons:
            scale = (2.0 * self.view_height) / max(self.wnd.buffer_size[1], 1)
            right, up = self._basis()
            self.target -= right * dx * scale + up * dy * scale
            return
        self.azim -= dx * 0.35
        self.elev = float(np.clip(self.elev - dy * 0.28, 8.0, 80.0))

    def on_mouse_scroll_event(self, x_offset, y_offset):
        if y_offset == 0:
            return
        factor = 0.9 if y_offset > 0 else 1.1
        self.view_height = float(np.clip(self.view_height * factor, 0.7, 4.2))

    def _eye(self) -> np.ndarray:
        elev = np.deg2rad(self.elev)
        azim = np.deg2rad(self.azim)
        direction = np.array(
            [
                np.cos(azim) * np.cos(elev),
                np.sin(azim) * np.cos(elev),
                np.sin(elev),
            ]
        )
        return self.target + direction * self.eye_distance

    def _basis(self):
        eye = self._eye()
        forward = self.target - eye
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        return right, up

    def _mvp(self, width, height):
        aspect = width / max(height, 1)
        half_h = self.view_height
        half_w = half_h * aspect
        # Leave the right side of the frame for the color bar.
        projection = ortho(-half_w * 0.92, half_w * 1.16, -half_h, half_h, 0.05, 80.0)
        view = look_at(self._eye(), self.target, np.array([0.0, 0.0, 1.0]))
        return projection @ view

    def _write_mvp(self, program, mvp):
        program["u_mvp"].write(np.ascontiguousarray(mvp.T, dtype=np.float32).tobytes())

    def _draw_quad(self, texture, quad, mvp):
        raw = np.ascontiguousarray(quad, dtype=np.float32).tobytes()
        self.text_vbo.write(raw)
        self._write_mvp(self.text_prog, mvp)
        texture.use(0)
        self.text_vao.render(moderngl.TRIANGLES, vertices=6)

    def _draw_lines(self, vao, count, mvp, thickness, bias):
        self._write_mvp(self.line_prog, mvp)
        self.line_prog["u_bias"].value = bias
        self.line_prog["u_thickness"].value = thickness
        vao.render(moderngl.LINES, vertices=count)

    def on_render(self, time, frame_time):
        width, height = self.wnd.buffer_size
        self.ctx.viewport = (0, 0, width, height)
        mvp = self._mvp(width, height)
        eye = self._eye()
        self._sync_panes(eye)
        light = eye - self.target
        light = light / np.linalg.norm(light) + np.array([0.15, -0.05, 0.9])

        self.ctx.depth_mask = True
        self._write_mvp(self.pane_prog, mvp)
        self.pane_prog["u_color"].write(PANE.astype(np.float32).tobytes())
        self.pane_vao.render(moderngl.TRIANGLES, vertices=self.pane_vertices)

        self._write_mvp(self.surf_prog, mvp)
        self.surf_prog["u_eye"].write(eye.astype(np.float32).tobytes())
        self.surf_prog["u_light"].write(light.astype(np.float32).tobytes())
        self.surf_vao.render(moderngl.TRIANGLES)

        self.line_prog["u_viewport"].value = (float(width), float(height))
        self.ctx.depth_mask = False
        self._draw_lines(self.grid_vao, self.grid_count, mvp, 1.15, 0.0015)
        self._draw_lines(self.border_vao, self.border_count, mvp, 2.05, 0.0022)
        self._draw_lines(self.wire_vao, self.wire_count, mvp, 0.85, 0.0028)
        self.ctx.depth_mask = True

        world_per_px = (2.0 * self.view_height) / height
        right, up = self._basis()
        for label in self.labels:
            px_w, px_h = label["px"]
            quad = _quad(label["anchor"], right, up, px_w * world_per_px, px_h * world_per_px)
            self._draw_quad(label["tex"], quad, mvp)

        self.ctx.disable(moderngl.DEPTH_TEST)
        identity = np.eye(4, dtype=np.float64)
        ndc_x = 2.0 / width
        ndc_y = 2.0 / height

        def hud(texture, px, cx, cy):
            self._draw_quad(texture, _quad((cx, cy, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), px[0] * ndc_x, px[1] * ndc_y), identity)

        hud(self.hud["title"][0], self.hud["title"][1], -0.06, 0.90)
        hud(self.hud["hint"][0], self.hud["hint"][1], -0.62, -0.93)

        bar_h = self.bar_y1 - self.bar_y0
        self._draw_quad(
            self.hud["bar"],
            _quad(
                (self.bar_x, (self.bar_y0 + self.bar_y1) * 0.5, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                self.bar_w,
                bar_h,
            ),
            identity,
        )
        self._draw_lines(self.cbar_border_vao, 8, identity, 1.6, 0.0)

        for value, texture, px in self.hud["ticks"]:
            t = (value - self.iv_min) / (self.iv_max - self.iv_min)
            hud(texture, px, self.bar_x + 0.055, self.bar_y0 + t * bar_h)
        hud(self.hud["label"][0], self.hud["label"][1], self.bar_x + 0.115, (self.bar_y0 + self.bar_y1) * 0.5)
        self.ctx.enable(moderngl.DEPTH_TEST)


if __name__ == "__main__":
    mglw.run_window_config(App)
