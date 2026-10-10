import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ---------------------------------------------------------------
# 1) Synthetic implied-volatility surface
#    x: delta (-1 deep put .. +1 deep call), y: tenor, z: IV %
# ---------------------------------------------------------------
delta = np.linspace(-1, 1, 41)
tenor_labels = ["1W", "2W", "1M", "2M", "3M", "6M", "9M", "1Y", "18M", "2Y", "3Y", "5Y", "10Y"]
tenor_years = np.array([7, 14, 30, 60, 91, 182, 273, 365, 547, 730, 1095, 1825, 3650]) / 365.0

D, T = np.meshgrid(delta, tenor_years)
smile = 8.0 * D**2 - 3.0 * D                      # curvature + skew
base_term = 42 + 10 * np.exp(-T * 6) + 4 * np.log1p(T)
flatten = np.exp(-T * 0.8)                          # skew flattens with tenor
IV = base_term + smile * (0.35 + 0.65 * flatten)
IV += np.random.default_rng(7).normal(0, 0.15, IV.shape)

# ---------------------------------------------------------------
# 2) Plot, terminal style
# ---------------------------------------------------------------
plt.rcParams["figure.facecolor"] = "#0d1117"
fig = plt.figure(figsize=(11, 8), facecolor="#0d1117")
ax = fig.add_subplot(111, projection="3d")
ax.set_facecolor("#0d1117")

cmap = LinearSegmentedColormap.from_list(
    "term_vol", ["#2fae4a", "#a8e63a", "#f7e017", "#f28c1c", "#e6221f"]
)
surf = ax.plot_surface(D, T, IV, cmap=cmap, rstride=1, cstride=1,
                       linewidth=0.25, edgecolor="#1a1a1a",
                       antialiased=True, shade=True)

# Pad limits so the surface floats away from the background sheets
x_pad = (delta.max() - delta.min()) * 0.35
y_pad = (tenor_years.max() - tenor_years.min()) * 0.35
z_pad_low = (IV.max() - IV.min()) * 0.45
z_pad_high = (IV.max() - IV.min()) * 0.20
ax.set_xlim(delta.min() - x_pad, delta.max() + x_pad)
ax.set_ylim(tenor_years.min() - y_pad, tenor_years.max() + y_pad)
ax.set_zlim(IV.min() - z_pad_low, IV.max() + z_pad_high)

# Kill matplotlib's connected pane box -> draw 3 independent floating sheets
ax.xaxis.pane.set_visible(False)
ax.yaxis.pane.set_visible(False)
ax.zaxis.pane.set_visible(False)
ax.grid(False)
for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
    axis.line.set_color((1, 1, 1, 0))  # hide box edges, keep ticks

xlim, ylim, zlim = ax.get_xlim(), ax.get_ylim(), ax.get_zlim()
pane_color = "#0a0f16"
grid_color = "#ffffff"
grid_alpha = 0.45
border_color = "#ffffff"
n_grid = 7

# independent inset margins -> gaps at every seam, sheets never touch
mx = (xlim[1] - xlim[0]) * 0.12
my = (ylim[1] - ylim[0]) * 0.12
mz = (zlim[1] - zlim[0]) * 0.12


def draw_pane(const_val, u_lim, v_lim, u_margin, v_margin, order):
    """One inset, gridded, white-bordered sheet on a single 3D wall."""
    us = np.linspace(u_lim[0] + u_margin, u_lim[1] - u_margin, n_grid)
    vs = np.linspace(v_lim[0] + v_margin, v_lim[1] - v_margin, n_grid)
    U, V = np.meshgrid(us, vs)
    C = np.full_like(U, const_val)
    c = {order[0]: U, order[1]: V, order[2]: C}
    ax.plot_surface(c["x"], c["y"], c["z"], color=pane_color, alpha=1.0,
                    linewidth=0, shade=False, zorder=-10)
    for u in us:
        l = {order[0]: [u, u], order[1]: [vs[0], vs[-1]], order[2]: [const_val] * 2}
        ax.plot(l["x"], l["y"], l["z"], color=grid_color, alpha=grid_alpha, lw=0.5, zorder=-9)
    for v in vs:
        l = {order[0]: [us[0], us[-1]], order[1]: [v, v], order[2]: [const_val] * 2}
        ax.plot(l["x"], l["y"], l["z"], color=grid_color, alpha=grid_alpha, lw=0.5, zorder=-9)
    # white perimeter border
    b = {order[0]: [us[0], us[-1], us[-1], us[0], us[0]],
         order[1]: [vs[0], vs[0], vs[-1], vs[-1], vs[0]],
         order[2]: [const_val] * 5}
    ax.plot(b["x"], b["y"], b["z"], color=border_color, lw=1.3, zorder=-8)


draw_pane(ylim[1], xlim, zlim, mx, mz, order=("x", "z", "y"))  # back wall
draw_pane(xlim[0], ylim, zlim, my, mz, order=("y", "z", "x"))  # side wall
draw_pane(zlim[0], xlim, ylim, mx, my, order=("x", "y", "z"))  # floor

ax.set_box_aspect((1.3, 1.6, 0.8), zoom=0.92)

tick_color = "#c9d1d9"
for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
    axis.label.set_color(tick_color)
ax.tick_params(colors=tick_color)

ax.set_xlabel("Delta (Put \u2190  \u2192 Call)", labelpad=14, fontsize=10)
ax.set_yticks(tenor_years[::2])
ax.set_yticklabels(tenor_labels[::2])
ax.set_ylabel("Tenor", labelpad=18, fontsize=10)
ax.set_zlabel("Implied Vol (%)", labelpad=8, fontsize=10)
ax.set_title("XBT Implied Volatility Surface", color="white", fontsize=14, pad=0, y=0.97)
ax.view_init(elev=22, azim=-60)

cbar = fig.colorbar(surf, ax=ax, shrink=0.55, pad=0.02, aspect=18)
cbar.set_label("Implied Vol (%)", color=tick_color)
cbar.ax.yaxis.set_tick_params(color=tick_color)
plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color=tick_color)

plt.tight_layout()
plt.savefig("vol_surface.png", dpi=200, facecolor=fig.get_facecolor())
plt.show()  # run locally -> drag to rotate
