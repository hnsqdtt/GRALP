from __future__ import annotations

"""Narrow-channel ("pillar lattice") scenario for episodic goal navigation.

A square arena enclosed by walls, tiled with one-cell square pillars on a
regular lattice so the free space degenerates to a grid of narrow corridors:

    +-----------------------------+   <- wall
    | . . . . . . . . . . . . . . |
    | . ## . ## . ## . ## . ## . |   ## = pillar (pillar_w x pillar_w)
    | . . . . . . . . . . . . . . |   .  = channel (width channel_w)
    | . ## . ## . ## . ## . ## . |
    | . . . . . . . . . . . . . . |
    +-----------------------------+

Geometry knobs (all metres):
  - ``pillar_w``  : pillar side length (the "one-grid-sized obstacle").
  - ``channel_w`` : gap between adjacent pillars (the navigable corridor).
  - ``n_pillars`` : pillars per row/column (square lattice => n_pillars**2).
  - ``robot_radius`` : robot footprint half-extent; obstacles are Minkowski-
    inflated by this so a *point* planner that respects the inflated map keeps
    the real footprint clear. The free corridor width is ``channel_w - 2*r``,
    so the lattice is passable iff ``channel_w > 2*robot_radius``.

Robot model: a square footprint of half-extent ``robot_radius`` (Chebyshev /
box inflation). This makes the inflated free space exactly the corridors shrunk
by ``r`` on each side -- uniform corridor width ``channel_w - 2r`` with open
square crossroads. It is a hair more conservative at pillar corners than a disk
footprint; for a narrow-corridor benchmark that is the safe direction.

What this module owns (pure geometry; no torch policy/DWA logic):
  * the obstacle box list (pillars + 4 walls), for rendering;
  * an *inflated* boolean occupancy grid as a torch tensor on ``device`` -- the
    single source of truth the env raycasts and collision-checks against;
  * the corridor-crossroad node lattice (candidate start/goal points, each a
    "channel centre"), plus an all-pairs BFS geodesic over the node graph so
    the eval harness can report SPL;
  * ASCII + matplotlib visualisations.

World frame: arena centred on the origin; +x right, +y up; grid cell
``occ[iy, ix]`` covers the world square centred at
``(x_min + (ix+0.5)*res, y_min + (iy+0.5)*res)``.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import math

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class NarrowChannelConfig:
    n_pillars: int = 6          # pillars per side (square lattice)
    pillar_w: float = 1.0       # pillar side length (m)
    channel_w: float = 0.5      # corridor width between pillars (m); the "narrow" knob
    robot_radius: float = 0.18  # footprint half-extent (m); layer-1 inflation amount
    obstacle_inflation_extra: float = 0.02  # layer-2 EXTRA inflation, LOS carrot only
    wall_thickness: float = 0.2  # perimeter wall thickness (m)
    res: float = 0.02           # occupancy raster resolution (m / cell)

    def __post_init__(self) -> None:
        if self.n_pillars < 1:
            raise ValueError(f"n_pillars must be >= 1, got {self.n_pillars}")
        for name in ("pillar_w", "channel_w", "wall_thickness", "res"):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be > 0, got {getattr(self, name)}")
        if self.robot_radius < 0.0:
            raise ValueError(f"robot_radius must be >= 0, got {self.robot_radius}")
        if self.obstacle_inflation_extra < 0.0:
            raise ValueError(
                f"obstacle_inflation_extra must be >= 0, got {self.obstacle_inflation_extra}")

    @property
    def pitch(self) -> float:
        return float(self.pillar_w + self.channel_w)

    @property
    def interior(self) -> float:
        """Interior side length: n pillars + (n+1) channels (channel on each border)."""
        n = int(self.n_pillars)
        return n * float(self.pillar_w) + (n + 1) * float(self.channel_w)

    @property
    def free_corridor_w(self) -> float:
        """Navigable corridor width after layer-1 (robot_radius) inflation."""
        return float(self.channel_w) - 2.0 * float(self.robot_radius)

    @property
    def los_corridor_w(self) -> float:
        """Corridor width seen through the layer-2 (robot_radius + extra) inflation
        used for LOS local-target selection; <= 0 means the carrot inflation
        closes the passage (too thick for this channel)."""
        return float(self.channel_w) - 2.0 * (float(self.robot_radius)
                                              + float(self.obstacle_inflation_extra))


# A box is (cx, cy, half_x, half_y) in world metres.
Box = Tuple[float, float, float, float]


@dataclass
class NarrowChannelMap:
    cfg: NarrowChannelConfig
    device: torch.device

    # filled by __post_init__
    pillars: List[Box] = field(default_factory=list)
    walls: List[Box] = field(default_factory=list)
    occ: torch.Tensor = field(default=None)        # layer-1 inflation (robot_radius), bool [H, W]
    occ_los: torch.Tensor = field(default=None)    # layer-2 inflation (radius + extra), LOS only
    occ_true: torch.Tensor = field(default=None)   # un-inflated, for rendering
    nodes: torch.Tensor = field(default=None)      # [K, 2] world xy of free crossroad nodes
    node_ij: List[Tuple[int, int]] = field(default_factory=list)  # lattice (col,row) per node
    geo_nodes: np.ndarray = field(default=None)    # [K, K] geodesic in node-steps (inf if unreachable)

    def __post_init__(self) -> None:
        self.device = torch.device(self.device)
        self._build_boxes()
        self._build_occupancy()
        self._build_nodes()
        self._build_geodesic()

    # -- geometry -------------------------------------------------------

    @property
    def half_extent(self) -> float:
        """Half side of the *interior* (origin-centred), excluding walls."""
        return 0.5 * self.cfg.interior

    @property
    def bound(self) -> float:
        """Half side of the full grid (interior + walls)."""
        return self.half_extent + float(self.cfg.wall_thickness)

    def _pillar_centers_1d(self) -> np.ndarray:
        """Pillar centre coordinate along one axis (origin-centred)."""
        cfg = self.cfg
        n = int(cfg.n_pillars)
        # c_i = channel_w + pillar_w/2 + i*pitch, then shift to origin-centred.
        offs = cfg.channel_w + 0.5 * cfg.pillar_w + np.arange(n) * cfg.pitch
        return offs - self.half_extent

    def _channel_centers_1d(self) -> np.ndarray:
        """Channel (corridor) centreline coordinate along one axis (origin-centred).

        There are n_pillars+1 channels (one against each wall plus the gaps).
        """
        cfg = self.cfg
        n = int(cfg.n_pillars)
        offs = 0.5 * cfg.channel_w + np.arange(n + 1) * cfg.pitch
        return offs - self.half_extent

    def _build_boxes(self) -> None:
        cfg = self.cfg
        hp = 0.5 * cfg.pillar_w
        cs = self._pillar_centers_1d()
        self.pillars = [(float(cx), float(cy), hp, hp) for cy in cs for cx in cs]

        he = self.half_extent
        t = 0.5 * float(cfg.wall_thickness)
        outer = he + 2.0 * t  # span walls a touch past the corners so they meet
        self.walls = [
            (0.0, he + t, outer, t),    # top
            (0.0, -(he + t), outer, t),  # bottom
            (-(he + t), 0.0, t, he),     # left
            (he + t, 0.0, t, he),        # right
        ]

    def _build_occupancy(self) -> None:
        cfg = self.cfg
        res = float(cfg.res)
        b = self.bound
        self.x_min = -b
        self.y_min = -b
        n_cells = int(math.ceil((2.0 * b) / res))
        self.W = n_cells
        self.H = n_cells
        dev = self.device

        # Cell-centre world coordinates.
        xs = self.x_min + (torch.arange(self.W, device=dev, dtype=torch.float32) + 0.5) * res
        ys = self.y_min + (torch.arange(self.H, device=dev, dtype=torch.float32) + 0.5) * res
        X = xs.view(1, self.W)
        Y = ys.view(self.H, 1)

        r = float(cfg.robot_radius)             # layer-1 (rays + collision)
        r_los = r + float(cfg.obstacle_inflation_extra)  # layer-2 (LOS carrot)
        occ = torch.zeros((self.H, self.W), dtype=torch.bool, device=dev)
        occ_los = torch.zeros((self.H, self.W), dtype=torch.bool, device=dev)
        occ_true = torch.zeros((self.H, self.W), dtype=torch.bool, device=dev)
        for cx, cy, hx, hy in (self.pillars + self.walls):
            dxabs = (X - cx).abs()
            dyabs = (Y - cy).abs()
            occ_true |= (dxabs <= hx) & (dyabs <= hy)
            occ |= (dxabs <= (hx + r)) & (dyabs <= (hy + r))
            occ_los |= (dxabs <= (hx + r_los)) & (dyabs <= (hy + r_los))
        self.occ = occ
        self.occ_los = occ_los
        self.occ_true = occ_true

    # -- world <-> grid -------------------------------------------------

    def _lookup(self, grid: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
        res = float(self.cfg.res)
        ix = torch.floor((xy[..., 0] - self.x_min) / res).long()
        iy = torch.floor((xy[..., 1] - self.y_min) / res).long()
        oob = (ix < 0) | (ix >= self.W) | (iy < 0) | (iy >= self.H)
        ixc = ix.clamp(0, self.W - 1)
        iyc = iy.clamp(0, self.H - 1)
        return grid[iyc, ixc] | oob

    def occupied(self, xy: torch.Tensor) -> torch.Tensor:
        """Layer-1 (robot_radius) occupancy for world points ``xy`` [..., 2] -> bool [...].

        Drives the policy rays and collision detection. Points outside the grid
        count as occupied (you have left the arena).
        """
        return self._lookup(self.occ, xy)

    def occupied_los(self, xy: torch.Tensor) -> torch.Tensor:
        """Layer-2 (robot_radius + extra) occupancy -> bool [...].

        Drives ONLY the LOS local-target (carrot) selection, mirroring the
        deployment sim's ``los_dilated`` so the carrot never sits on the
        layer-1 inflated boundary.
        """
        return self._lookup(self.occ_los, xy)

    # -- node lattice + geodesic ---------------------------------------

    def _build_nodes(self) -> None:
        """Crossroad node lattice = free corridor centreline intersections.

        Candidate start/goal points ("channel centres"). A node is kept only if
        its cell is free in the inflated map (so a robot can sit there).
        """
        cc = self._channel_centers_1d()  # length n+1
        m = len(cc)
        coords: List[Tuple[float, float]] = []
        ij: List[Tuple[int, int]] = []
        # node index grid -> position in coords (or -1 if dropped)
        grid_to_idx = -np.ones((m, m), dtype=np.int64)
        kept = 0
        # Evaluate freeness in one batched lookup.
        pts = torch.tensor(
            [[float(cc[i]), float(cc[j])] for j in range(m) for i in range(m)],
            device=self.device, dtype=torch.float32,
        )
        free = (~self.occupied(pts)).view(m, m).cpu().numpy()  # free[j, i]
        for j in range(m):
            for i in range(m):
                if free[j, i]:
                    grid_to_idx[j, i] = kept
                    coords.append((float(cc[i]), float(cc[j])))
                    ij.append((i, j))
                    kept += 1
        self._grid_to_idx = grid_to_idx
        self._node_grid_m = m
        self.nodes = torch.tensor(coords, device=self.device, dtype=torch.float32) \
            if coords else torch.zeros((0, 2), device=self.device, dtype=torch.float32)
        self.node_ij = ij

    def _edge_free(self, p0: Tuple[float, float], p1: Tuple[float, float]) -> bool:
        """True if the straight corridor segment p0->p1 is clear in the inflated map."""
        res = float(self.cfg.res)
        d = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        n = max(2, int(math.ceil(d / res)) + 1)
        ts = torch.linspace(0.0, 1.0, n, device=self.device)
        xs = p0[0] + ts * (p1[0] - p0[0])
        ys = p0[1] + ts * (p1[1] - p0[1])
        pts = torch.stack([xs, ys], dim=-1)
        return not bool(self.occupied(pts).any().item())

    def _build_geodesic(self) -> None:
        """All-pairs BFS over the 4-connected node graph; weights in node-steps."""
        K = self.nodes.shape[0]
        geo = np.full((K, K), np.inf, dtype=np.float64)
        if K == 0:
            self.geo_nodes = geo
            self._adj = []
            return
        coords = self.nodes.cpu().numpy()
        m = self._node_grid_m
        g2i = self._grid_to_idx
        adj: List[List[int]] = [[] for _ in range(K)]
        for k, (i, j) in enumerate(self.node_ij):
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ni, nj = i + di, j + dj
                if 0 <= ni < m and 0 <= nj < m and g2i[nj, ni] >= 0:
                    nk = int(g2i[nj, ni])
                    if self._edge_free(tuple(coords[k]), tuple(coords[nk])):
                        adj[k].append(nk)
        self._adj = adj
        for s in range(K):
            dist = geo[s]
            dist[s] = 0.0
            q = deque([s])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == np.inf:
                        dist[v] = dist[u] + 1.0
                        q.append(v)
        self.geo_nodes = geo

    def geodesic_m(self, start_idx: Sequence[int], goal_idx: Sequence[int]) -> np.ndarray:
        """Geodesic distance in METRES for arrays of node indices (inf if unreachable)."""
        s = np.asarray(start_idx, dtype=np.int64)
        g = np.asarray(goal_idx, dtype=np.int64)
        return self.geo_nodes[s, g] * float(self.cfg.pitch)

    @property
    def passable(self) -> bool:
        """Whether at least one pair of distinct nodes is mutually reachable."""
        K = self.nodes.shape[0]
        if K < 2:
            return False
        finite = np.isfinite(self.geo_nodes)
        np.fill_diagonal(finite, False)
        return bool(finite.any())

    # -- introspection / viz -------------------------------------------

    def summary(self) -> str:
        cfg = self.cfg
        lines = [
            f"NarrowChannelMap: {cfg.n_pillars}x{cfg.n_pillars} pillars "
            f"({len(self.pillars)} total), interior {cfg.interior:.2f}m, "
            f"full {2*self.bound:.2f}m",
            f"  pillar_w={cfg.pillar_w:.3f}m  channel_w={cfg.channel_w:.3f}m  "
            f"pitch={cfg.pitch:.3f}m  robot_radius={cfg.robot_radius:.3f}m  "
            f"infl_extra={cfg.obstacle_inflation_extra:.3f}m",
            f"  layer-1 free corridor (rays/collision) = channel_w - 2*r = "
            f"{cfg.free_corridor_w:.3f}m  ({'PASSABLE' if cfg.free_corridor_w > 0 else 'BLOCKED'})",
            f"  layer-2 LOS corridor (carrot) = channel_w - 2*(r+extra) = "
            f"{cfg.los_corridor_w:.3f}m",
            f"  occupancy grid {self.H}x{self.W} @ {cfg.res:.3f}m  "
            f"({100.0*float(self.occ.float().mean()):.1f}% inflated-occupied)",
            f"  free crossroad nodes: {self.nodes.shape[0]}  "
            f"(graph connected: {self.passable})",
        ]
        fcw = cfg.free_corridor_w
        if fcw > 0.0 and self.nodes.shape[0] == 0:
            lines.append(
                f"  WARNING: free corridor {fcw:.3f}m is thinner than ~2*res "
                f"({2*cfg.res:.3f}m); rasterisation closed it. Lower --res below "
                f"{0.5*fcw:.3f}m to recover the passage."
            )
        elif 0.0 < fcw < 2.0 * cfg.res:
            lines.append(
                f"  WARNING: free corridor {fcw:.3f}m spans < ~2 cells at res "
                f"{cfg.res:.3f}m; consider --res <= {0.5*fcw:.3f}m for a clean raster."
            )
        if cfg.free_corridor_w > 0.0 and cfg.los_corridor_w <= cfg.res:
            lines.append(
                f"  WARNING: layer-2 LOS corridor {cfg.los_corridor_w:.3f}m <= res; the "
                f"carrot inflation nearly closes the passage. Lower "
                f"--obstacle-inflation-extra (< {0.5*fcw:.3f}m) for usable LOS guidance."
            )
        return "\n".join(lines)

    def ascii_art(self, max_cols: int = 80) -> str:
        """Coarse ASCII view of the *true* (un-inflated) map for a stdout sanity check."""
        H, W = self.H, self.W
        stride = max(1, int(math.ceil(W / max_cols)))
        occ = self.occ_true.cpu().numpy()
        rows = []
        for iy in range(H - 1, -1, -stride):  # +y up => print top row first
            cells = []
            for ix in range(0, W, stride):
                block = occ[max(0, iy - stride + 1):iy + 1, ix:ix + stride]
                cells.append("#" if block.any() else " ")
            rows.append("".join(cells))
        return "\n".join(rows)

    def render(self, path: str, *,
               trajectories: Optional[Sequence[Tuple[str, np.ndarray, Tuple[float, float], Tuple[float, float]]]] = None,
               title: Optional[str] = None,
               show_nodes: bool = False) -> Optional[str]:
        """Save a matplotlib PNG of the map (+ optional trajectories).

        ``trajectories``: list of (label, path_xy[N,2], start_xy, goal_xy).
        Returns the written path, or None if matplotlib is unavailable.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.patches import Rectangle
        except Exception as exc:  # pragma: no cover
            print(f"[scene] matplotlib unavailable, skipping render: {exc}")
            return None

        fig, ax = plt.subplots(figsize=(7, 7))
        for cx, cy, hx, hy in self.pillars:
            ax.add_patch(Rectangle((cx - hx, cy - hy), 2 * hx, 2 * hy,
                                   facecolor="0.45", edgecolor="0.25", linewidth=0.5))
        for cx, cy, hx, hy in self.walls:
            ax.add_patch(Rectangle((cx - hx, cy - hy), 2 * hx, 2 * hy,
                                   facecolor="0.2", edgecolor="none"))
        if show_nodes and self.nodes.shape[0] > 0:
            nd = self.nodes.cpu().numpy()
            ax.scatter(nd[:, 0], nd[:, 1], s=6, c="0.7", marker=".", zorder=2)

        colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
        if trajectories:
            for k, (label, xy, start, goal) in enumerate(trajectories):
                c = colors[k % len(colors)]
                xy = np.asarray(xy, dtype=np.float32)
                if xy.size:
                    ax.plot(xy[:, 0], xy[:, 1], "-", color=c, linewidth=1.5,
                            label=label, zorder=4)
                ax.plot([start[0]], [start[1]], "o", color=c, markersize=8,
                        markeredgecolor="k", zorder=5)
                ax.plot([goal[0]], [goal[1]], "*", color=c, markersize=14,
                        markeredgecolor="k", zorder=5)
            ax.legend(loc="upper right", fontsize=8)

        b = self.bound
        ax.set_xlim(-b, b)
        ax.set_ylim(-b, b)
        ax.set_aspect("equal")
        ax.set_title(title or self.cfg.__class__.__name__)
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)
        return path


def build_narrow_channel_map(cfg: NarrowChannelConfig,
                             device: torch.device | str = "cpu") -> NarrowChannelMap:
    return NarrowChannelMap(cfg=cfg, device=torch.device(device))
