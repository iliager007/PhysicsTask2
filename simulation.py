import numpy as np
from js import document, window
from pyodide.ffi import create_proxy

W, H = 800, 600
FLOOR_Y = 560.0
GRAVITY = np.array([0.0, 980.0])
TAU = 2.0 * np.pi
DT = 1.0 / 60.0

SCENE_NAMES = [
    "Single Point",
    "Two Corners",
    "Three Points",
    "Full Top Edge",
    "Free Fall to Floor",
    "Drop onto Ball",
    "Dense Crumple",
    "Two Distant Corners",
    "Four Corners Spread",
    "Compressed Edges",
]


class Simulation:
    def __init__(self):
        self.n = 0
        self.pos = np.zeros((0, 2))
        self.vel = np.zeros((0, 2))
        self.predicted = np.zeros((0, 2))
        self.inv_mass = np.zeros(0)
        self.pin_mask = np.zeros(0, dtype=bool)
        self.pinned_pos = np.zeros((0, 2))

        self.c_i = np.zeros(0, dtype=np.int32)
        self.c_j = np.zeros(0, dtype=np.int32)
        self.c_rest = np.zeros(0)
        self.c_wi = np.zeros(0)
        self.c_wj = np.zeros(0)
        self.c_wsum = np.zeros(0)
        self.nc = 0

        self.struct_pairs = []
        self.shear_pairs = []
        self.rows = self.cols = 0

        self.iterations = 15
        self.stiffness = 0.80
        self.damping = 0.998
        self.restitution = 0.30
        self.floor_friction = 0.25
        self.solver_type = 0
        self.self_collision = False

        self.ball_pos = np.array([400.0, 350.0])
        self.ball_radius = 60.0
        self.ball_active = True

        self.grabbed = -1
        self.mouse = np.array([0.0, 0.0])
        self.particle_r = 2.5
        self.scene_id = 0

        self.canvas = document.getElementById("c")
        self.ctx = self.canvas.getContext("2d")
        self.setup_scene(0)

    # Cloth builder
    def _build_cloth(self, rows, cols, sp, ox, oy, pin_mode):
        self.rows, self.cols = rows, cols
        self.n = rows * cols
        n = self.n

        self.pos = np.zeros((n, 2))
        for r in range(rows):
            for c in range(cols):
                self.pos[r * cols + c] = [ox + c * sp, oy + r * sp]

        self.vel = np.zeros((n, 2))
        self.predicted = self.pos.copy()
        self.inv_mass = np.ones(n)
        self.pin_mask = np.zeros(n, dtype=bool)

        if pin_mode == "top":
            for c in range(cols):
                self.pin_mask[c] = True
        elif pin_mode == "corners":
            self.pin_mask[0] = True
            self.pin_mask[cols - 1] = True
        elif pin_mode == "single_center":
            self.pin_mask[cols // 2] = True
        elif pin_mode == "three_top":
            self.pin_mask[0] = True
            self.pin_mask[cols // 2] = True
            self.pin_mask[cols - 1] = True
        elif pin_mode == "four_corners":
            self.pin_mask[0] = True
            self.pin_mask[cols - 1] = True
            self.pin_mask[(rows - 1) * cols] = True
            self.pin_mask[(rows - 1) * cols + cols - 1] = True
        elif pin_mode == "sides":
            for r in range(rows):
                self.pin_mask[r * cols] = True
                self.pin_mask[r * cols + cols - 1] = True

        self.inv_mass[self.pin_mask] = 0.0
        self.pinned_pos = self.pos.copy()

        ci, cj, cr = [], [], []
        self.struct_pairs = []
        self.shear_pairs = []
        idx = 0

        for r in range(rows):
            for c in range(cols):
                p = r * cols + c
                if c < cols - 1:
                    ci.append(p); cj.append(p + 1); cr.append(sp)
                    self.struct_pairs.append(idx); idx += 1
                if r < rows - 1:
                    ci.append(p); cj.append(p + cols); cr.append(sp)
                    self.struct_pairs.append(idx); idx += 1
                if r < rows - 1 and c < cols - 1:
                    ci.append(p); cj.append(p + cols + 1); cr.append(sp * 1.41421356)
                    self.shear_pairs.append(idx); idx += 1
                if r < rows - 1 and c > 0:
                    ci.append(p); cj.append(p + cols - 1); cr.append(sp * 1.41421356)
                    self.shear_pairs.append(idx); idx += 1
                if c < cols - 2:
                    ci.append(p); cj.append(p + 2); cr.append(sp * 2)
                    idx += 1
                if r < rows - 2:
                    ci.append(p); cj.append(p + 2 * cols); cr.append(sp * 2)
                    idx += 1

        self.c_i = np.array(ci, dtype=np.int32)
        self.c_j = np.array(cj, dtype=np.int32)
        self.c_rest = np.array(cr)
        self.nc = len(ci)
        self.c_wi = self.inv_mass[self.c_i]
        self.c_wj = self.inv_mass[self.c_j]
        self.c_wsum = self.c_wi + self.c_wj

    def _snap_pins(self):
        self.pos[self.pin_mask] = self.pinned_pos[self.pin_mask]
        self.predicted = self.pos.copy()
        self.vel[self.pin_mask] = 0.0

    # Scenes
    def setup_scene(self, sid):
        self.scene_id = sid
        self.grabbed = -1
        sc = document.getElementById("selfcol")

        # --- Category 1: Hanging from fixed points ---
        if sid == 0:  # single center point
            self._build_cloth(14, 14, 22, 250, 30, "single_center")
            self.ball_active = True
            self.ball_pos = np.array([400.0, 350.0])
            self.ball_radius = 55.0
            sc.checked = False

        elif sid == 1:  # two top corners
            self._build_cloth(14, 20, 22, 170, 30, "corners")
            self.ball_active = True
            self.ball_pos = np.array([390.0, 310.0])
            self.ball_radius = 55.0
            sc.checked = False

        elif sid == 2:  # three spread points
            self._build_cloth(12, 22, 20, 160, 30, "three_top")
            self.ball_active = True
            self.ball_pos = np.array([380.0, 270.0])
            self.ball_radius = 50.0
            sc.checked = False

        elif sid == 3:  # full top edge
            self._build_cloth(15, 20, 22, 170, 30, "top")
            self.ball_active = True
            self.ball_pos = np.array([400.0, 310.0])
            self.ball_radius = 60.0
            sc.checked = False

        # --- Category 2: Floor & ball collisions (bounce + friction) ---
        elif sid == 4:  # free fall to floor
            self._build_cloth(10, 18, 22, 180, 60, "none")
            self.ball_active = False
            sc.checked = True

        elif sid == 5:  # drop onto ball then floor
            self._build_cloth(10, 14, 22, 230, 40, "none")
            self.ball_active = True
            self.ball_pos = np.array([380.0, 380.0])
            self.ball_radius = 65.0
            sc.checked = True

        elif sid == 6:  # dense crumple — small cloth, self-collision prominent
            self._build_cloth(10, 10, 18, 310, 40, "none")
            self.ball_active = True
            self.ball_pos = np.array([390.0, 460.0])
            self.ball_radius = 45.0
            sc.checked = True

        # --- Category 3: Overconstrained / incompatible constraints ---
        elif sid == 7:  # two distant corners
            self._build_cloth(8, 20, 22, 160, 80, "corners")
            nat_w = 19 * 22
            target_w = nat_w * 1.8
            cx = 400.0
            self.pinned_pos[0] = [cx - target_w / 2, 80]
            self.pinned_pos[self.cols - 1] = [cx + target_w / 2, 80]
            self._snap_pins()
            self.ball_active = False
            sc.checked = False

        elif sid == 8:  # four corners spread wide
            sp = 22
            self._build_cloth(12, 12, sp, 250, 100, "four_corners")
            nw = 11 * sp
            nh = 11 * sp
            spread = 1.7
            cx, cy = 400.0, 310.0
            hw, hh = nw * spread / 2, nh * spread / 2
            c = self.cols
            self.pinned_pos[0]                          = [cx - hw, cy - hh]
            self.pinned_pos[c - 1]                      = [cx + hw, cy - hh]
            self.pinned_pos[(self.rows - 1) * c]        = [cx - hw, cy + hh]
            self.pinned_pos[(self.rows - 1) * c + c - 1]= [cx + hw, cy + hh]
            self._snap_pins()
            self.ball_active = False
            sc.checked = False

        elif sid == 9:  # compressed edges (buckling)
            sp = 22
            self._build_cloth(8, 18, sp, 100, 220, "sides")
            nat_w = 17 * sp
            target_w = nat_w * 0.35
            cx = 400.0
            for r in range(self.rows):
                self.pinned_pos[r * self.cols] = [cx - target_w / 2, 220 + r * sp]
                self.pinned_pos[r * self.cols + self.cols - 1] = [cx + target_w / 2, 220 + r * sp]
            self._snap_pins()
            self.ball_active = False
            sc.checked = True

    # UI sync
    def _read_ui(self):
        self.iterations = int(document.getElementById("iter").value)
        self.stiffness = float(document.getElementById("stiff").value)
        self.damping = float(document.getElementById("damp").value)
        self.restitution = float(document.getElementById("rest").value)
        self.floor_friction = float(document.getElementById("ffric").value)
        self.solver_type = int(document.getElementById("solver").value)
        self.self_collision = bool(document.getElementById("selfcol").checked)

    # Main step
    def step(self):
        self._read_ui()
        if self.solver_type == 1:
            self._render_todo("Projective Dynamics")
            return
        if self.solver_type == 2:
            self._render_todo("VBD (Vertex Block Descent)")
            return
        self._xpbd_step()
        self._render()
    
    # XPBD solver  (Hitman-style Position Based Dynamics)-
    def _xpbd_step(self):
        dt = DT
        mask = self.inv_mass > 0

        pre_vel_y = self.vel[:, 1].copy()
        was_above = self.pos[:, 1] < FLOOR_Y - 1.0

        self.vel[mask] += GRAVITY * dt
        self.vel *= self.damping
        self.predicted = self.pos + self.vel * dt

        if self.grabbed >= 0:
            self.predicted[self.grabbed] = self.mouse

        alpha = max(1e-8, (1.0 - self.stiffness) * 0.1) / (dt * dt)

        for _ in range(self.iterations):
            p1 = self.predicted[self.c_i]
            p2 = self.predicted[self.c_j]
            diff = p2 - p1
            dist = np.sqrt(diff[:, 0] ** 2 + diff[:, 1] ** 2)
            dist_s = np.maximum(dist, 1e-7)

            C = dist - self.c_rest
            denom = self.c_wsum + alpha
            cmag = np.where(denom > 0, C / denom, 0.0)
            nx = diff[:, 0] / dist_s
            ny = diff[:, 1] / dist_s
            cx = cmag * nx
            cy = cmag * ny

            dp = np.zeros((self.n, 2))
            np.add.at(dp[:, 0], self.c_i,  self.c_wi * cx)
            np.add.at(dp[:, 1], self.c_i,  self.c_wi * cy)
            np.add.at(dp[:, 0], self.c_j, -self.c_wj * cx)
            np.add.at(dp[:, 1], self.c_j, -self.c_wj * cy)
            self.predicted += dp

            for i in np.where(self.pin_mask)[0]:
                if i != self.grabbed:
                    self.predicted[i] = self.pinned_pos[i]

            # Floor: clamp + position-based friction
            below = self.predicted[:, 1] > FLOOR_Y
            if np.any(below):
                self.predicted[below, 1] = FLOOR_Y
                dx = self.predicted[below, 0] - self.pos[below, 0]
                self.predicted[below, 0] = (
                    self.pos[below, 0] + dx * (1.0 - self.floor_friction * 0.5)
                )

            # Ball collision
            if self.ball_active:
                d = self.predicted - self.ball_pos
                dn = np.sqrt(d[:, 0] ** 2 + d[:, 1] ** 2)
                mindist = self.ball_radius + self.particle_r
                hit = dn < mindist
                if np.any(hit):
                    dns = np.maximum(dn[hit], 1e-7)
                    norm = d[hit] / dns[:, np.newaxis]
                    self.predicted[hit] = self.ball_pos + norm * mindist

            if self.self_collision:
                self._solve_self_collision()

            if self.grabbed >= 0:
                self.predicted[self.grabbed] = self.mouse

        # Velocity update
        self.vel[mask] = (self.predicted[mask] - self.pos[mask]) / dt
        self.pos[mask] = self.predicted[mask]

        # Elastic bounce: particles that just landed on the floor
        now_on_floor = self.pos[:, 1] >= FLOOR_Y - 0.5
        just_landed = now_on_floor & was_above
        if np.any(just_landed):
            impact = np.abs(pre_vel_y[just_landed])
            self.vel[just_landed, 1] = -impact * self.restitution

        # Continuous floor friction on velocity
        if np.any(now_on_floor):
            self.vel[now_on_floor, 0] *= 1.0 - self.floor_friction * 0.4

    # Self-collision via dynamic spheres  (spatial hash)
    def _solve_self_collision(self):
        min_d = self.particle_r * 5
        cell = min_d * 2
        grid = {}
        for i in range(self.n):
            kx = int(self.predicted[i, 0] / cell)
            ky = int(self.predicted[i, 1] / cell)
            key = kx * 10007 + ky
            if key not in grid:
                grid[key] = []
            grid[key].append(i)

        cols = self.cols
        for key, pts in grid.items():
            cx0 = pts[0]
            cxi = int(self.predicted[cx0, 0] / cell)
            cyi = int(self.predicted[cx0, 1] / cell)
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    nk = (cxi + dx) * 10007 + (cyi + dy)
                    npts = grid.get(nk)
                    if npts is None:
                        continue
                    for i in pts:
                        for j in npts:
                            if j <= i:
                                continue
                            ri, ci_c = divmod(i, cols)
                            rj, cj_c = divmod(j, cols)
                            if abs(ri - rj) <= 1 and abs(ci_c - cj_c) <= 1:
                                continue
                            diff = self.predicted[j] - self.predicted[i]
                            dist = (diff[0] ** 2 + diff[1] ** 2) ** 0.5
                            if dist < min_d and dist > 1e-7:
                                normal = diff / dist
                                corr = (min_d - dist) * 0.5
                                wi = self.inv_mass[i]
                                wj = self.inv_mass[j]
                                ws = wi + wj
                                if ws > 0:
                                    self.predicted[i] -= normal * corr * (wi / ws)
                                    self.predicted[j] += normal * corr * (wj / ws)

    # Stub renderer for unimplemented solvers
    def _render_todo(self, name):
        ctx = self.ctx
        ctx.fillStyle = "#1a1a2e"
        ctx.fillRect(0, 0, W, H)
        ctx.fillStyle = "#ffaa00"
        ctx.font = "bold 26px monospace"
        ctx.textAlign = "center"
        ctx.fillText(f"TODO: Implement {name}", W / 2, H / 2 - 40)
        ctx.fillStyle = "#887766"
        ctx.font = "16px monospace"
        ctx.fillText("Solver stub ready for future implementation.", W / 2, H / 2)
        ctx.fillText("Select PBD / XPBD to run the active simulation.", W / 2, H / 2 + 30)
        ctx.textAlign = "start"
    
    # Canvas rendering
    def _render(self):
        ctx = self.ctx
        pos = self.pos.tolist()

        ctx.fillStyle = "#1a1a2e"
        ctx.fillRect(0, 0, W, H)

        ctx.strokeStyle = "#1f1f35"
        ctx.lineWidth = 0.5
        x = 0
        while x <= W:
            ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke()
            x += 50
        y = 0
        while y <= H:
            ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke()
            y += 50

        ctx.fillStyle = "#1e1e38"
        ctx.fillRect(0, FLOOR_Y, W, H - FLOOR_Y)
        ctx.strokeStyle = "#4a4a7a"
        ctx.lineWidth = 2
        ctx.beginPath(); ctx.moveTo(0, FLOOR_Y); ctx.lineTo(W, FLOOR_Y); ctx.stroke()

        if self.ball_active:
            bx = float(self.ball_pos[0])
            by = float(self.ball_pos[1])
            br = self.ball_radius
            g = ctx.createRadialGradient(
                bx - br * 0.25, by - br * 0.25, br * 0.1, bx, by, br
            )
            g.addColorStop(0, "#555580")
            g.addColorStop(1, "#2a2a50")
            ctx.fillStyle = g
            ctx.beginPath(); ctx.arc(bx, by, br, 0, TAU); ctx.fill()
            ctx.strokeStyle = "#7777aa"
            ctx.lineWidth = 1.5
            ctx.stroke()

        ctx.strokeStyle = "rgba(0,190,255,0.55)"
        ctx.lineWidth = 1.2
        ctx.beginPath()
        for k in self.struct_pairs:
            i = int(self.c_i[k]); j = int(self.c_j[k])
            ctx.moveTo(pos[i][0], pos[i][1])
            ctx.lineTo(pos[j][0], pos[j][1])
        ctx.stroke()

        ctx.strokeStyle = "rgba(0,190,255,0.15)"
        ctx.lineWidth = 0.5
        ctx.beginPath()
        for k in self.shear_pairs:
            i = int(self.c_i[k]); j = int(self.c_j[k])
            ctx.moveTo(pos[i][0], pos[i][1])
            ctx.lineTo(pos[j][0], pos[j][1])
        ctx.stroke()

        ctx.fillStyle = "#44ddff"
        ctx.beginPath()
        for i in range(self.n):
            if not self.pin_mask[i] and i != self.grabbed:
                ctx.moveTo(pos[i][0] + 2.5, pos[i][1])
                ctx.arc(pos[i][0], pos[i][1], 2.5, 0, TAU)
        ctx.fill()

        ctx.fillStyle = "#ff4466"
        ctx.beginPath()
        for i in range(self.n):
            if self.pin_mask[i]:
                ctx.moveTo(pos[i][0] + 5, pos[i][1])
                ctx.arc(pos[i][0], pos[i][1], 5, 0, TAU)
        ctx.fill()

        if self.grabbed >= 0:
            gi = self.grabbed
            ctx.fillStyle = "#ffee44"
            ctx.beginPath()
            ctx.arc(pos[gi][0], pos[gi][1], 6, 0, TAU)
            ctx.fill()

        # HUD
        ctx.fillStyle = "#667788"
        ctx.font = "12px monospace"
        ctx.textAlign = "left"
        name = SCENE_NAMES[self.scene_id] if self.scene_id < len(SCENE_NAMES) else "?"
        ctx.fillText(
            f"Particles: {self.n}  |  Constraints: {self.nc}  |  XPBD Solver",
            10, 20,
        )
        ctx.fillText(
            f"Scene: {name}  |  Drag particles with mouse", 10, 36
        )


# Bootstrap
sim = Simulation()


def _find_nearest(mx, my, radius=30.0):
    best, bd = -1, radius
    for i in range(sim.n):
        d = ((sim.pos[i, 0] - mx) ** 2 + (sim.pos[i, 1] - my) ** 2) ** 0.5
        if d < bd:
            bd, best = d, i
    return best


def on_mouse_down(e):
    rect = sim.canvas.getBoundingClientRect()
    mx, my = e.clientX - rect.left, e.clientY - rect.top
    sim.mouse[:] = [mx, my]
    sim.grabbed = _find_nearest(mx, my)


def on_mouse_move(e):
    rect = sim.canvas.getBoundingClientRect()
    sim.mouse[:] = [e.clientX - rect.left, e.clientY - rect.top]


def on_mouse_up(_):
    sim.grabbed = -1


def on_touch_start(e):
    e.preventDefault()
    t = e.touches[0]
    rect = sim.canvas.getBoundingClientRect()
    mx, my = t.clientX - rect.left, t.clientY - rect.top
    sim.mouse[:] = [mx, my]
    sim.grabbed = _find_nearest(mx, my, 40.0)


def on_touch_move(e):
    e.preventDefault()
    t = e.touches[0]
    rect = sim.canvas.getBoundingClientRect()
    sim.mouse[:] = [t.clientX - rect.left, t.clientY - rect.top]


def on_touch_end(_):
    sim.grabbed = -1


def on_scene_change(_):
    sim.setup_scene(int(document.getElementById("scene").value))


_proxies = [
    ("mousedown",  on_mouse_down),
    ("mousemove",  on_mouse_move),
    ("mouseup",    on_mouse_up),
    ("mouseleave", on_mouse_up),
    ("touchstart", on_touch_start),
    ("touchmove",  on_touch_move),
    ("touchend",   on_touch_end),
    ("touchcancel",on_touch_end),
]
for evt, fn in _proxies:
    sim.canvas.addEventListener(evt, create_proxy(fn))

document.getElementById("scene").addEventListener("change", create_proxy(on_scene_change))


def _loop(_ts):
    sim.step()
    window.requestAnimationFrame(_lp)

_lp = create_proxy(_loop)
window.requestAnimationFrame(_lp)
