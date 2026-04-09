import numpy as np
from js import document, window
from pyodide.ffi import create_proxy

from constants import W, H, FLOOR_Y, TAU, DT
from solvers import SOLVERS
from scenes import SCENES

SCENE_NAMES = [s.name for s in SCENES]

# Lighting
LIGHT_DIR = np.array([0.3, 0.7, 0.5])
LIGHT_DIR = LIGHT_DIR / np.linalg.norm(LIGHT_DIR)

# Orbit camera with perspective projection
class Camera:
    def __init__(self):
        self.theta: float = 0.5
        self.phi: float = 0.35
        self.distance: float = 800.0
        self.target = np.array([0.0, 100.0, 0.0])
        self.fov: float = 45.0

    def position(self):
        st, ct = np.sin(self.theta), np.cos(self.theta)
        sp, cp = np.sin(self.phi), np.cos(self.phi)
        return self.target + self.distance * np.array([cp * st, sp, cp * ct])

    def basis(self):
        pos = self.position()
        fwd = self.target - pos
        fwd = fwd / np.linalg.norm(fwd)
        right = np.cross(fwd, np.array([0.0, 1.0, 0.0]))
        rn = np.linalg.norm(right)
        if rn < 1e-6:
            right = np.array([1.0, 0.0, 0.0])
        else:
            right = right / rn
        up = np.cross(right, fwd)
        return fwd, right, up, pos

    def project(self, pts):
        fwd, right, up, pos = self.basis()
        rel = pts - pos
        x = rel @ right
        y = rel @ up
        z = rel @ fwd
        f = 1.0 / np.tan(np.radians(self.fov / 2))
        half = min(W, H) / 2
        z_safe = np.maximum(z, 1.0)
        sx = W / 2 + x / z_safe * f * half
        sy = H / 2 - y / z_safe * f * half
        return sx, sy, z

    def screen_to_world(self, mx, my, depth):
        fwd, right, up, pos = self.basis()
        f = 1.0 / np.tan(np.radians(self.fov / 2))
        half = min(W, H) / 2
        rx = (mx - W / 2) / (f * half)
        ry = -(my - H / 2) / (f * half)
        ray = fwd + rx * right + ry * up
        t = depth / np.dot(ray, fwd)
        return pos + ray * t


# Simulation
class Simulation:

    def __init__(self) -> None:
        # Particle state 
        self.n: int = 0
        self.pos = np.zeros((0, 3))
        self.vel = np.zeros((0, 3))
        self.predicted = np.zeros((0, 3))
        self.inv_mass = np.zeros(0)
        self.pin_mask = np.zeros(0, dtype=bool)
        self.pinned_pos = np.zeros((0, 3))

        # Constraints
        self.c_i = np.zeros(0, dtype=np.int32)
        self.c_j = np.zeros(0, dtype=np.int32)
        self.c_rest = np.zeros(0)
        self.c_wi = np.zeros(0)
        self.c_wj = np.zeros(0)
        self.c_wsum = np.zeros(0)
        self.nc: int = 0
        self.struct_pairs: list = []
        self.shear_pairs: list = []

        # Triangle faces (built by _build_cloth)
        self.face_i0 = np.zeros(0, dtype=np.int32)
        self.face_i1 = np.zeros(0, dtype=np.int32)
        self.face_i2 = np.zeros(0, dtype=np.int32)
        self.n_faces: int = 0

        self.rows = self.cols = 0

        # Solver params
        self.iterations: int = 15
        self.stiffness: float = 0.80
        self.damping: float = 0.998
        self.restitution: float = 0.30
        self.floor_friction: float = 0.25
        self.self_collision: bool = False

        # Ball
        self.ball_pos = np.array([0.0, 50.0, 80.0])
        self.ball_radius: float = 60.0
        self.ball_active: bool = True

        # Interaction
        self.grabbed: int = -1
        self.grab_depth: float = 0.0
        self.mouse = np.array([0.0, 0.0, 0.0])
        self.particle_r: float = 2.5

        self.orbiting: bool = False
        self.orbit_last = (0, 0)

        self.scene_id: int = 0
        self.solver_idx: int = 0

        self.camera = Camera()
        self.canvas = document.getElementById("c")
        self.ctx = self.canvas.getContext("2d")

        self._init_floor_grid()
        self.setup_scene(0)


    # Floor grid (for rendering)
    def _init_floor_grid(self):
        pts, pairs = [], []
        extent, step, idx = 400, 40, 0
        for x in range(-extent, extent + 1, step):
            pts.append([x, FLOOR_Y, -extent])
            pts.append([x, FLOOR_Y, extent])
            pairs.append((idx, idx + 1)); idx += 2
        for z in range(-extent, extent + 1, step):
            pts.append([-extent, FLOOR_Y, z])
            pts.append([extent, FLOOR_Y, z])
            pairs.append((idx, idx + 1)); idx += 2
        self._floor_pts = np.array(pts, dtype=np.float64)
        self._floor_pairs = pairs

    # Cloth builder  (3-D)
    def _build_cloth(self, rows, cols, sp, center, pin_mode, plane="xy"):
        cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
        self.rows, self.cols = rows, cols
        self.n = rows * cols
        n = self.n
        hw = (cols - 1) * sp / 2
        hh = (rows - 1) * sp / 2

        self.pos = np.zeros((n, 3))
        for r in range(rows):
            for c in range(cols):
                idx = r * cols + c
                if plane == "xy":
                    self.pos[idx] = [cx - hw + c * sp, cy + hh - r * sp, cz]
                else:
                    self.pos[idx] = [cx - hw + c * sp, cy, cz - hh + r * sp]

        self.vel = np.zeros((n, 3))
        self.predicted = self.pos.copy()
        self.inv_mass = np.ones(n)
        self.pin_mask = np.zeros(n, dtype=bool)

        pm = pin_mode
        if pm == "top":
            self.pin_mask[:cols] = True
        elif pm == "corners":
            self.pin_mask[0] = True
            self.pin_mask[cols - 1] = True
        elif pm == "single_center":
            self.pin_mask[cols // 2] = True
        elif pm == "three_top":
            self.pin_mask[0] = True
            self.pin_mask[cols // 2] = True
            self.pin_mask[cols - 1] = True
        elif pm == "four_corners":
            self.pin_mask[0] = True
            self.pin_mask[cols - 1] = True
            self.pin_mask[(rows - 1) * cols] = True
            self.pin_mask[(rows - 1) * cols + cols - 1] = True
        elif pm == "sides":
            for r in range(rows):
                self.pin_mask[r * cols] = True
                self.pin_mask[r * cols + cols - 1] = True

        self.inv_mass[self.pin_mask] = 0.0
        self.pinned_pos = self.pos.copy()

        # Distance constraints
        ci, cj, cr = [], [], []
        self.struct_pairs, self.shear_pairs = [], []
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
                    ci.append(p); cj.append(p + cols + 1)
                    cr.append(sp * 1.41421356)
                    self.shear_pairs.append(idx); idx += 1
                if r < rows - 1 and c > 0:
                    ci.append(p); cj.append(p + cols - 1)
                    cr.append(sp * 1.41421356)
                    self.shear_pairs.append(idx); idx += 1
                if c < cols - 2:
                    ci.append(p); cj.append(p + 2); cr.append(sp * 2); idx += 1
                if r < rows - 2:
                    ci.append(p); cj.append(p + 2 * cols); cr.append(sp * 2); idx += 1

        self.c_i = np.array(ci, dtype=np.int32)
        self.c_j = np.array(cj, dtype=np.int32)
        self.c_rest = np.array(cr)
        self.nc = len(ci)
        self.c_wi = self.inv_mass[self.c_i]
        self.c_wj = self.inv_mass[self.c_j]
        self.c_wsum = self.c_wi + self.c_wj

        # Triangle faces for rendering
        fi0, fi1, fi2 = [], [], []
        for r in range(rows - 1):
            for c in range(cols - 1):
                p00 = r * cols + c
                p01 = p00 + 1
                p10 = p00 + cols
                p11 = p10 + 1
                fi0.append(p00); fi1.append(p01); fi2.append(p10)
                fi0.append(p01); fi1.append(p11); fi2.append(p10)
        self.face_i0 = np.array(fi0, dtype=np.int32)
        self.face_i1 = np.array(fi1, dtype=np.int32)
        self.face_i2 = np.array(fi2, dtype=np.int32)
        self.n_faces = len(fi0)

    def _snap_pins(self) -> None:
        self.pos[self.pin_mask] = self.pinned_pos[self.pin_mask]
        self.predicted = self.pos.copy()
        self.vel[self.pin_mask] = 0.0

    # Scene / UI
    def setup_scene(self, sid: int) -> None:
        self.scene_id = sid
        self.grabbed = -1
        SCENES[sid].setup(self)
        document.getElementById("selfcol").checked = SCENES[sid].default_self_collision

    def _read_ui(self) -> None:
        self.iterations = int(document.getElementById("iter").value)
        self.stiffness = float(document.getElementById("stiff").value)
        self.damping = float(document.getElementById("damp").value)
        self.restitution = float(document.getElementById("rest").value)
        self.floor_friction = float(document.getElementById("ffric").value)
        self.solver_idx = int(document.getElementById("solver").value)
        self.self_collision = bool(document.getElementById("selfcol").checked)

    # Step
    def step(self) -> None:
        self._read_ui()
        solver = SOLVERS[self.solver_idx]
        if not solver.implemented:
            self._render_todo(solver.name)
            return
        solver.step(self)
        self._render()

    # TODO screen
    def _render_todo(self, name: str) -> None:
        ctx = self.ctx
        ctx.fillStyle = "#1a1a2e"
        ctx.fillRect(0, 0, W, H)
        ctx.fillStyle = "#ffaa00"
        ctx.font = "bold 26px monospace"
        ctx.textAlign = "center"
        ctx.fillText(f"TODO: Implement {name}", W / 2, H / 2 - 40)
        ctx.fillStyle = "#887766"
        ctx.font = "16px monospace"
        ctx.fillText("Solver stub — see solvers/ for notes.", W / 2, H / 2)
        ctx.fillText("Switch to PBD / XPBD to run.", W / 2, H / 2 + 30)
        ctx.textAlign = "start"

    # 3-D canvas rendering
    def _render(self) -> None:
        ctx = self.ctx
        cam = self.camera

        ctx.fillStyle = "#1a1a2e"
        ctx.fillRect(0, 0, W, H)

        # --- Project cloth particles ---
        sx, sy, depth = cam.project(self.pos)
        sx_l = sx.tolist()
        sy_l = sy.tolist()
        dep_l = depth.tolist()

        # --- Floor grid ---
        fsx, fsy, fd = cam.project(self._floor_pts)
        fsx_l, fsy_l, fd_l = fsx.tolist(), fsy.tolist(), fd.tolist()
        ctx.strokeStyle = "rgba(60,60,100,0.25)"
        ctx.lineWidth = 0.5
        ctx.beginPath()
        for a, b in self._floor_pairs:
            if fd_l[a] > 0 and fd_l[b] > 0:
                ctx.moveTo(fsx_l[a], fsy_l[a])
                ctx.lineTo(fsx_l[b], fsy_l[b])
        ctx.stroke()

        # --- Ball ---
        if self.ball_active:
            bsx, bsy, bd = cam.project(self.ball_pos[np.newaxis])
            if bd[0] > 1:
                _, right, _, _ = cam.basis()
                edge = self.ball_pos + right * self.ball_radius
                esx, esy, _ = cam.project(edge[np.newaxis])
                pr = max(((esx[0] - bsx[0]) ** 2 + (esy[0] - bsy[0]) ** 2) ** 0.5, 2)
                g = ctx.createRadialGradient(
                    float(bsx[0] - pr * 0.2), float(bsy[0] - pr * 0.2),
                    pr * 0.05, float(bsx[0]), float(bsy[0]), float(pr),
                )
                g.addColorStop(0, "#666699")
                g.addColorStop(1, "#2a2a50")
                ctx.fillStyle = g
                ctx.beginPath()
                ctx.arc(float(bsx[0]), float(bsy[0]), float(pr), 0, TAU)
                ctx.fill()
                ctx.strokeStyle = "#8888bb"
                ctx.lineWidth = 1
                ctx.stroke()

        # --- Cloth faces (back then front) ---
        if self.n_faces > 0:
            v0 = self.pos[self.face_i0]
            v1 = self.pos[self.face_i1]
            v2 = self.pos[self.face_i2]
            normals = np.cross(v1 - v0, v2 - v0)
            nlen = np.linalg.norm(normals, axis=1, keepdims=True)
            normals = normals / np.maximum(nlen, 1e-7)

            cam_pos = cam.position()
            centers = (v0 + v1 + v2) / 3
            view_dir = centers - cam_pos
            dots = np.sum(normals * view_dir, axis=1)
            front = dots < 0

            center_depth = (depth[self.face_i0] + depth[self.face_i1] + depth[self.face_i2]) / 3
            visible = center_depth > 0

            fi0_l = self.face_i0.tolist()
            fi1_l = self.face_i1.tolist()
            fi2_l = self.face_i2.tolist()

            # Back faces
            ctx.fillStyle = "rgba(18,50,95,0.75)"
            ctx.beginPath()
            for fi in np.where(~front & visible)[0]:
                a, b, c_ = fi0_l[fi], fi1_l[fi], fi2_l[fi]
                ctx.moveTo(sx_l[a], sy_l[a])
                ctx.lineTo(sx_l[b], sy_l[b])
                ctx.lineTo(sx_l[c_], sy_l[c_])
                ctx.closePath()
            ctx.fill()

            # Front faces
            ctx.fillStyle = "rgba(35,105,175,0.82)"
            ctx.beginPath()
            for fi in np.where(front & visible)[0]:
                a, b, c_ = fi0_l[fi], fi1_l[fi], fi2_l[fi]
                ctx.moveTo(sx_l[a], sy_l[a])
                ctx.lineTo(sx_l[b], sy_l[b])
                ctx.lineTo(sx_l[c_], sy_l[c_])
                ctx.closePath()
            ctx.fill()

        # --- Wireframe (structural) ---
        ctx.strokeStyle = "rgba(0,200,255,0.45)"
        ctx.lineWidth = 1
        ctx.beginPath()
        for k in self.struct_pairs:
            i = int(self.c_i[k]); j = int(self.c_j[k])
            if dep_l[i] > 0 and dep_l[j] > 0:
                ctx.moveTo(sx_l[i], sy_l[i])
                ctx.lineTo(sx_l[j], sy_l[j])
        ctx.stroke()

        # --- Pinned particles ---
        ctx.fillStyle = "#ff4466"
        ctx.beginPath()
        for i in range(self.n):
            if self.pin_mask[i] and dep_l[i] > 0:
                ctx.moveTo(sx_l[i] + 5, sy_l[i])
                ctx.arc(sx_l[i], sy_l[i], 5, 0, TAU)
        ctx.fill()

        # --- Grabbed particle ---
        if self.grabbed >= 0 and dep_l[self.grabbed] > 0:
            gi = self.grabbed
            ctx.fillStyle = "#ffee44"
            ctx.beginPath()
            ctx.arc(sx_l[gi], sy_l[gi], 7, 0, TAU)
            ctx.fill()

        # --- HUD ---
        scene = SCENES[self.scene_id]
        solver = SOLVERS[self.solver_idx]
        ctx.fillStyle = "#778899"
        ctx.font = "12px monospace"
        ctx.textAlign = "left"
        ctx.fillText(
            f"Particles: {self.n}  |  Faces: {self.n_faces}  |  {solver.name}",
            10, 20,
        )
        ctx.fillText(f"Scene: {scene.name}", 10, 36)
        ctx.fillStyle = "#556677"
        ctx.fillText("Left-drag: grab  |  Right-drag: orbit  |  Scroll: zoom", 10, H - 12)



# Bootstrap
sim = Simulation()


def _canvas_xy(e):
    rect = sim.canvas.getBoundingClientRect()
    return e.clientX - rect.left, e.clientY - rect.top


def on_mouse_down(e):
    mx, my = _canvas_xy(e)
    if e.button == 0:
        sx, sy, depth = sim.camera.project(sim.pos)
        dist_sq = (sx - mx) ** 2 + (sy - my) ** 2
        dist_sq[depth <= 0] = 1e10
        best = int(np.argmin(dist_sq))
        if dist_sq[best] < 900:
            sim.grabbed = best
            sim.grab_depth = float(depth[best])
            sim.mouse = sim.camera.screen_to_world(mx, my, sim.grab_depth)
        else:
            sim.grabbed = -1
    elif e.button == 2:
        sim.orbiting = True
        sim.orbit_last = (e.clientX, e.clientY)


def on_mouse_move(e):
    if sim.orbiting:
        dx = e.clientX - sim.orbit_last[0]
        dy = e.clientY - sim.orbit_last[1]
        sim.camera.theta -= dx * 0.005
        sim.camera.phi = np.clip(sim.camera.phi + dy * 0.005, -1.3, 1.3)
        sim.orbit_last = (e.clientX, e.clientY)
    elif sim.grabbed >= 0:
        mx, my = _canvas_xy(e)
        sim.mouse = sim.camera.screen_to_world(mx, my, sim.grab_depth)


def on_mouse_up(e):
    btn = int(e.button) if hasattr(e, "button") else 0
    if btn == 0:
        sim.grabbed = -1
    elif btn == 2:
        sim.orbiting = False


def on_wheel(e):
    e.preventDefault()
    sim.camera.distance = np.clip(
        sim.camera.distance * (1 + float(e.deltaY) * 0.001), 200, 3000,
    )


def on_touch_start(e):
    e.preventDefault()
    t = e.touches[0]
    mx, my = _canvas_xy(t)
    sx, sy, depth = sim.camera.project(sim.pos)
    dist_sq = (sx - mx) ** 2 + (sy - my) ** 2
    dist_sq[depth <= 0] = 1e10
    best = int(np.argmin(dist_sq))
    if dist_sq[best] < 1600:
        sim.grabbed = best
        sim.grab_depth = float(depth[best])
        sim.mouse = sim.camera.screen_to_world(mx, my, sim.grab_depth)


def on_touch_move(e):
    e.preventDefault()
    if sim.grabbed >= 0:
        t = e.touches[0]
        mx, my = _canvas_xy(t)
        sim.mouse = sim.camera.screen_to_world(mx, my, sim.grab_depth)


def on_touch_end(_):
    sim.grabbed = -1


def on_scene_change(_):
    sim.setup_scene(int(document.getElementById("scene").value))


_handlers = [
    ("mousedown",   on_mouse_down),
    ("mousemove",   on_mouse_move),
    ("mouseup",     on_mouse_up),
    ("mouseleave",  lambda _: setattr(sim, "orbiting", False) or setattr(sim, "grabbed", -1)),
    ("wheel",       on_wheel),
    ("touchstart",  on_touch_start),
    ("touchmove",   on_touch_move),
    ("touchend",    on_touch_end),
    ("touchcancel", on_touch_end),
    ("contextmenu", lambda e: e.preventDefault()),
]
for _evt, _fn in _handlers:
    sim.canvas.addEventListener(_evt, create_proxy(_fn))

document.getElementById("scene").addEventListener("change", create_proxy(on_scene_change))


def _loop(_ts):
    sim.step()
    window.requestAnimationFrame(_lp)

_lp = create_proxy(_loop)
window.requestAnimationFrame(_lp)
