"""Instanced signal furniture and lamp records driven by the real controllers."""
from __future__ import annotations

import math
import numpy as np

from ..core.photometry import LUMINANCE_NITS
from .traffic_intersections import SignalController, SignalState
from ..vehicles.vehicle_instances import VEHICLE_STRIDE
from ..vehicles.vehicle_models import _Mesh, P_MATTE

P_SIGNAL_RED, P_SIGNAL_YELLOW, P_SIGNAL_GREEN, P_SIGNAL_LEFT = 18, 19, 20, 21
LEFT_ARROW = 16
SIGNAL_CENTER = np.array([0., 1.6, 2.8], np.float32)
SIGNAL_RADIUS = 3.5  # encloses pole, mast, hooded heads, backplates and the separate arrow head
LENS_ANCHORS = np.array([(-.216, 2.4, 5.10), (-.216, 2.4, 4.72),
                         (-.216, 2.4, 4.34), (-.216, 3.05, 4.72)], np.float32)
# LED signal colours: deep red, amber, and the blue-leaning green of real LED modules.
# (vehicle.py hard-codes the same lens centres for its LED-dot pattern.)
LENS_COLORS = np.array([(1., .045, .02), (1., .56, .012),
                        (.03, 1., .50), (.03, 1., .50)], np.float32)
CURB_CLEARANCE = .65
CURB_HEIGHT = .17

# Dark green-black polycarbonate housings, galvanised steel, retro-reflective backplate border.
HOUSING = (.026, .029, .028)
HOUSING_EDGE = (.048, .053, .050)
VISOR_INNER = (.004, .004, .004)
STEEL = (.20, .21, .21)
BACKPLATE = (.012, .013, .013)
BACKPLATE_BORDER = (.46, .40, .11)
FRONT_X = -.205          # housing front face; lenses sit .011 proud, inside the hood
LENS_RADIUS = .145
HEAD_Y, ARROW_Y = 2.4, 3.05
ARM_Z = 5.48


def _ring_z(z, r, n=10):
    a = np.linspace(0., 2. * np.pi, n, endpoint=False)
    return np.stack([r * np.cos(a), r * np.sin(a), np.full(n, z)], axis=1)


def _ring_y(y, r, z0, n=10):
    a = np.linspace(0., 2. * np.pi, n, endpoint=False)
    return np.stack([r * np.cos(a), np.full(n, y), z0 + r * np.sin(a)], axis=1)


def _disc_pt(x, yc, zc, r, a):
    """Point on a circle in the lens plane; a = 0 is straight up."""
    return (x, yc + r * np.sin(a), zc + r * np.cos(a))


def _visor(mesh, yc, zc, x0, length, r_in=.166, r_out=.180, half=np.radians(128.), segs=12):
    """Open-bottom cap hood: lit outer skin, black inner skin, rim and both end faces."""
    x1 = x0 - length
    ang = np.linspace(-half, half, segs + 1)
    for a, b in zip(ang[:-1], ang[1:]):
        m = .5 * (a + b)
        out = (0., np.sin(m), np.cos(m))
        mesh.quad(_disc_pt(x0, yc, zc, r_out, a), _disc_pt(x1, yc, zc, r_out, a),
                  _disc_pt(x1, yc, zc, r_out, b), _disc_pt(x0, yc, zc, r_out, b),
                  HOUSING, P_MATTE, hint=out)
        mesh.quad(_disc_pt(x0, yc, zc, r_in, a), _disc_pt(x1, yc, zc, r_in, a),
                  _disc_pt(x1, yc, zc, r_in, b), _disc_pt(x0, yc, zc, r_in, b),
                  VISOR_INNER, P_MATTE, hint=(0., -out[1], -out[2]))
        mesh.quad(_disc_pt(x1, yc, zc, r_in, a), _disc_pt(x1, yc, zc, r_out, a),
                  _disc_pt(x1, yc, zc, r_out, b), _disc_pt(x1, yc, zc, r_in, b),
                  HOUSING_EDGE, P_MATTE, hint=(-1., 0., 0.))
    for a, s in ((ang[0], -1.), (ang[-1], 1.)):
        mesh.quad(_disc_pt(x0, yc, zc, r_in, a), _disc_pt(x1, yc, zc, r_in, a),
                  _disc_pt(x1, yc, zc, r_out, a), _disc_pt(x0, yc, zc, r_out, a),
                  HOUSING, P_MATTE, hint=(0., s * np.cos(a), -s * np.sin(a)))


def _bezel(mesh, yc, zc, segs=16):
    """Black ring that frames the lens so its edge reads crisply."""
    ang = np.linspace(0., 2. * np.pi, segs + 1)
    x = FRONT_X - .006
    for a, b in zip(ang[:-1], ang[1:]):
        mesh.quad(_disc_pt(x, yc, zc, LENS_RADIUS - .004, a), _disc_pt(x, yc, zc, LENS_RADIUS + .024, a),
                  _disc_pt(x, yc, zc, LENS_RADIUS + .024, b), _disc_pt(x, yc, zc, LENS_RADIUS - .004, b),
                  VISOR_INNER, P_MATTE, hint=(-1., 0., 0.))


def _backplate(mesh, yc, zc, w, h):
    x = .052
    mesh.box(x, yc, zc, .022, w, h, BACKPLATE, P_MATTE)
    b, xf = .030, x - .0125
    mesh.box(xf, yc, zc + h / 2 - b / 2 - .008, .004, w - .016, b, BACKPLATE_BORDER, P_MATTE)
    mesh.box(xf, yc, zc - h / 2 + b / 2 + .008, .004, w - .016, b, BACKPLATE_BORDER, P_MATTE)
    mesh.box(xf, yc - w / 2 + b / 2 + .008, zc, .004, b, h - .016, BACKPLATE_BORDER, P_MATTE)
    mesh.box(xf, yc + w / 2 - b / 2 - .008, zc, .004, b, h - .016, BACKPLATE_BORDER, P_MATTE)


def _housing(mesh, yc, zs, width, sec_h, depth=.30):
    cx = FRONT_X + depth / 2
    for z in zs:
        mesh.box(cx, yc, z, depth, width, sec_h, HOUSING, P_MATTE)
    for za, zb in zip(zs[:-1], zs[1:]):  # moulded seam between sections
        mesh.box(cx, yc, .5 * (za + zb), depth + .016, width + .016, .020, HOUSING_EDGE, P_MATTE)
    mesh.box(cx, yc, max(zs) + sec_h / 2 + .010, depth + .030, width + .030, .022, HOUSING_EDGE, P_MATTE)
    mesh.box(cx, yc, min(zs) - sec_h / 2 - .010, depth + .030, width + .030, .022, HOUSING_EDGE, P_MATTE)
    return max(zs) + sec_h / 2 + .021


def build_signal_template():
    """Mast-arm signal, +x forward; hooded lenses face the approaching car (-x).

    Dark polycarbonate heads with cap visors, black backplates with a reflective border,
    tapered galvanised pole and arm. Lens centres are unchanged (LENS_ANCHORS) so the
    glow sprites and the LED-dot shader line up with the geometry.
    """
    mesh = _Mesh()
    # tapered octagonal pole, base plate and collar
    mesh.box(0., 0., .020, .46, .46, .040, STEEL, P_MATTE)
    mesh.loft(np.array([_ring_z(.04, .155), _ring_z(.30, .135), _ring_z(.70, .105),
                        _ring_z(5.50, .072)]), STEEL, P_MATTE, cap_first=False, cap_last=False)
    # tapered mast arm, junction box and end plug
    mesh.box(0., .02, ARM_Z, .20, .20, .27, STEEL, P_MATTE)
    mesh.loft(np.array([_ring_y(-.10, .085, ARM_Z), _ring_y(1.50, .066, ARM_Z),
                        _ring_y(3.50, .045, ARM_Z)]), STEEL, P_MATTE, cap_first=False, cap_last=False)
    mesh.box(0., 3.51, ARM_Z, .09, .03, .09, STEEL, P_MATTE)

    # three-section head
    zs = [float(z) for z in LENS_ANCHORS[:3, 2]]
    top = _housing(mesh, HEAD_Y, zs, .42, .37)
    _backplate(mesh, HEAD_Y, float(np.mean(zs)), .66, 1.34)
    mesh.box(FRONT_X + .15, HEAD_Y, .5 * (top + ARM_Z - .06), .10, .10, ARM_Z - .06 - top + .02, HOUSING, P_MATTE)
    for anchor, color in zip(LENS_ANCHORS[:3], LENS_COLORS[:3]):
        _bezel(mesh, HEAD_Y, float(anchor[2]))
        _visor(mesh, HEAD_Y, float(anchor[2]), FRONT_X, .30)
    for i, (anchor, color) in enumerate(zip(LENS_ANCHORS[:3], LENS_COLORS[:3])):
        mesh.disc(anchor, LENS_RADIUS, (-1, 0, 0), color, P_SIGNAL_RED + i, segs=20, axis='x')

    # separate left-arrow head
    x, y, z = (float(v) for v in LENS_ANCHORS[3])
    atop = _housing(mesh, ARROW_Y, [z], .44, .44)
    _backplate(mesh, ARROW_Y, z, .64, .66)
    mesh.box(FRONT_X + .15, ARROW_Y, .5 * (atop + ARM_Z - .06), .10, .10, ARM_Z - .06 - atop + .02, HOUSING, P_MATTE)
    _visor(mesh, ARROW_Y, z, FRONT_X, .24, r_in=.210, r_out=.224)
    color = LENS_COLORS[3]
    # A real left arrow, not another green round lens.
    # From the approaching driver's view +y is screen-left, not screen-right.
    mesh.quad((x, y-.16, z-.035), (x, y+.025, z-.035),
              (x, y+.025, z+.035), (x, y-.16, z+.035), color, P_SIGNAL_LEFT, hint=(-1, 0, 0))
    mesh.triangle((x, y+.16, z), (x, y+.015, z+.15),
                  (x, y+.015, z-.15), color, P_SIGNAL_LEFT, hint=(-1, 0, 0))
    return mesh.template()


class SignalStream:
    """Worker-owned static placements with one packed phase update per publication."""
    def __init__(self, graph, manager):
        self.manager = manager
        self.approaches = [(n.id, e) for n in graph.nodes if n.signal for e in n.incoming]
        nodes = tuple(dict.fromkeys(n for n, _ in self.approaches))
        controllers = {node: i for i, node in enumerate(nodes)}
        self._controllers = tuple(manager.signals[n] for n in nodes)
        self._controller_rows = np.array([controllers[n] for n, _ in self.approaches], np.int64)
        self._groups = np.array([manager.signals[n].groups.get(e, 0) for n, e in self.approaches])
        # A tiny negative time can round modulo to PERIOD itself. Match the
        # scalar controller's all-red fallthrough rather than indexing past it.
        self._phase_groups = np.array([p[0] for p in SignalController.PHASES] + [-1])
        self._phase_states = np.array([p[1] for p in SignalController.PHASES] + [SignalState.RED], np.int32)
        self.base = np.zeros((len(self.approaches), VEHICLE_STRIDE), np.float32)
        for i, (node_id, eid) in enumerate(self.approaches):
            node, edge = graph.nodes[node_id], graph.edges[eid]
            lane = graph.lanes[edge.lanes[-1]]
            yaw = lane.heading[-1]
            c, s = math.cos(yaw), math.sin(yaw)
            forward, right = np.array([c, s]), np.array([s, -c])
            lateral = edge.width * .5 + CURB_CLEARANCE
            far_side = 3.  # standalone crossing: ahead of its stop line
            for other_id in (*node.incoming, *node.outgoing):
                other = graph.edges[other_id]
                angle = other.heading_end if other.end == node_id else other.heading_start
                normal = np.array([-math.sin(angle), math.cos(angle)])
                crossing = float(normal @ forward)
                if abs(crossing) > .35:
                    # The far-right sidewalk corner must also clear the crossing
                    # street, including skew intersections, not just our own lane.
                    corner = (other.width * .5 + CURB_CLEARANCE
                              - math.copysign(1., crossing) * float(normal @ right) * lateral)
                    far_side = max(far_side, corner / abs(crossing))
            row = self.base[i]
            row[:3], row[4:7], row[8:11] = (c, s, 0.), (-s, c, 0.), (0., 0., 1.)
            row[12:14] = node.position[:2] + forward * far_side + right * lateral
            row[14] = node.position[2] + math.tan(lane.pitch[-1]) * far_side + CURB_HEIGHT
            row[15], row[23], row[16:19] = 1., 1., (.17, .18, .19)

    def snapshot(self, clock):
        if not len(self.base): return None
        rows = self.base.copy()
        # The actuated controllers own their current phase slot; broadcast it
        # from each junction to all of its approaches.
        slots = np.fromiter((s.index for s in self._controllers), np.int64,
                            count=len(self._controllers))
        phases = slots[self._controller_rows]
        own = self._phase_groups[phases] == self._groups
        states = self._phase_states[phases]
        left = states == SignalState.DEDICATED_LEFT
        rows[:, 19] = (np.where(own & ~left, states, SignalState.RED)
                       | np.where(own & left, LEFT_ARROW, 0))
        rows.flags.writeable = False
        return rows


def signal_lamps(rows, eye, far=1200.):
    """Active lenses only, using the same centres/colors/state as the solid mesh."""
    if rows is None or not len(rows): return None
    rows = rows[np.sum((rows[:, 12:15] - eye) ** 2, axis=1) < far**2]
    if not len(rows): return None
    flags = rows[:, 19].astype(np.int32)
    states = flags & 3
    active = np.c_[states == SignalState.RED, states == SignalState.YELLOW,
                   states == SignalState.GREEN, (flags & LEFT_ARROW) != 0]
    lamps = np.empty((len(rows), 4, 8), np.float32)
    lamps[:, :, :3] = (rows[:, None, 12:15] + rows[:, None, :3] * LENS_ANCHORS[None, :, :1]
                       + rows[:, None, 4:7] * LENS_ANCHORS[None, :, 1:2]
                       + rows[:, None, 8:11] * LENS_ANCHORS[None, :, 2:3])
    direction = np.asarray(eye) - lamps[:, :, :3]
    cosine = np.sum(direction * (-rows[:, None, :3]), axis=2) / np.maximum(np.linalg.norm(direction, axis=2), 1e-6)
    angle = np.clip(cosine / .25, 0., 1.)
    angle = angle**2 * (3. - 2. * angle)  # same smoothstep(0, .25, NdotV) as the mesh
    lamps[:, :, 3], lamps[:, :, 4:7] = .12, LENS_COLORS
    lamps[:, :, 7] = LUMINANCE_NITS['TRAFFIC_SIGNAL'] * rows[:, None, 23] * angle
    result = lamps[active & (lamps[:, :, 7] > 0.)]
    return result if len(result) else None
