"""Lightweight camera-depth-aware contact drawings, without selection or ROS."""
import cv2
import numpy as np
from .press_targets import pre_press_center

# Canonical per-mode identity color (BGR), shared by the live orchestrator's cheap overlay
# and every offline/montage/3D rendering below -- one place, so a mode always looks the same.
CONTACT_COLORS = {"planar_push": (0, 200, 0), "forward_tip": (13, 140, 255), "press": (255, 0, 255)}


def mode_palette(mode_color=None):
    """BGR colors for pivot / contact-ball / pre-contact-ball / contact-point.

    Shared by contact_graphic()'s 2D overlay and any 3D contact-marker generation, so they
    never drift apart. pre-contact-ball and contact-point are fixed across every mode (cyan,
    yellow) so those two roles stay recognizable regardless of which mode drew them; mode_color
    (a mode's identity color) tints pivot and contact-ball together, since those are what need
    to stay distinguishable *between* modes when several share one image. None (default) is
    press's identity color, magenta -- used by callers that don't care about multi-mode
    disambiguation (a lone press image, or a generic single-candidate preview).
    """
    identity = mode_color or CONTACT_COLORS['press']
    return {'pivot': identity, 'ball': identity, 'pre': (255, 255, 0), 'point': (0, 255, 255)}


class DepthDrawing:
    """Solid visible geometry, dotted occluded or unknown-depth geometry.

    Aligned uint16 millimetre depth is required. Three millimetres of tolerance
    avoids drawing measured surface contacts as hidden due to quantization.
    This is an image diagnostic, not a collision or visibility guarantee.
    """
    def __init__(self, color, depth, info, occlusion=True):
        self.image = color.copy()
        self.depth = depth
        self.occlusion = occlusion
        self.K = np.asarray(info['K'], dtype=float)
        self.D = np.asarray(info.get('D', []), dtype=float)
        self.R = np.asarray(info['base_to_optical']['R'], dtype=float)
        self.t = np.asarray(info['base_to_optical']['t'], dtype=float)
        self.last_dot = {}

    def project(self, points):
        optical = np.atleast_2d(points) @ self.R.T + self.t
        uv, _ = cv2.projectPoints(optical, np.zeros(3), np.zeros(3), self.K, self.D)
        return uv.reshape(-1, 2), optical[:, 2]

    def visibility(self, points):
        uv, z = self.project(points)
        height, width = self.image.shape[:2]
        valid = np.isfinite(uv).all(axis=1) & (z > .05)
        pixels = np.rint(np.nan_to_num(uv)).astype(int)
        valid &= ((pixels[:, 0] >= 0) & (pixels[:, 0] < width)
                  & (pixels[:, 1] >= 0) & (pixels[:, 1] < height))
        state = np.full(len(z), 'outside', dtype='<U8')
        state[valid] = 'unknown'
        if not self.occlusion:
            state[valid] = 'visible'
            return pixels, state
        if self.depth is not None and self.depth.shape == (height, width):
            ids = np.flatnonzero(valid)
            measured = self.depth[pixels[ids, 1], pixels[ids, 0]].astype(float)/1000
            known = measured > 0
            state[ids[known]] = np.where(z[ids[known]] > measured[known]+.003,
                                         'hidden', 'visible')
        return pixels, state

    def line(self, a, b, color, arrow=False):
        ends, z = self.project([a, b])
        if np.any(z <= .05) or not np.isfinite(ends).all():
            return
        if not self.occlusion:
            a_px, b_px = [tuple(np.rint(p).astype(int)) for p in ends]
            if arrow:
                cv2.arrowedLine(self.image, a_px, b_px, color, 1, cv2.LINE_AA, tipLength=.2)
            else:
                cv2.line(self.image, a_px, b_px, color, 1, cv2.LINE_AA)
            return
        length = np.linalg.norm(ends[1]-ends[0])
        count = min(4096, max(2, int(np.ceil(length*2))+1))
        points = np.linspace(a, b, count)
        pixels, states = self.visibility(points)
        for i, (pixel, state) in enumerate(zip(pixels, states)):
            if i and state == states[i-1] == 'visible':
                cv2.line(self.image, tuple(pixels[i-1]), tuple(pixel), color, 1, cv2.LINE_AA)
            elif state != 'outside' and np.linalg.norm(pixel-self.last_dot.get(color, pixel-100)) >= 7:
                shade = color if state != 'unknown' else tuple(int(c*.55) for c in color)
                cv2.circle(self.image, tuple(pixel), 1, shade, -1, cv2.LINE_AA)
                self.last_dot[color] = pixel.copy()
        if arrow and states[-1] != 'outside' and length > 1:
            unit = (ends[1]-ends[0])/length
            side = np.array([-unit[1], unit[0]])
            for sign in (-1, 1):
                tip = ends[1]-8*unit+sign*4*side
                if states[-1] == 'visible':
                    cv2.line(self.image, tuple(pixels[-1]), tuple(np.rint(tip).astype(int)), color, 1, cv2.LINE_AA)
                else:
                    shade = color if states[-1] == 'hidden' else tuple(int(c*.55) for c in color)
                    for p in np.linspace(ends[1], tip, 3):
                        cv2.circle(self.image, tuple(np.rint(p).astype(int)), 1, shade, -1)

    def marker(self, point, color):
        pixels, states = self.visibility([point])
        if states[0] == 'visible':
            cv2.circle(self.image, tuple(pixels[0]), 3, color, -1, cv2.LINE_AA)
        elif states[0] != 'outside':
            shade = color if states[0] == 'hidden' else tuple(int(c*.55) for c in color)
            for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
                p = pixels[0]+np.rint(4*np.array([np.cos(angle), np.sin(angle)])).astype(int)
                cv2.circle(self.image, tuple(p), 1, shade, -1)


def contact_graphic(color, depth, info, contact=None, geometry=None, occlusion=False,
                    pre=None, reference_arrow=True, mode_color=None, mode_label=None):
    """Pivot, contact and fingertip spheres at the contact and pre-contact centers.

    pre: pre-contact ball center (base_link); default is the press pre-squash center.
    reference_arrow: the grey -X arrow at the pivot (press-pull's pull direction).
    mode_color: BGR tint, see mode_palette() -- lets several modes share one image and stay
    distinguishable; None keeps the original single-mode palette.
    mode_label: optional text drawn by the pivot (or contact, if no pivot) naming the mode.
    """
    drawing = DepthDrawing(color, depth, info, occlusion=occlusion)
    palette = mode_palette(mode_color)
    if geometry and geometry.get('pivot') is not None:
        pivot = np.asarray(geometry['pivot']); axis = np.asarray(geometry['axis'])
        drawing.line(pivot-.04*axis, pivot+.04*axis, palette['pivot'])
        if reference_arrow:
            drawing.line(pivot, pivot+[-.06, 0, 0], (220, 220, 220), arrow=True)
        drawing.line(pivot, pivot+.06*np.asarray(geometry['direction']), palette['point'], arrow=True)
        drawing.marker(pivot, palette['pivot'])
        if mode_label:
            uv, z = drawing.project([pivot])
            if z[0] > .05 and np.isfinite(uv).all():
                pixel = tuple(np.rint(uv[0]+[8, 16]).astype(int))
                cv2.putText(drawing.image, mode_label, pixel, cv2.FONT_HERSHEY_SIMPLEX, .4,
                            palette['pivot'], 1, cv2.LINE_AA)
    if contact and contact.get('point') is not None:
        point = np.asarray(contact['point']); center = np.asarray(contact['ball_center'])
        radius = float(np.linalg.norm(center-point))
        camera = -drawing.R.T @ drawing.t
        pre_press = pre_press_center(center) if pre is None else np.asarray(pre, dtype=float)
        for ring_center, ring_color in [(center, palette['ball']), (pre_press, palette['pre'])]:
            delta = camera-ring_center; distance = np.linalg.norm(delta)
            if distance > radius > 0:
                normal = delta/distance
                reference = np.eye(3)[np.argmin(np.abs(normal))]
                u = np.cross(normal, reference); u /= np.linalg.norm(u)
                v = np.cross(normal, u)
                theta = np.linspace(0, 2*np.pi, 65)
                ring = (ring_center+radius**2/distance*normal + radius*np.sqrt(1-(radius/distance)**2)*
                        (np.cos(theta)[:, None]*u+np.sin(theta)[:, None]*v))
                for a, b in zip(ring[:-1], ring[1:]):
                    drawing.line(a, b, ring_color)
        drawing.line(pre_press, center, palette['pre'], arrow=True)
        drawing.marker(pre_press, palette['pre'])
        drawing.marker(center, palette['ball'])
        for name, position, shade in [("PRE", pre_press, palette['pre']),
                                      ("CONTACT", point, palette['point'])]:
            uv, z = drawing.project([position])
            if z[0] > .05 and np.isfinite(uv).all():
                pixel = tuple(np.rint(uv[0]+[8, -5]).astype(int))
                cv2.putText(drawing.image, name, pixel, cv2.FONT_HERSHEY_SIMPLEX, .35, shade, 1, cv2.LINE_AA)
        drawing.marker(point, palette['point'])
        if mode_label and (not geometry or geometry.get('pivot') is None):
            uv, z = drawing.project([point])
            if z[0] > .05 and np.isfinite(uv).all():
                pixel = tuple(np.rint(uv[0]+[8, 16]).astype(int))
                cv2.putText(drawing.image, mode_label, pixel, cv2.FONT_HERSHEY_SIMPLEX, .4,
                            palette['point'], 1, cv2.LINE_AA)
    return drawing.image


def final_selection_montage(review_frames, contacts, forward_pre=None):
    """Full camera views of the final contacts, one image per camera hstacked together.

    Press and forward_tip both get the full contact_graphic() ball/sphere rendering (pivot,
    pre-contact and contact fingertip spheres), color-coded by mode (press: orange pivot /
    magenta ball / cyan pre, the original single-mode palette; forward_tip: its CONTACT_COLORS
    hue throughout) so the two stay distinguishable merged onto one image. planar_push has no
    pivot or pre-contact to show, so it stays a single colored marker + direction arrow + label.
    None if press is unavailable, or none of review_frames' cameras have one.

    review_frames: {cam: (color, depth, info)}. contacts: select_contact_points() output
    (mode -> dict with point/geometry/available/direction), same shape the live orchestrator
    and the offline FFS adapter both produce. forward_pre: forward-tip's pre-contact ball
    center (base_link), e.g. from forward_tip_motion.tip_targets(...)[3] -- without it,
    forward_tip falls back to a plain marker (no pre/contact spheres; its real pre-tip position
    depends on the object cloud and tipping-edge geometry, so it isn't guessed here).
    """
    press = contacts.get("press") or {}
    if not press.get("available"):
        return None
    forward = contacts.get("forward_tip") or {}
    push = contacts.get("planar_push") or {}
    panels = []
    for cam in ("cam1", "cam2", "cam3"):
        if cam not in review_frames:
            continue
        color, depth, info = review_frames[cam]
        img = contact_graphic(color, depth, info, press, press.get("geometry"), mode_label="PRESS")
        if forward.get("available") and forward_pre is not None:
            img = contact_graphic(img, depth, info, forward, forward.get("geometry"),
                                  pre=forward_pre, reference_arrow=False,
                                  mode_color=CONTACT_COLORS["forward_tip"], mode_label="FORWARD TIP")
        drawing = DepthDrawing(img, depth, info, occlusion=False)
        if forward.get("available") and forward_pre is None:
            # No pre-contact position supplied: plain marker, same as planar_push below.
            shade = CONTACT_COLORS["forward_tip"]
            point = np.asarray(forward["point"])
            drawing.marker(point, shade)
            geometry = forward.get("geometry") or {}
            if geometry.get("pivot") is not None:
                pivot, axis = np.asarray(geometry["pivot"]), np.asarray(geometry["axis"])
                drawing.line(pivot - .04*axis, pivot + .04*axis, shade)
                drawing.marker(pivot, shade)
            direction = geometry.get("direction")
            if direction is not None:
                drawing.line(point, point+.04*np.asarray(direction), shade, arrow=True)
            uv, z = drawing.project([point])
            if z[0] > .05 and np.isfinite(uv).all():
                cv2.putText(drawing.image, "FORWARD TIP", tuple(np.rint(uv[0]+[8, 20]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, .4, shade, 1, cv2.LINE_AA)
        if push.get("available"):
            shade = CONTACT_COLORS["planar_push"]
            point = np.asarray(push["point"])
            drawing.marker(point, shade)
            direction = (push.get("geometry") or {}).get("direction", push.get("direction"))
            if direction is not None:
                drawing.line(point, point+.04*np.asarray(direction), shade, arrow=True)
            uv, z = drawing.project([point])
            if z[0] > .05 and np.isfinite(uv).all():
                cv2.putText(drawing.image, "PUSH", tuple(np.rint(uv[0]+[8, -6]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, .4, shade, 1, cv2.LINE_AA)
        panels.append(drawing.image)
    if not panels:
        return None
    height = min(p.shape[0] for p in panels)
    return np.hstack([cv2.resize(p, (int(p.shape[1]*height/p.shape[0]), height)) for p in panels])
