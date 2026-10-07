"""Lightweight camera-depth-aware contact drawings, without selection or ROS."""
import cv2
import numpy as np


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


def contact_graphic(color, depth, info, contact=None, geometry=None, occlusion=False):
    drawing = DepthDrawing(color, depth, info, occlusion=occlusion)
    if geometry and geometry.get('pivot') is not None:
        pivot = np.asarray(geometry['pivot']); axis = np.asarray(geometry['axis'])
        drawing.line(pivot-.04*axis, pivot+.04*axis, (0, 165, 255))
        drawing.line(pivot, pivot+[-.06, 0, 0], (220, 220, 220), arrow=True)
        drawing.line(pivot, pivot+.06*np.asarray(geometry['direction']), (0, 255, 255), arrow=True)
        drawing.marker(pivot, (0, 165, 255))
    if contact and contact.get('point') is not None:
        point = np.asarray(contact['point']); center = np.asarray(contact['ball_center'])
        radius = float(np.linalg.norm(center-point))
        camera = -drawing.R.T @ drawing.t
        delta = camera-center; distance = np.linalg.norm(delta)
        if distance > radius > 0:
            normal = delta/distance
            reference = np.eye(3)[np.argmin(np.abs(normal))]
            u = np.cross(normal, reference); u /= np.linalg.norm(u)
            v = np.cross(normal, u)
            theta = np.linspace(0, 2*np.pi, 65)
            ring = (center+radius**2/distance*normal + radius*np.sqrt(1-(radius/distance)**2)*
                    (np.cos(theta)[:, None]*u+np.sin(theta)[:, None]*v))
            for a, b in zip(ring[:-1], ring[1:]):
                drawing.line(a, b, (255, 0, 255))
        drawing.line(center+[0, 0, .03], center, (255, 255, 0), arrow=True)
        drawing.marker(center, (255, 0, 255))
        drawing.marker(point, (0, 255, 255))
    cv2.rectangle(drawing.image, (0, 0), (color.shape[1], 45), (20, 20, 20), -1)
    for y, line in enumerate([('Solid: visible | dotted: hidden | dim dots: unknown' if occlusion else
                              'Flat projection: outlines do not certify visibility or clearance'),
                              'Ball: magenta; descent: cyan; pivot tangent: orange; pull: yellow; -X: gray']):
        cv2.putText(drawing.image, line, (6, 16+20*y), cv2.FONT_HERSHEY_SIMPLEX,
                    .4, (255, 255, 255), 1)
    return drawing.image
