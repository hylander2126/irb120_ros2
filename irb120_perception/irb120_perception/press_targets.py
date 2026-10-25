"""Saved press geometry shared by camera preview, RViz, and arc approach."""
import numpy as np

PRESS_STANDOFF = .03
APPROACH_BOX_PADDING = .010  # irb120_control push_geometry.OBJECT_PADDING (the approach's object box)
FINGER_BALL_RADIUS = .01325


def pre_press_center(ball):
    return np.asarray(ball, dtype=float) + [0., 0., PRESS_STANDOFF]


def press_approachable(ball, object_top_z):
    """The press approach plans against the object's padded bounding box: a pre-press ball
    below its top puts the finger shaft inside it at any pitch (toy_robot 20261025_14xxxx)."""
    return bool(pre_press_center(ball)[2] - FINGER_BALL_RADIUS >= object_top_z + APPROACH_BOX_PADDING)


def press_targets(press, rotation=None, translation=None):
    """Return targets in the requested frame; approach is along that frame's +Z."""
    if not press.get('available'):
        raise ValueError('No available saved press contact')
    rotation = np.eye(3) if rotation is None else np.asarray(rotation, dtype=float)
    translation = np.zeros(3) if translation is None else np.asarray(translation, dtype=float)
    result = {}
    for key, value in [('contact', press['point']), ('ball', press['ball_center']),
                       ('pivot', press['geometry']['pivot'])]:
        value = np.asarray(value, dtype=float)
        if value.shape != (3,) or not np.isfinite(value).all():
            raise ValueError('Invalid saved press '+key)
        result[key] = rotation @ value + translation
    result['pre_press'] = pre_press_center(result['ball'])
    result['axis'] = rotation @ np.asarray(press['geometry']['axis'], dtype=float)
    result['radius'] = float(np.linalg.norm(result['ball']-result['contact']))
    return result


def build_press_markers(targets, frame, stamp, label='Saved press'):
    from geometry_msgs.msg import Point
    from visualization_msgs.msg import Marker, MarkerArray
    result = MarkerArray()
    def xyz(value):
        return Point(x=float(value[0]), y=float(value[1]), z=float(value[2]))
    def marker(kind, color):
        item = Marker()
        item.header.frame_id = frame
        item.header.stamp = stamp
        item.ns = 'saved_press_targets'
        item.id = len(result.markers)
        item.type = kind
        item.action = Marker.ADD
        item.pose.orientation.w = 1.
        item.color.r, item.color.g, item.color.b = color
        item.color.a = .8
        item.lifetime.sec = 3
        result.markers.append(item)
        return item
    for name, color, diameter in [('contact', (1., 1., 0.), .006),
                                  ('ball', (1., 0., 1.), 2*targets['radius']),
                                  ('pre_press', (0., 1., 1.), 2*targets['radius'])]:
        item = marker(Marker.SPHERE, color)
        item.pose.position = xyz(targets[name])
        item.scale.x = item.scale.y = item.scale.z = diameter
    arrow = marker(Marker.ARROW, (0., 1., 1.))
    arrow.points = [xyz(targets['pre_press']), xyz(targets['ball'])]
    arrow.scale.x, arrow.scale.y, arrow.scale.z = .002, .006, .008
    pivot = marker(Marker.LINE_LIST, (1., .65, 0.))
    pivot.points = [xyz(targets['pivot']-.04*targets['axis']),
                    xyz(targets['pivot']+.04*targets['axis'])]
    pivot.scale.x = .003
    text = marker(Marker.TEXT_VIEW_FACING, (1., 1., 1.))
    text.pose.position = xyz(targets['pre_press']+[0, 0, .025])
    text.scale.z = .012
    text.text = label+'\nCyan: pre-press | magenta: contact ball | yellow: contact'
    return result
