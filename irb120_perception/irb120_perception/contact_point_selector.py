"""Standalone geometric contact selection; no robot motion or legacy imports.

Inputs are segmented observations of ONE stationary object, in metres in a
common Z-up frame. Use select_contact_points() from Python, or run this module
with .npy clouds / --topic. See ../CONTACT_SELECTION.md for assumptions.
"""

import argparse
import json
import time

import numpy as np
from scipy.spatial import ConvexHull, QhullError, cKDTree

DEFAULT_TOPIC = '/object_detector/object_points'
FRAME = 'base_link'
TIMEOUT = 5.0
RELAXED_PULL_EPSILON = 0.5
TOPMOST_BAND = 0.02  # press fallback: use points this close to the highest point


def estimate_normals(points, radius=0.025, neighbors=24):
    """Local PCA normals, oriented away from the cloud median (convex assumption).

    Sparse, line-like and strongly nonplanar neighborhoods get NaN normals and
    cannot be selected. The support footprint still uses their positions.
    """
    normals = np.full_like(points, np.nan)
    tree = cKDTree(points)
    distances, indices = tree.query(points, k=min(neighbors, len(points)))
    center = np.median(points, axis=0)
    for i, (distance, index) in enumerate(zip(distances, indices)):
        local = points[index[distance <= radius]]
        if len(local) < 6:
            continue
        local = local - local.mean(axis=0)
        values, vectors = np.linalg.eigh(local.T @ local)
        if values[1] <= 1e-12 or values[0] / values.sum() > 0.12:
            continue
        normal = vectors[:, 0]
        outward = normal @ (points[i] - center)
        if abs(outward) < 1e-8:
            continue
        normals[i] = normal if outward > 0 else -normal
    return normals


def estimate_pivot(points, preferred_direction, table_z, support_band=0.02,
                   min_edge_length=0.03):
    """Estimate the support edge facing a nominal horizontal direction.

    Use only observed points within support_band of the known table. Long
    hull edges within 45 degrees of the preference are extended-edge pivots;
    otherwise use the extreme vertex and the tangent normal to the preference.
    A missing/degenerate support footprint is an error, not a whole-cloud hull
    fallback (which could mistake an overhang for the base).
    """
    direction = np.asarray(preferred_direction, dtype=float)
    if (direction.shape != (3,) or not np.isfinite(direction).all()
            or abs(direction[2]) > 1e-9 or np.linalg.norm(direction) < 1e-9):
        raise ValueError('preferred_direction must be a nonzero horizontal vector')
    direction = direction / np.linalg.norm(direction)
    base = points[(points[:, 2] >= table_z - 0.003)
                  & (points[:, 2] <= table_z + support_band), :2]
    try:
        hull = ConvexHull(base)
    except (ValueError, QhullError) as exc:
        raise ValueError('Not enough observed support points near the table') from exc
    vertices = base[hull.vertices]
    ends = np.roll(vertices, -1, axis=0)
    edges = ends - vertices
    lengths = np.linalg.norm(edges, axis=1)
    outward = np.column_stack((edges[:, 1], -edges[:, 0])) / lengths[:, None]
    alignment = outward @ direction[:2]
    # Tiny corner facets occur even on boxes when the cameras miss the exact
    # corners. Prefer a substantial facing edge rather than such a facet.
    facing = np.flatnonzero((lengths >= min_edge_length)
                            & (alignment >= np.cos(np.pi / 4)))
    if len(facing):
        i = facing[np.argmax(alignment[facing])]
        pivot = np.r_[0.5 * (vertices[i] + ends[i]), table_z]
        direction = np.r_[outward[i], 0.0]
        kind = 'extended_edge'
        edge = np.column_stack((np.vstack((vertices[i], ends[i])), [table_z] * 2))
    else:
        pivot = np.r_[vertices[np.argmax(vertices @ direction[:2])], table_z]
        kind = 'isolated_point'
        edge = None
    return dict(pivot=pivot, axis=np.cross([0., 0., 1.], direction),
                direction=direction, kind=kind, edge=edge)


def _press_band(points, ids, scores, geometry, press_pivot_weight, score_tolerance):
    """Boolean mask into `ids`: the double-scored press candidate band.

    Double-scoring moment-arm heuristic, press mode only (forward_tip keeps
    a single height-only score). Combines two moment arms, both in metres:

      1. Tipping torque from the subsequent pull, proportional to contact
         height above the pivot (`scores`) -- maximize.
      2. Anti-tipping resistance, proportional to how far "inboard" of the
         pivot edge the contact sits along the estimated pull direction --
         the object's own mass between the pivot and the contact resists
         rotating over that edge, so this should be minimized (i.e.
         proximity to the edge maximized).

    Weighing both together, rather than treating height as a hard cutoff
    and pivot-proximity as a subordinate tie-break, keeps the choice robust
    to top-surface noise/tilt: a single noisy high point elsewhere on the
    top can otherwise exclude the true near-edge points from the candidate
    band entirely once they fall more than score_tolerance below it, even
    though they are the physically better contact.
    """
    pivot_proximity = np.cross(
        points[ids] - geometry['pivot'], [0., 0., -1.]) @ geometry['axis']
    combined = scores[ids] + press_pivot_weight * pivot_proximity
    return combined >= combined.max() - score_tolerance


def select_contact_points(clouds, *, normals=None,
                          table_z=-0.021,
                          finger_radius=0.01325, 
                          table_buffer=0.003,
                          min_ball_center_z=None, 
                          y_band=0.005,
                          support_band=0.02, 
                          min_edge_length=0.03,
                          press_inset=0.0, # 0.005 was reasonable but failed for flashlight.
                          score_tolerance=0.003,
                          pull_normal_epsilon=0.1,
                          normal_radius=0.025,
                          press_pivot_weight=1.0):
    """Return three independent selections and their estimated pivot geometry.

    clouds: an (N,3) array or list of arrays already registered in one frame.
    normals: optional outward (N,3) normals in the concatenated input order.
    The contact point is always a sensed point. ball_center = point + radius*n.
    Missing candidates return {available: False, reason: ...} for that mode.
    min_ball_center_z can impose a previously collision-checked tool-height
    floor; this function itself checks only the fingertip against the table.

    press_pivot_weight: weight (metres of pull-moment-arm per metre of
    horizontal pivot-distance) applied to the press mode's double-scoring —
    see the 'Press' comment below. Unused by forward_tip/planar_push.
    """
    scalars = [table_z, finger_radius, table_buffer, y_band, support_band,
               min_edge_length, press_inset, score_tolerance,
               pull_normal_epsilon, normal_radius, press_pivot_weight]
    if not np.isfinite(scalars).all():
        raise ValueError('All parameters must be finite')
    if (finger_radius <= 0 or normal_radius <= 0 or support_band <= 0
            or min_edge_length <= 0 or min(table_buffer, y_band, press_inset,
                                          score_tolerance) < 0
            or not 0 <= pull_normal_epsilon <= 1 or press_pivot_weight < 0):
        raise ValueError('Invalid radius, band, tolerance or normal threshold')
    if min_ball_center_z is not None and not np.isfinite(min_ball_center_z):
        raise ValueError('min_ball_center_z must be finite')
    if isinstance(clouds, np.ndarray) and clouds.ndim == 2:
        points = np.asarray(clouds, dtype=float)
    else:
        points = np.concatenate([np.asarray(c, dtype=float) for c in clouds])
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('Expected (N,3) object cloud(s)')
    finite = np.isfinite(points).all(axis=1)
    supplied = None
    if normals is not None:
        supplied = np.asarray(normals, dtype=float)
        if supplied.shape != points.shape:
            raise ValueError('Normals must match concatenated points')
        supplied = supplied[finite]
    points = points[finite]
    # Deduplicate camera overlap; lexicographic ordering also stabilizes ties.
    points, unique = np.unique(points, axis=0, return_index=True)
    if len(points) < 6:
        raise ValueError('Need at least six distinct finite object points')
    if supplied is None:
        normals = estimate_normals(points, radius=normal_radius)
    else:
        normals = supplied[unique]
        lengths = np.linalg.norm(normals, axis=1)
        valid = np.isfinite(normals).all(axis=1) & (lengths > 1e-9)
        normals[valid] /= lengths[valid, None]
        normals[~valid] = np.nan

    centers = points + finger_radius * normals
    floor = table_z + finger_radius + table_buffer
    if min_ball_center_z is not None:
        floor = max(floor, min_ball_center_z)
    clear = np.isfinite(normals).all(axis=1) & (centers[:, 2] >= floor)

    def missing(reason):
        return dict(available=False, reason=reason)

    def contact(i, score):
        return dict(available=True, point=points[i].copy(), normal=normals[i].copy(),
                    ball_center=centers[i].copy(), score=float(score))

    result = dict(point_count=len(points), normal_count=int(np.isfinite(normals).all(axis=1).sum()))
    # Planar sliding stays along +X and close to y=0. Restrict to an actually
    # inward-facing side; a tangential top normal is not a useful push contact.
    ids = np.flatnonzero(clear & (normals[:, 0] <= -0.2)
                         & (np.abs(normals[:, 2]) <= 0.5)
                         & (np.abs(points[:, 1]) <= y_band))
    if len(ids):
        i = ids[np.lexsort((points[ids, 0], np.abs(points[ids, 1]), points[ids, 2]))[0]]
        result['planar_push'] = contact(i, -(points[i, 2] - table_z))
        result['planar_push']['direction'] = np.array([1., 0., 0.])
    else:
        result['planar_push'] = missing('No inward side contact near y=0 with fingertip clearance')

    def inset_top(ids):
        try:
            eq = ConvexHull(points[ids, :2]).equations
        except (QhullError, ValueError):
            return np.array([], dtype=int)
        distances = -(points[ids, :2] @ eq[:, :2].T + eq[:, 2])
        return ids[np.min(distances, axis=1) >= press_inset - 1e-10]

    for mode, preferred in [('forward_tip', [1., 0., 0.]), ('press', [-1., 0., 0.])]:
        try:
            geometry = estimate_pivot(points, preferred, table_z, support_band, min_edge_length)
        except ValueError as exc:
            result[mode] = missing(str(exc))
            continue
        direction = geometry['direction']
        # Press: if nothing passes, retry once with a looser pull tolerance so a
        # rounded top (normals tilted toward the pull direction) still yields a contact.
        epsilons = [0.0] if mode == 'forward_tip' else [pull_normal_epsilon, max(pull_normal_epsilon, RELAXED_PULL_EPSILON)]
        for epsilon in epsilons:
            counts = {'clearance': int(clear.sum())}
            feasible = clear & (normals @ direction <= epsilon)
            counts['direction'] = int(feasible.sum())
            if mode == 'forward_tip':
                feasible &= (normals @ direction <= -0.2) & (np.abs(normals[:, 2]) <= 0.5)
            else:
                feasible &= normals[:, 2] >= 0.7
            counts['surface'] = int(feasible.sum())
            scores = np.cross(points - geometry['pivot'], direction) @ geometry['axis']
            feasible &= scores > 0
            ids = np.flatnonzero(feasible)
            counts['positive_moment'] = len(ids)
            # Inset the full feasible top region BEFORE selecting the best moment
            # band. A narrow high strip on a sloping/noisy top is not the face edge.
            if len(ids) and mode == 'press' and press_inset > 0:
                ids = inset_top(ids)
            counts['inset'] = len(ids)
            if len(ids):
                ids = ids[_press_band(points, ids, scores, geometry, press_pivot_weight, score_tolerance)
                          if mode == 'press' else scores[ids] >= scores[ids].max() - score_tolerance]
            counts['best_score_band'] = len(ids)
            if len(ids):
                break
        # Last resort for tiny rounded tops where no neighborhood is flat enough for a
        # usable normal: press on the highest points and assume a horizontal surface.
        topmost = False
        if not len(ids) and mode == 'press':
            band = points[:, 2] >= points[:, 2].max() - TOPMOST_BAND
            # Only with evidence of a cap (some known upward-leaning normal up there); NaN
            # normals among them are then tolerated, clearly vertical ones (a side wall) are not.
            cap = band & (normals[:, 2] >= 0.3)
            ids = np.flatnonzero(band & ~(normals[:, 2] < 0.3) & (scores > 0) & (points[:, 2] + finger_radius >= floor)) \
                if cap.any() else np.array([], dtype=int)
            if len(ids) and press_inset > 0:
                ids = inset_top(ids)
            counts['topmost_fallback'] = len(ids)
            if len(ids):
                ids = ids[_press_band(points, ids, scores, geometry, press_pivot_weight, score_tolerance)]
                topmost = True
        if not len(ids):
            failed = next(stage for stage, count in counts.items() if count == 0)
            result[mode] = missing(f'No candidates after {failed} filter')
        else:
            # `best_score_band` above already double-scores height and pivot
            # proximity together for press. What remains here is centering
            # along the pivot edge itself (an axis orthogonal to both terms).
            if mode == 'press' and geometry['edge'] is not None:
                edge_midpoint = geometry['edge'].mean(axis=0)
                along_edge = (points[ids] - edge_midpoint) @ geometry['axis']
                # The desired-axis moments do not distinguish locations
                # along a straight pivot edge. Center there, in the edge's
                # own coordinate system, rather than in a world axis.
                ids = ids[np.abs(along_edge) <= np.abs(along_edge).min() + 1e-10]
                counts['centered_on_edge'] = len(ids)
                # If several points are equally centered, keep the most
                # edgeward one; the median below then only resolves truly
                # equivalent samples instead of drifting inward.
                centered_press_scores = np.cross(
                    points[ids] - geometry['pivot'], [0., 0., -1.]) @ geometry['axis']
                ids = ids[centered_press_scores >= centered_press_scores.max() - 1e-10]
                counts['centered_edgeward'] = len(ids)
            target = np.median(points[ids], axis=0)
            if mode == 'forward_tip':
                target[1] = 0.0
            i = ids[np.argmin(np.linalg.norm(points[ids] - target, axis=1))]
            result[mode] = contact(i, scores[i])
            if mode == 'press':
                result[mode]['press_score'] = float(
                    geometry['axis'] @ np.cross(
                        points[i] - geometry['pivot'], [0., 0., -1.]))
                if geometry['edge'] is not None:
                    edge_midpoint = geometry['edge'].mean(axis=0)
                    result[mode]['edge_midpoint'] = edge_midpoint
                    result[mode]['along_edge_offset'] = float(
                        (points[i] - edge_midpoint) @ geometry['axis'])
            if topmost:
                result[mode]['normal'] = np.array([0., 0., 1.])
                result[mode]['ball_center'] = points[i] + finger_radius * result[mode]['normal']
                result[mode]['fallback'] = 'topmost_points_assumed_horizontal'
        result[mode]['candidate_counts'] = counts
        if mode == 'press' and epsilon != pull_normal_epsilon:
            result[mode]['relaxed_pull_epsilon'] = epsilon
        result[mode]['geometry'] = geometry
    return result


def read_topic(topic, frame, object_id, timeout):
    """Read one labeled object cloud; ROS imports stay out of the pure API."""
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import PointCloud2
    from sensor_msgs_py import point_cloud2

    rclpy.init(args=[])
    node = rclpy.create_node('contact_point_selector')
    received = []
    node.create_subscription(PointCloud2, topic, received.append, qos_profile_sensor_data)
    try:
        deadline = time.monotonic() + timeout
        while not received and rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        if not received:
            raise ValueError(f'No cloud on {topic} within {timeout}s; enable perception first')
        msg = received[-1]
        if msg.header.frame_id != frame:
            raise ValueError(f'Expected frame {frame}, received {msg.header.frame_id}; transform first')
        labeled = 'label' in {field.name for field in msg.fields}
        fields = ('x', 'y', 'z', 'label') if labeled else ('x', 'y', 'z')
        data = point_cloud2.read_points(msg, field_names=fields, skip_nans=True).reshape(-1)
        if labeled:
            labels = np.unique(data['label'])
            if object_id is None:
                if len(labels) != 1:
                    raise ValueError('Cloud must contain one object; specify --object-id')
                object_id = int(labels[0])
            data = data[data['label'] == object_id]
        elif object_id is not None:
            raise ValueError('--object-id requires a labeled cloud')
        return np.column_stack([data[field] for field in ('x', 'y', 'z')])
    finally:
        node.destroy_node()
        rclpy.shutdown()


def build_contact_markers(result, frame, stamp):
    """Colored contact spheres, labels, motion arrows and signed pivot axes."""
    from geometry_msgs.msg import Point
    from visualization_msgs.msg import Marker, MarkerArray

    markers = MarkerArray()
    for mode, color in [('planar_push', (0.1, 0.9, 0.2)),
                        ('forward_tip', (1.0, 0.55, 0.05)),
                        ('press', (1.0, 0.1, 0.85))]:
        selected = result[mode]

        def marker(marker_id, kind):
            item = Marker()
            item.header.frame_id = frame
            item.header.stamp = stamp
            item.ns = mode
            item.id = marker_id
            item.type = kind
            item.action = Marker.ADD
            item.pose.orientation.w = 1.0
            item.color.r, item.color.g, item.color.b = color
            item.color.a = 0.95
            # Expire shortly after the publisher stops; do not leave stale
            # contact targets in RViz after Ctrl-C.
            item.lifetime.sec = 3
            markers.markers.append(item)
            return item

        def point(xyz):
            return Point(x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2]))

        geometry = selected.get('geometry')
        if geometry:
            # Display-only lift so the table does not obscure the axis. The
            # computed pivot and geometry in JSON remain at table height.
            pivot = geometry['pivot'] + np.array([0., 0., 0.008])
            axis = marker(3, Marker.ARROW)
            axis.points = [point(pivot - 0.04 * geometry['axis']),
                           point(pivot + 0.04 * geometry['axis'])]
            axis.scale.x, axis.scale.y, axis.scale.z = 0.004, 0.009, 0.012
            axis_label = marker(4, Marker.TEXT_VIEW_FACING)
            axis_label.pose.position = point(pivot + np.array([0., 0., 0.015]))
            axis_label.scale.z = 0.012
            axis_label.text = mode.replace('_', ' ') + ' axis'
            if not selected['available']:
                axis_label.text += ' (no contact)'
            if geometry['edge'] is not None:
                # The support edge itself is distinct from the signed rotation
                # axis arrow.  Publish it so camera overlays can show exactly
                # which observed hull feature supplied the pivot hypothesis.
                support_edge = marker(5, Marker.LINE_LIST)
                support_edge.points = [point(geometry['edge'][0]), point(geometry['edge'][1])]
                support_edge.scale.x = 0.004
        if not selected['available']:
            continue

        position = selected['point']
        sphere = marker(0, Marker.SPHERE)
        sphere.pose.position = point(position)
        sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.015
        label = marker(1, Marker.TEXT_VIEW_FACING)
        label.pose.position = point(position + np.array([0., 0., 0.025]))
        label.scale.z = 0.015
        label.text = mode.replace('_', ' ')

        geometry = selected.get('geometry')
        direction = geometry['direction'] if geometry else selected['direction']
        arrow = marker(2, Marker.ARROW)
        arrow.points = [point(position), point(position + 0.06 * direction)]
        arrow.scale.x, arrow.scale.y, arrow.scale.z = 0.003, 0.007, 0.01
    return markers


def visualize_result(result, frame):
    """Keep one selection visible until Ctrl-C; does not resample the cloud."""
    import rclpy
    from rclpy.qos import DurabilityPolicy, QoSProfile
    from visualization_msgs.msg import MarkerArray

    rclpy.init(args=[])
    node = rclpy.create_node('contact_point_visualizer')
    publisher = node.create_publisher(
        MarkerArray, '/contact_point_selector/markers',
        QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))

    def publish():
        publisher.publish(build_contact_markers(result, frame, node.get_clock().now().to_msg()))

    node.create_timer(1.0, publish)
    try:
        publish()
        node.get_logger().info(
            'Showing selection snapshot on /contact_point_selector/markers. '
            'Green: planar push; orange: forward tip; magenta: press. '
            'Ctrl-C to stop; rerun after the object moves.')
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('clouds', nargs='*',
                        help='Nx3 .npy file(s) in base_link; omit to read one message from '
                             f'{DEFAULT_TOPIC}')
    parser.add_argument('--object-id', type=int, help='Label to select if the cloud has several objects')
    options = parser.parse_args(args)
    try:
        clouds = ([np.load(path, allow_pickle=False) for path in options.clouds] if options.clouds else
                  read_topic(DEFAULT_TOPIC, FRAME, options.object_id, TIMEOUT))
        result = select_contact_points(clouds)
    except (ValueError, OSError) as exc:
        parser.exit(2, f'{exc}\n')
    result['frame'] = FRAME
    print(json.dumps(result, indent=2, default=lambda value: value.tolist(), allow_nan=False))
    visualize_result(result, FRAME)


if __name__ == '__main__':
    main()
