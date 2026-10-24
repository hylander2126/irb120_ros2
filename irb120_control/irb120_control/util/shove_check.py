"""Planning-only candidate selection. No ROS imports or motion side effects."""
import math


def validate_request(request):
    """Require explicit contact, release and stopped-tool samples in base_link."""
    names = request['joint_names']
    joints = request['initial_joints']
    if (not names or len(set(names)) != len(names) or len(joints) != len(names)
            or not all(math.isfinite(x) for x in joints)):
        raise ValueError('Expected unique joint_names and finite initial_joints')
    if 'joint_6' not in names or abs(joints[names.index('joint_6')]) > math.pi / 3:
        raise ValueError('Initial joint_6 must be inside +/-60 degrees')
    if not request['candidates']:
        raise ValueError('At least one candidate is required')
    ids = set()
    for candidate in request['candidates']:
        if candidate['id'] in ids:
            raise ValueError('Candidate ids must be unique')
        ids.add(candidate['id'])
        samples = candidate['samples']
        times = [s['time_s'] for s in samples]
        release = candidate['release_index']
        if (len(samples) < 3 or times[0] != 0 or not all(math.isfinite(t) for t in times)
                or any(b <= a for a, b in zip(times, times[1:]))
                or not isinstance(release, int) or not 0 < release < len(samples)-1):
            raise ValueError('Samples must span contact t=0, release, and braking end')
        for sample in samples:
            p, q = sample['position'], sample['orientation_xyzw']
            if (len(p) != 3 or len(q) != 4 or not all(math.isfinite(x) for x in p+q)
                    or abs(sum(x*x for x in q)-1) > 1e-5):
                raise ValueError('Expected finite base_link position and unit xyzw quaternion')
    return request


def check_candidates(request, planner):
    """Try independent, physics-recomputed candidates in caller preference order.

    A failed geometric plan is not proof of workspace exclusion: collision,
    cable constraints, seed, or solver timeout can also cause failure.
    """
    validate_request(request)
    attempts = []
    selected = None
    for candidate in request['candidates']:
        result = dict(planner(request, candidate))
        result['candidate_id'] = candidate['id']
        attempts.append(result)
        if result.get('geometric_path_found') is True:
            selected = candidate['id']
            break
    return dict(selected_candidate=selected, attempts=attempts,
                timing_verified=False, execution_ready=False,
                request=request,
                limitations=['MoveIt timing is not the requested launch timing.',
                             'Uses the existing planning scene; moving-object sweep is not verified.',
                             'No approach, contact establishment, or hardware execution is performed.'])
