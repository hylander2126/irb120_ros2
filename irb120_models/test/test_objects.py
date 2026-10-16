import math
from xml.etree import ElementTree

from irb120_models import cli, mesh_tools
from irb120_models.objects import list_objects, load_object, objects_dir, parse_object
import numpy as np
import pytest

# What irb120_control/object_params.json held when it was replaced (2026-10-09):
# (mass_gt, com_gt, CAD theta*). The measured mass and CoM must not change; the CAD
# theta* is kept as information only (the cylinder's was a placeholder and was dropped).
LEGACY = {
    'box': (0.676, (0.05, 0.0, 0.15), 17.532),
    'heart': (0.239, (0.0458, 0.0, 0.10), 23.984),
    'flashlight': (0.387, (0.028, 0.0, 0.0938), 15.126),
    'monitor': (5.04, (0.06, 0.0, 0.232), 14.748),
    'soda': (2.054, (0.0525, 0.0, 0.15), 20.126),
    'cylinder': (0.364, (0.0, 0.0, 0.1), 15.0),
}


def test_every_object_parses():
    names = list_objects()
    assert set(LEGACY) <= set(names)
    for name in names:
        assert load_object(name).name == name


@pytest.mark.parametrize('name', sorted(LEGACY))
def test_legacy_ground_truth_is_preserved(name):
    mass, com, theta = LEGACY[name]
    gt = load_object(name).ground_truth('default')
    assert gt['mass'] == pytest.approx(mass)
    assert (gt['com_x'], gt['com_z']) == pytest.approx((com[0], com[2]))
    assert gt['theta_star_deg'] == pytest.approx(math.degrees(math.atan2(com[0], com[2])))
    if name != 'cylinder':
        assert gt['theta_star_cad_deg'] == pytest.approx(theta)


def test_generated_files_are_current():
    assert cli.cmd_check(list_objects()) == 0, 'run `object_models generate`'


@pytest.mark.parametrize('name', list_objects())
def test_urdf_is_valid_and_meshes_exist(name):
    model = load_object(name)
    if model.urdf is None:
        assert model.mesh is None
        return
    root = ElementTree.parse(model.urdf).getroot()
    assert root.tag == 'robot' and root.find('link').get('name') == name
    for mesh in root.iter('mesh'):
        rel = mesh.get('filename').removeprefix('package://irb120_models/objects/')
        assert (objects_dir() / rel).exists()


def test_unit_cube_mass_properties():
    corners = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], float)
    faces = [(0, 2, 3), (0, 3, 1), (4, 5, 7), (4, 7, 6), (0, 1, 5), (0, 5, 4),
             (2, 6, 7), (2, 7, 3), (0, 4, 6), (0, 6, 2), (1, 3, 7), (1, 7, 5)]
    faces = np.array(faces)[:, ::-1]  # listed clockwise from outside; flip to outward
    props = mesh_tools.mass_properties(corners[faces], mass=6.0)
    assert props['volume'] == pytest.approx(1.0)
    assert props['centroid'] == pytest.approx([0.5, 0.5, 0.5])
    assert props['inertia'] == pytest.approx(np.eye(3))  # m a^2 / 6 = 1


def test_gltf_to_stl_round_trip(tmp_path):
    tri = mesh_tools.load_triangles(load_object('cylinder').mesh)
    out = tmp_path / 'c.stl'
    mesh_tools.write_stl(out, tri)
    again = mesh_tools.load_triangles(out)
    assert again.shape == tri.shape
    assert np.abs(again - tri).max() < 1e-6
    lo, hi = mesh_tools.mass_properties(tri)['bounds']
    assert lo == pytest.approx([-0.03, -0.03, 0.0], abs=1e-4)
    assert hi == pytest.approx([0.03, 0.03, 0.1], abs=1e-4)


def test_ground_truth_from_body_frame_com(tmp_path):
    folder = tmp_path / 'probe'
    folder.mkdir()
    (folder / 'probe.yaml').write_text(
        'name: probe\nstatus: provisional\nbody_frame: tip_edge\n'
        'rest_pose: {xyz: [0, 0, 0], rpy: [0, 0, 1.5707963267948966]}\n'
        'mass: {value: 1.0}\ncom: {value: [0.04, 0.0, 0.1]}\n'
        'tip_edges: {default: {point: [0.0, 0.0, 0.0], direction: [1.0, 0.0, 0.0]}}\n')
    gt = parse_object(folder / 'probe.yaml').ground_truth()
    # Rotated 90 deg about z, the CoM sits at y = 0.04: 0.04 from an edge along x.
    assert gt['source'] == 'body_frame'
    assert (gt['com_x'], gt['com_z']) == pytest.approx((0.04, 0.1))
    assert gt['theta_star_deg'] == pytest.approx(math.degrees(math.atan2(0.04, 0.1)))


def test_complete_status_requires_measurements_but_not_cad(tmp_path):
    folder = tmp_path / 'probe'
    folder.mkdir()
    (folder / 'probe.yaml').write_text('name: probe\nstatus: complete\nmass: {value: 1.0}\n')
    with pytest.raises(ValueError, match='complete'):
        parse_object(folder / 'probe.yaml')
    (folder / 'probe.yaml').write_text(
        'name: probe\nstatus: complete\nbody_frame: tip_edge\nmesh: null\n'
        'mass: {value: 1.0}\ncom: {value: [0.04, 0.0, 0.1]}\n'
        'tip_edges: {default: {point: [0, 0, 0], direction: [0, 1, 0]}}\n')
    assert parse_object(folder / 'probe.yaml').status == 'complete'  # no mesh needed


def test_experiment_ground_truth_adapters():
    from irb120_models.experiment_gt import forward_tip_ground_truth, measured_ground_truth
    regular = forward_tip_ground_truth()
    all_configs = forward_tip_ground_truth(include_temporary=True)
    assert 'tipT_c_2slugs' not in regular
    assert all_configs['tipT_c_2slugs']['com'] == pytest.approx([.030, 0., .11912])
    assert all_configs['tipT_c_2slugs']['mass'] == pytest.approx(.365)
    assert regular['tipT_d']['com'][0] == pytest.approx(.0526)
    assert regular['tipT_e']['com'][0] == pytest.approx(.0074)
    for name, gt in measured_ground_truth(LEGACY).items():
        expected = load_object(name).ground_truth()
        assert gt['mass'] == expected['mass']
        assert gt['theta_deg'] == expected['theta_star_deg']
