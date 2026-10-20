"""Variant references must preserve unknowns and independent geometry inputs."""
import pytest
import yaml

from irb120_models import configuration_mesh, list_configurations, load_configuration
from irb120_models import configurations
from irb120_models.configurations import com_from_near_edge, planar_com, CENTER_FRAME
from irb120_models.experiment_gt import cylinder_sweep_references, forward_tip_ground_truth


def test_existing_tipt_references_and_shared_mesh():
    gt = forward_tip_ground_truth()
    record = load_configuration('tipT_d')
    assert gt['tipT_d']['mass'] == record['mass']
    assert gt['tipT_d']['com_obj'] == planar_com(record)
    assert gt['tipT_d']['com'][0] == pytest.approx(record['base_w']-record['com_obj'][0])
    assert 'tipT_c_2slugs' not in list_configurations('tipT')
    assert 'tipT_c_2slugs' in list_configurations('tipT', include_temporary=True)
    assert configuration_mesh('tipT_a') == configuration_mesh('tipT_e')


def test_unknown_values_do_not_inherit_legacy_placeholders(tmp_path, monkeypatch):
    import irb120_models.experiment_gt as adapter
    fixture_family(tmp_path, monkeypatch, dict(fillet_radius=.008,mesh=None))
    monkeypatch.setattr(adapter, 'list_configurations', lambda family: ['sample_a'])
    record = load_configuration('sample_a')
    assert record['mass'] is None
    assert record['com_obj'] == [None, None]
    assert configuration_mesh('sample_a') is None
    assert adapter.cylinder_sweep_references()['sample_a'] == {'radius_mm': 8.}


def fixture_family(tmp_path, monkeypatch, variant):
    family = tmp_path/'sample'
    family.mkdir()
    data = dict(defaults=dict(com_obj=[None, None], mass=None),
                configurations=dict(sample_a=variant))
    path = family/'configurations.yaml'
    path.write_text(yaml.safe_dump(data))
    monkeypatch.setattr(configurations, 'objects_dir', lambda: tmp_path)
    return path


@pytest.mark.parametrize('variant', [dict(mass=-.1), dict(com_obj=[.01]),
                                    dict(com_obj=[.08,.1], base_w=.06),
                                    dict(uncertainty=dict(mass=-.01)),
                                    dict(mesh='meshes/missing.gltf')])
def test_invalid_measurement_or_missing_mesh_is_rejected(tmp_path, monkeypatch, variant):
    fixture_family(tmp_path, monkeypatch, variant)
    with pytest.raises(ValueError):
        load_configuration('sample_a')


def test_partial_values_and_uncertainty_override(tmp_path, monkeypatch):
    fixture_family(tmp_path, monkeypatch, dict(mass=.364, com_obj=[None,.08],
                                              uncertainty=dict(mass=.001)))
    record = load_configuration('sample_a')
    assert record['mass'] == .364
    assert record['com_obj'] == [None, .08]
    assert record['uncertainty']['mass'] == .001


def test_rounded_support_is_not_assumed_from_outer_width(tmp_path, monkeypatch):
    import irb120_models.experiment_gt as adapter
    record = dict(fillet_radius=.024, mass=.4, com_obj=[.03,.08], base_w=.06)
    fixture_family(tmp_path, monkeypatch, record)
    monkeypatch.setattr(adapter, 'list_configurations', lambda family: ['sample_a'])
    assert adapter.cylinder_sweep_references()['sample_a'] == dict(
        radius_mm=24.,mass_kg=.4,height_mm=80.)


@pytest.mark.parametrize('radius, expected_d', [(0., 28.), (.008, None)])
def test_center_frame_translation_and_sharp_support(tmp_path, monkeypatch, radius, expected_d):
    import irb120_models.experiment_gt as adapter
    fixture_family(tmp_path, monkeypatch, dict(frame=CENTER_FRAME, fillet_radius=radius,
                   com_obj=[-.002, .08], base_w=.06))
    record = load_configuration('sample_a')
    assert record['com_obj'] == [-.002, .08]  # authored coordinates stay intact
    assert com_from_near_edge(record) == pytest.approx([.028,.08])
    monkeypatch.setattr(adapter, 'list_configurations', lambda family: ['sample_a'])
    ref = adapter.cylinder_sweep_references()['sample_a']
    if expected_d is None:
        assert 'd_mm' not in ref  # translated silhouette is not a curved-base pivot
    else:
        assert ref['d_mm'] == pytest.approx(expected_d)


def test_center_frame_unknown_width_does_not_become_zero_support(tmp_path, monkeypatch):
    import irb120_models.experiment_gt as adapter
    fixture_family(tmp_path, monkeypatch, dict(frame=CENTER_FRAME, fillet_radius=0.,
                                              com_obj=[0., .08]))
    monkeypatch.setattr(adapter, 'list_configurations', lambda family: ['sample_a'])
    assert 'd_mm' not in adapter.cylinder_sweep_references()['sample_a']


@pytest.mark.parametrize('variant', [dict(frame='typo'),
                                    dict(frame=CENTER_FRAME, com_obj=[-.04,.08], base_w=.06)])
def test_invalid_frame_or_centered_coordinate_is_rejected(tmp_path, monkeypatch, variant):
    fixture_family(tmp_path, monkeypatch, variant)
    with pytest.raises(ValueError):
        load_configuration('sample_a')


@pytest.mark.parametrize('radius, expected_mm', [(0.,40.),(.008,32.),(.016,24.),(.024,16.),(.030,10.)])
def test_fillet_support_uses_shared_outer_diameter(tmp_path, monkeypatch, radius, expected_mm):
    import irb120_models.experiment_gt as adapter
    fixture_family(tmp_path, monkeypatch, dict(frame=CENTER_FRAME, support_model='circular_base_fillet',
                   fillet_radius=radius, base_w=.08, com_obj=[0.,None]))
    record = load_configuration('sample_a')
    assert record['initial_support_distances'] == pytest.approx(
        dict(forward=expected_mm/1000,press_pull=expected_mm/1000))
    monkeypatch.setattr(adapter, 'list_configurations', lambda family: ['sample_a'])
    assert adapter.cylinder_sweep_references()['sample_a']['d_mm'] == pytest.approx(expected_mm)


def test_fillet_support_preserves_offset_and_explicit_override(tmp_path, monkeypatch):
    fixture_family(tmp_path, monkeypatch, dict(frame=CENTER_FRAME, support_model='circular_base_fillet',
                   fillet_radius=.008,base_w=.08,com_obj=[-.002,.1],
                   support_distances=dict(press_pull=.029)))
    distances=load_configuration('sample_a')['initial_support_distances']
    assert distances == pytest.approx(dict(forward=.034,press_pull=.029))


def test_full_radius_fillet_is_zero_support_without_clamping_to_positive(tmp_path, monkeypatch):
    fixture_family(tmp_path, monkeypatch, dict(frame=CENTER_FRAME,support_model='circular_base_fillet',
                   fillet_radius=.04,base_w=.08,com_obj=[0.,.1]))
    assert load_configuration('sample_a')['initial_support_distances']['press_pull'] == 0.


def test_fillet_cannot_exceed_outer_radius(tmp_path, monkeypatch):
    fixture_family(tmp_path, monkeypatch, dict(support_model='circular_base_fillet',
                                              fillet_radius=.041,base_w=.08))
    with pytest.raises(ValueError,match='exceeds outer radius'):
        load_configuration('sample_a')


def test_all_cylinder_variants_have_derived_initial_support():
    for name in list_configurations('cylinder'):
        record=load_configuration(name)
        expected=record['base_w']/2-record['fillet_radius']
        assert record['initial_support_distances']['press_pull'] == pytest.approx(
            expected+record['com_obj'][0])


def test_xyz_com_preserved_and_height_uses_z(tmp_path, monkeypatch):
    import irb120_models.experiment_gt as adapter
    fixture_family(tmp_path, monkeypatch, dict(frame=CENTER_FRAME,
                   support_model='circular_base_fillet',base_w=.08,fillet_radius=.008,
                   mass=.362,com_obj=[0.,0.,.072]))
    record=load_configuration('sample_a')
    assert record['com_obj'] == [0.,0.,.072]
    assert planar_com(record) == [0.,.072]
    monkeypatch.setattr(adapter,'list_configurations',lambda family:['sample_a'])
    assert adapter.cylinder_sweep_references()['sample_a'] == pytest.approx(
        dict(radius_mm=8.,mass_kg=.362,height_mm=72.,d_mm=32.))


def test_everyday_family_record_with_table_edge_distances(tmp_path, monkeypatch):
    import yaml
    from irb120_models import configurations
    family = tmp_path / 'everyday'
    family.mkdir()
    source = (configurations.objects_dir() / 'everyday' / 'configurations.yaml').read_text()
    data = yaml.safe_load(source)
    assert data['configurations'] == {}          # shipped empty; Steven fills it in
    data['configurations'] = {'mug_test': {
        'placement': 'handle left', 'com_obj': [0.041, 0.002, None], 'base_w': 0.082,
        'support_distances': {'press_pull': 0.041, 'forward': 0.0415},
        'uncertainty': {'support_distances': {'press_pull': 0.001, 'forward': 0.001}}}}
    (family / 'configurations.yaml').write_text(yaml.safe_dump(data))
    monkeypatch.setattr(configurations, 'objects_dir', lambda: tmp_path)
    record = configurations.load_configuration('mug_test')
    assert record['mass'] is None and record['com_obj'][2] is None
    assert record['initial_support_distances'] == {'press_pull': 0.041, 'forward': 0.0415}
    assert abs(configurations.support_closure(record) - 0.0005) < 1e-12
    assert 'mug_test' in configurations.list_configurations('everyday')
