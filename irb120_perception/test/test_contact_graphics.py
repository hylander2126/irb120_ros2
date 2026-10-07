import numpy as np

from irb120_perception.contact_graphics import DepthDrawing, contact_graphic


def fixture():
    return (np.zeros((100, 100, 3), dtype=np.uint8),
            np.full((100, 100), 500, dtype=np.uint16),
            dict(K=[[100, 0, 50], [0, 100, 50], [0, 0, 1]], D=[0]*5,
                 base_to_optical=dict(R=np.eye(3).tolist(), t=[0, 0, 0])))


def test_depth_visibility_tolerance_missing_and_behind_camera():
    color, depth, info = fixture()
    d = DepthDrawing(color, depth, info)
    _, states = d.visibility([[0, 0, .502], [0, 0, .504], [0, 0, -.5]])
    assert list(states) == ['visible', 'hidden', 'outside']
    depth[50, 50] = 0
    assert d.visibility([[0, 0, .6]])[1][0] == 'unknown'
    assert DepthDrawing(color, None, info).visibility([[0, 0, .5]])[1][0] == 'unknown'


def test_occluded_line_has_gaps_and_visible_line_does_not():
    color, depth, info = fixture()
    visible = DepthDrawing(color, depth, info)
    hidden = DepthDrawing(color, depth, info)
    visible.line(np.array([-.15, 0, .5]), np.array([.15, 0, .5]), (0, 255, 0))
    hidden.line(np.array([-.18, 0, .6]), np.array([.18, 0, .6]), (0, 255, 0))
    assert np.count_nonzero(hidden.image[50, 20:81, 1]) < np.count_nonzero(visible.image[50, 20:81, 1])
    assert not color.any()


def test_graphic_handles_unavailable_contact_without_modifying_color():
    color, depth, info = fixture()
    img = contact_graphic(color, depth, info, {'available': False})
    assert img.shape == color.shape
    assert not color.any()
