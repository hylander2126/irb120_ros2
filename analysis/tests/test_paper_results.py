"""GT-free helpers used by make_paper_results."""
import numpy as np
import pytest

import hull_com
import references


def test_hull_centroid_of_a_box_resting_on_the_table():
    rng = np.random.default_rng(0)
    # Visible surface of a 60 x 40 x 100 mm box, centred at (0.6, 0.1), bottom on the table.
    pts = rng.uniform([-.03, -.02, 0], [.03, .02, .1], (4000, 3))
    face = rng.integers(0, 3, 4000)
    pts[face == 0, 0] = np.sign(pts[face == 0, 0]) * .03
    pts[face == 1, 1] = np.sign(pts[face == 1, 1]) * .02
    pts[face == 2, 2] = .1
    pts += [.6, .1, hull_com.TABLE_Z_BASE_LINK]
    c, v = hull_com.hull_centroid(pts)
    assert c == pytest.approx([.6, .1, hull_com.TABLE_Z_BASE_LINK + .05], abs=1e-3)
    assert v == pytest.approx(.06 * .04 * .1, rel=.02)
    assert hull_com.horizontal_distance(c, [.57, .1, 0.], [0, -1, 0]) == pytest.approx(.03, abs=1e-3)


def test_unknown_objects_get_no_reference():
    assert references.lookup('everyday_mug_1') is None
    ref = references.lookup('tipT_a')
    assert ref['mass'] > 0 and references.estimator_reference(ref)['com'][0] > 0
    free = references.estimator_reference(None, .02)
    assert np.isnan(free['mass']) and free['com_obj'][0] == .02


def test_supplied_distance_is_the_outer_edge_not_the_support():
    cyl = references.lookup('cylinder_c')          # 80 mm diameter, 16 mm fillet
    assert cyl['d_env']['press_pull'] == pytest.approx(.040)
    assert cyl['d']['press_pull'] == pytest.approx(.024)   # oracle: needs the fillet
    sharp = references.lookup('tipT_d')
    assert sharp['d_env'] == sharp['d']
    assert references.lookup('heart') is None      # older single-object records are not used
    assert references.lookup('cylinder_a_repeat') is None   # labels are never guessed
