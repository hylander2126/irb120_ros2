import json

import pytest

from irb120_control.util import episode as module


def test_snapshot_episode_allows_unconfigured_safe_object_name(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "_episodes_dir", lambda object_name="*": tmp_path / object_name)
    monkeypatch.setattr(module, "git_commit_info", lambda: {})

    episode = module.Episode.create(
        "cylinder_r5", allow_unconfigured_object=True, note="snapshot only")

    data = json.loads((episode.path / "episode.json").read_text())
    assert data["object"] == "cylinder_r5"
    assert data["object_params"] == {}
    assert data["object_params_configured"] is False


def test_motion_episode_still_rejects_unconfigured_object(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "_episodes_dir", lambda object_name="*": tmp_path / object_name)
    with pytest.raises(ValueError, match="Unknown object"):
        module.Episode.create("cylinder_r5")


@pytest.mark.parametrize("name", ["../escape", "heart/test", "", "bad name"])
def test_snapshot_episode_rejects_unsafe_object_name(tmp_path, monkeypatch, name):
    monkeypatch.setattr(module, "_episodes_dir", lambda object_name="*": tmp_path / object_name)
    with pytest.raises(ValueError, match="Object name"):
        module.Episode.create(name, allow_unconfigured_object=True)


def test_object_params_keep_the_legacy_values():
    """object_params.json moved to irb120_models; the numbers handed to arc_static must not."""
    from irb120_control.util.runtime_log_dir import load_object_params
    box = load_object_params("box")
    assert box["mass_gt"] == pytest.approx(0.676)
    assert box["com_gt"] == pytest.approx([0.05, 0.0, 0.15])
    # theta* now follows the measured CoM, not the CAD value (17.532) the JSON stored.
    assert box["theta_star"] == pytest.approx(18.4349, abs=1e-3)
    assert box["force_ref_n"] == pytest.approx(5.0)
    assert load_object_params("heart")["force_ref_n"] == pytest.approx(7.0)
    assert "force_ref_n" not in load_object_params("tipT")  # needs --force-ref
