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
