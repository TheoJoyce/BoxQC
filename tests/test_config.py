import json

import pytest
import yaml

from boxqc.config import load_config
from boxqc.device.hardware import default_buzzer_setting
from boxqc.device.settings import RuntimeSettings
from boxqc.studio.state import StudioState, effective_roi


def test_load_config_resolves_paths_and_rejects_typos(tmp_path):
    f = tmp_path / "boxqc.yaml"
    f.write_text(yaml.safe_dump({"data_dir": "d", "inference": {"conf_threshold": 0.3}}))
    cfg = load_config(f)
    assert cfg.data_dir == tmp_path / "d"
    assert cfg.inference.conf_threshold == 0.3

    f.write_text(yaml.safe_dump({"inference": {"conf_treshold": 0.3}}))
    with pytest.raises(ValueError, match="inference.conf_treshold"):
        load_config(f)


def test_local_overlay_and_environment_override(tmp_path, monkeypatch):
    f = tmp_path / "boxqc.yaml"
    f.write_text(yaml.safe_dump({"device": {"url": "http://10.42.0.2:8000/", "port": 9000}, "classes": ["a"]}))
    (tmp_path / "boxqc.local.yaml").write_text(yaml.safe_dump({"device": {"url": "http://pi.local:8000"}}))
    cfg = load_config(f)
    assert cfg.device.url == "http://pi.local:8000"     # overlay wins...
    assert cfg.device.port == 9000 and cfg.classes == ["a"]   # ...key by key
    assert cfg.overlay == tmp_path / "boxqc.local.yaml"

    monkeypatch.setenv("BOXQC_DEVICE_URL", "http://env:1/")
    monkeypatch.setenv("BOXQC_API_TOKEN", "s3cret")
    cfg = load_config(f)
    assert cfg.device.url == "http://env:1" and cfg.device.api_token == "s3cret"


def test_legacy_server_and_remote_keys_still_load(tmp_path):
    f = tmp_path / "boxqc.yaml"
    f.write_text(yaml.safe_dump({"server": {"host": "127.0.0.1", "port": 8123}, "remote": {"url": "http://pi:8000"}}))
    cfg = load_config(f)
    assert (cfg.device.host, cfg.device.port, cfg.device.url) == ("127.0.0.1", 8123, "http://pi:8000")

    f.write_text(yaml.safe_dump({"server": {"prot": 1}}))
    with pytest.raises(ValueError, match="Unknown config key: server"):
        load_config(f)


def test_runtime_settings_persist_and_validate(cfg):
    path = cfg.data_dir / "settings.json"
    s = RuntimeSettings(path, cfg)
    s.update(conf_threshold=0.7, roi=[0.1, 0.1, 0.9, 0.9])
    assert RuntimeSettings(path, cfg).get("conf_threshold") == 0.7
    with pytest.raises(ValueError):
        s.update(roi=[0.5, 0.5, 0.4, 0.9])
    with pytest.raises(ValueError):
        s.update(capture_dataset="x")       # a studio setting, not a device one

    # a settings file written by 0.1 has that key; it is ignored and dropped on the next save
    path.write_text(json.dumps({"conf_threshold": 0.6, "roi": [0, 0, 1, 1], "active_model": "m.onnx",
                                "capture_dataset": "Testing Data"}))
    old = RuntimeSettings(path, cfg)
    assert old.as_dict() == {"conf_threshold": 0.6, "roi": [0, 0, 1, 1], "active_model": "m.onnx",
                             "buzzer": default_buzzer_setting()}     # added in a later version
    old.update(conf_threshold=0.5)
    assert "capture_dataset" not in json.loads(path.read_text())


def test_studio_state_starts_from_a_0_1_settings_file(cfg):
    cfg.data_dir.mkdir(parents=True)
    (cfg.data_dir / "settings.json").write_text(json.dumps(
        {"roi": [0.5, 0.0, 1.0, 1.0], "capture_dataset": "Testing Data", "active_model": "m.onnx"}))
    state = StudioState(cfg.data_dir)
    assert state.as_dict() == {"capture_dataset": "Testing Data", "roi": [0.5, 0.0, 1.0, 1.0]}
    assert effective_roi(cfg) == [0.5, 0.0, 1.0, 1.0]

    state.update(roi=[0.0, 0.0, 0.5, 0.5])
    assert (cfg.data_dir / "studio.json").is_file()
    assert effective_roi(cfg) == [0.0, 0.0, 0.5, 0.5]   # studio.json now wins over the old file


def test_wrong_types_and_broken_yaml_are_configuration_errors(tmp_path):
    f = tmp_path / "boxqc.yaml"
    for text, message in [
        ("device:\n  port: eight thousand\n", "device.port must be a whole number"),
        ("device:\n  port: 80.5\n", "device.port must be a whole number"),
        ("classes: dirty\n", "classes must be a list"),
        ("roi: [0, 0, wide, 1]\n", r"roi\[2\] must be a number"),
        ("inference:\n  save_inspections: 1\n", "inference.save_inspections must be true or false"),
        ("hardware:\n  buzzer_pin: seventeen\n", "hardware.buzzer_pin must be a whole number"),
        ("data_dir: [a]\n", "data_dir must be a path"),
        ("camera: 5\n", "camera must be a mapping"),
        ("device:\n  port: [8000\n", "not valid YAML"),
    ]:
        f.write_text(text)
        with pytest.raises(ValueError, match=message):
            load_config(f)


def test_harmless_type_differences_are_accepted(tmp_path):
    f = tmp_path / "boxqc.yaml"
    f.write_text("""
train:
  device: 0              # a GPU index is a number in YAML, text in the config
  val_fraction: 1        # a whole number where a decimal is expected
hardware:
  buzzer_pin: null       # optional values may be null
  passive_frequency: 2700
roi: [0, 0, 1, 1]
camera:                  # a section with every key commented out
  # backend: mock
""")
    cfg = load_config(f)
    assert cfg.train.device == "0" and cfg.train.val_fraction == 1.0
    assert cfg.hardware.buzzer_pin is None and cfg.hardware.passive_frequency == 2700.0
    assert cfg.roi == [0.0, 0.0, 1.0, 1.0] and cfg.camera.backend == "auto"
