"""The documentation has to match the code. These tests fail when it drifts."""

import dataclasses
import re
from pathlib import Path

import pytest

from boxqc.cli import build_parser
from boxqc.config import Config, load_config

ROOT = Path(__file__).parent.parent
MARKDOWN = sorted(p for p in ROOT.rglob("*.md")
                  if not any(part.startswith(".") or part in ("work", "data", "models") for part in p.parts))


def test_every_device_route_is_documented(device):
    """docs/device-api.md is the reference for the public API, so it must list every route."""
    text = (ROOT / "docs" / "device-api.md").read_text()
    paths = device.get("/openapi.json").json()["paths"]
    assert paths, "the device app has no documented routes?"
    for path, methods in paths.items():
        route = path.removeprefix("/api/v1")
        # a path parameter may be named differently in the docs: {item_id} matches {id}
        pattern = re.sub(r"\\\{[^}]*\}", r"\\{[^}]+\\}", re.escape(route))
        for method in methods:
            assert re.search(rf"`{method.upper()} {pattern}[`?]", text), \
                f"{method.upper()} {route} is missing from docs/device-api.md"


def test_boxqc_yaml_lists_every_config_key():
    """boxqc.yaml doubles as the configuration reference: each key needs its commented entry."""
    text = (ROOT / "boxqc.yaml").read_text()

    def keys(obj, indent=""):
        for field in dataclasses.fields(obj):
            if field.name in ("source", "overlay"):
                continue
            yield indent, field.name
            value = getattr(obj, field.name)
            if dataclasses.is_dataclass(value):
                yield from keys(value, indent + "  ")

    for indent, name in keys(Config()):
        assert re.search(rf"^{indent}{name}:", text, re.MULTILINE), f"{name} is missing from boxqc.yaml"
    load_config(ROOT / "boxqc.yaml")       # and the file itself loads


def test_every_command_is_in_the_command_reference():
    text = (ROOT / "docs" / "cli.md").read_text()
    subparsers = next(a for a in build_parser()._actions if a.dest == "command")
    commands = {action.dest for action in subparsers._choices_actions}      # those with help text
    assert {"device", "studio", "train", "deploy", "export", "evaluate", "inspect", "plugins"} <= commands
    for command in commands:
        assert f"## `boxqc {command}" in text, f"`boxqc {command}` is missing from docs/cli.md"
        for action in subparsers.choices[command]._actions:
            for option in action.option_strings:
                if option.startswith("--") and option not in ("--help", "--url", "--push"):
                    assert option in text, f"{option} of `boxqc {command}` is missing from docs/cli.md"


@pytest.mark.parametrize("path", MARKDOWN, ids=lambda p: str(p.relative_to(ROOT)))
def test_relative_links_point_at_existing_files(path):
    for target in re.findall(r"\]\(([^)#\s]+)(?:#[^)]*)?\)", path.read_text()):
        if "://" in target or target.startswith("mailto:"):
            continue
        assert (path.parent / target).exists(), f"{path.relative_to(ROOT)} links to missing {target}"
