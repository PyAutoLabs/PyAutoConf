import json
import os
from pathlib import Path

import pytest

from autonerves.json_prior import generate as g

directory = Path(__file__).parent
package_directory = directory / "source_code"
module_path = package_directory / "module.py"


@pytest.fixture(name="prior_json")
def make_prior_json():
    return {
        "MyClass": {
            "simple": {
                "limits": {"lower": 0.0, "upper": 1.0},
                "lower_limit": 0.0,
                "type": "Uniform",
                "upper_limit": 1.0,
                "width_modifier": {"type": "Absolute", "value": 0.2},
            },
            "tup_0": {
                "limits": {"lower": 0.0, "upper": 1.0},
                "lower_limit": 0.0,
                "type": "Uniform",
                "upper_limit": 1.0,
                "width_modifier": {"type": "Absolute", "value": 0.2},
            },
            "tup_1": {
                "limits": {"lower": 0.0, "upper": 1.0},
                "lower_limit": 0.0,
                "type": "Uniform",
                "upper_limit": 1.0,
                "width_modifier": {"type": "Absolute", "value": 0.2},
            },
        }
    }


@pytest.fixture(autouse=True)
def cleanup():
    yield
    try:
        os.remove("priors/module.json")
    except FileNotFoundError:
        pass


def test_generate_for_file(prior_json):
    result = g.for_file(module_path)
    assert result == prior_json


def test_generate(prior_json):
    g.generate(package_directory)

    with open(f"priors/module.json") as f:
        assert json.load(f) == prior_json
