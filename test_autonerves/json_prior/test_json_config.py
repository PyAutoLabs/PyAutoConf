import pytest

import autonerves as aconf
from autonerves.mock.mock_real import SphProfile


@pytest.fixture(name="geometry_profile_path")
def make_geometry_profile_path():
    return ["autonerves", "mock", "mock_real", "SphProfile"]


def test_path_for_class(geometry_profile_path):
    assert aconf.path_for_class(SphProfile) == geometry_profile_path


@pytest.mark.parametrize(
    "config_dict, paths",
    [
        (
            {
                "autonerves.mock.mock_real.SphProfile": "test",
                "autonerves.mock.mock_real.Other": "toast",
            },
            ["autonerves.mock.mock_real.SphProfile", "autonerves.mock.mock_real.Other"],
        ),
        (
            {"autonerves.mock.mock_real": {"SphProfile": "test", "Other": "toast"}},
            [
                "autonerves.mock.mock_real",
                "autonerves.mock.mock_real.SphProfile",
                "autonerves.mock.mock_real.Other",
            ],
        ),
        (
            {
                "autonerves": {
                    "mock": {"mock_real": {"SphProfile": "test", "Other": "toast"}}
                }
            },
            [
                "autonerves",
                "autonerves.mock",
                "autonerves.mock.mock_real",
                "autonerves.mock.mock_real.SphProfile",
                "autonerves.mock.mock_real.Other",
            ],
        ),
        (
            {
                "autonerves": {
                    "mock": {"mock_real.SphProfile": "test", "mock_real.Other": "toast"}
                }
            },
            [
                "autonerves",
                "autonerves.mock",
                "autonerves.mock.mock_real.SphProfile",
                "autonerves.mock.mock_real.Other",
            ],
        ),
        ({"SphProfile": "test", "Other": "toast"}, ["SphProfile", "Other"]),
        (
            {"mock_real.SphProfile": "test", "mock_real.Other": "toast"},
            ["mock_real.SphProfile", "mock_real.Other"],
        ),
        (
            {"mock_real": {"SphProfile": "test", "Other": "toast"}},
            ["mock_real", "mock_real.SphProfile", "mock_real.Other"],
        ),
    ],
)
def test_paths(config_dict, paths):
    config = aconf.JSONPriorConfig(config_dict)
    assert config.paths == paths


@pytest.mark.parametrize(
    "config_dict",
    [
        {
            "autonerves.mock.mock_real.SphProfile": "test",
            "autonerves.mock.mock_real.Other": "toast",
        },
        {"autonerves.mock.mock_real": {"SphProfile": "test", "Other": "toast"}},
        {"autonerves": {"mock": {"mock_real": {"SphProfile": "test", "Other": "toast"}}}},
        {
            "autonerves": {
                "mock": {"mock_real.SphProfile": "test", "mock_real.Other": "toast"}
            }
        },
        {"SphProfile": "test", "Other": "toast"},
        {"mock_real": {"SphProfile": "test", "Other": "toast"}},
    ],
)
def test_config_for_path(geometry_profile_path, config_dict):
    config = aconf.JSONPriorConfig(config_dict)
    assert config(geometry_profile_path) == "test"
    assert config(["autonerves", "mock", "mock_real", "Other"]) == "toast"


def test_path_double():
    config = aconf.JSONPriorConfig({"mock_real": {"SphProfile": "test"}})
    assert config(["something", "mock_real", "mock_real", "SphProfile"]) == "test"
