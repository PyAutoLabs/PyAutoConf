import pytest

from autonerves import conf
from autonerves.conf import with_config


@pytest.fixture(autouse=True)
def push_configs(files_directory):
    conf.instance.push(files_directory / "config")
    conf.instance.push(files_directory / "default")


@with_config("general", "output", "identifier_version", value=9)
def test_with_config():
    assert conf.instance["general"]["output"]["identifier_version"] == 9


def test_config():
    assert conf.instance["general"]["output"]["identifier_version"] == 4
