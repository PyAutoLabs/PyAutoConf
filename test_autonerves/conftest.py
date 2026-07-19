import pathlib

import pytest

from autonerves import conf


@pytest.fixture(scope="session", name="files_directory")
def make_files_directory():
    return pathlib.Path(__file__).parent / "files"


@pytest.fixture(scope="session", name="session_config")
def make_session_config(files_directory):
    return conf.Config(
        files_directory / "config",
        files_directory / "default",
    )


@pytest.fixture(name="config")
def make_config(files_directory):
    return conf.Config(
        files_directory / "config",
        files_directory / "default",
    )
