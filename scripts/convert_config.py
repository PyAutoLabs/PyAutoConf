#!/usr/bin/env python
"""
Converts the configuration files and directories in a given directory into YAML configs.

Usage:
./convert_config.py path/to/directory
"""

import os
import shutil
import sys
from pathlib import Path

import yaml

from autoconf.directory_config import RecursiveConfig, YAMLConfig

target_path = Path(sys.argv[1])

config = RecursiveConfig(str(target_path))

for key in config.keys():
    value = config[key]
    if isinstance(value, YAMLConfig):
        continue

    d = value.dict()
    path = target_path / key
    with open(path.with_suffix(".yaml"), "w") as f:
        yaml.dump(d, f)

    try:
        os.remove(path.with_suffix(".ini"))
    except FileNotFoundError:
        pass

    shutil.rmtree(path, ignore_errors=True)
