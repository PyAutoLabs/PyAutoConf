#!/usr/bin/env python
"""
Converts JSON prior configs to YAML equivalent.

Usage:
./convert_prior_configs.py /path/to/prior/directory
"""
import json
import os
import sys
from pathlib import Path

import oyaml as yaml

ORDER = [
    "type",
    "mean",
    "sigma",
    "lower_limit",
    "upper_limit",
    "width_modifier",
    "limits",
]

for path in Path(sys.argv[1]).rglob("*.json"):
    with open(path) as f:
        d = json.load(f)

    with open(path.with_suffix(".yaml"), "w") as f:
        yaml.dump(d, f)

    os.remove(path)


def sort_dict(obj):
    if isinstance(obj, dict):
        return {
            key: sort_dict(value)
            for key, value in sorted(
                obj.items(),
                key=lambda item: ORDER.index(item[0]) if item[0] in ORDER else 999,
            )
        }
    if isinstance(obj, list):
        return list(map(sort_dict, obj))
    return obj


for path in Path(sys.argv[1]).rglob("*.yaml"):
    with open(path) as f:
        d = yaml.safe_load(f)

    print(sort_dict(d))
    with open(path, "w") as f:
        yaml.dump(sort_dict(d), f)
