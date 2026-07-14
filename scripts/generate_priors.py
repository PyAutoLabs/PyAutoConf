#!/usr/bin/env python
"""
Generate prior configuration for all classes in a package, recursively
"""

from sys import argv

from autoconf.json_prior import generate

if __name__ == "__main__":
    try:
        generate.generate(argv[1])
    except KeyError:
        print("Usage: ./generate_priors.py /path/to/project")
