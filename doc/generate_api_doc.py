#!/usr/bin/env python
#
# Copyright (C) 2011 Marie E. Rognes
#
# This file is part of DOLFIN.
#
# DOLFIN is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DOLFIN is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
#
# Utility script for generating .rst documentation for DOLFIN

from __future__ import print_function
import os, sys

dolfin_import_error_msg = """
Unable to import the %s module
Error: %s
Did you forget to update your PYTHONPATH variable?"""

try:
    import dolfin_utils
except Exception as what:
    raise ImportError(dolfin_import_error_msg % ("dolfin_utils", what))

from dolfin_utils.documentation import generate_cpp_api_documentation
from dolfin_utils.documentation import generate_python_api_documentation

def generate_dolfin_doc(input_dir, output_dir, version=None):

    if version is None:
        version = "dev"

    # Make output directory (or use current if existing)
    try:
        os.makedirs(output_dir)
    except:
        pass

    # Generate .rst for C++ documentation
    api_output_dir = os.path.join(output_dir, "cpp", "programmers-reference")
    generate_cpp_api_documentation(input_dir, api_output_dir, version)

    # Try to import DOLFIN Python module
    module_name = "dolfin"
    try:
        exec("import %s" % module_name)
        exec("module = %s" % module_name)
    except Exception as what:
        raise ImportError(dolfin_import_error_msg % (module_name, what))

    # Generate .rst for Python documentation
    api_output_dir = os.path.join(output_dir, "python", "programmers-reference")
    generate_python_api_documentation(module, api_output_dir, version)

    print("\nSuccessfully generated API documentation.\n")

if __name__ == "__main__":

    args = sys.argv[1:]

    if len(args) != 3:
        usage= "Usage: python generate_api_doc.py source_dir output_dir version"
        print(usage)
        sys.exit(2)

    generate_dolfin_doc(args[0], args[1], args[2])
