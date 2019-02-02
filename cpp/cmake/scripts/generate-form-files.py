# Copyright (C) 2005-2019 Anders Logg and Garth N. Wells
#
# This file is part of DOLFIN.
#
# DOLFIN is free software: you can redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DOLFIN is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
# License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
"""Recompile all forms. This script should be run from the top level C++ directory."""

import os
import sys

import ffc

# Call with "./generate-form-files.py 1" for PETSc complex mode
complex_mode = (sys.argv[-1] == "1")

# UFL files to skip
skip = set()
if complex_mode is True:
    skip.update(["HyperElasticity.ufl"])

# Directories to scan
subdirs = ["demo", "test"]

# Compile all form files
topdir = os.getcwd()
failures = []
for subdir in subdirs:
    for root, dirs, files in os.walk(subdir):
        # Build list of UFL form files
        formfiles = [f for f in files if f[-4:] == ".ufl"]
        if not formfiles:
            continue

        # Compile files
        os.chdir(root)
        print("Compiling %d forms in %s..." % (len(formfiles), root))
        for f in set(formfiles) - skip:
            args = []
            if complex_mode:
                args += ["-f", "scalar_type", "double complex"]
            args.append(f)
            try:
                ffc.main(args)
            except Exception as e:
                failures.append((root, f, e))
        os.chdir(topdir)

# Raise exception of any error have been caught
if failures:
    s = ''.join("\nForm: {}/{}\nException type: {} \nMessage: {}\n".format(
        failure[0], failure[1],
        type(failure[2]).__name__, failure[2]) for failure in failures)
    raise RuntimeError("Failed to compile the forms:\n{}".format(s))
