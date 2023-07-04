# Copyright (C) 2013 Johan Hake
#
# This file is part of DOLFINx.
#
# DOLFINx is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DOLFINx is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DOLFINx. If not, see <http://www.gnu.org/licenses/>.
#
# Copy all data, tests and demo to a given directory relative to the top DOLFINx
# source directory

import os
import re
import shutil
import sys

# Subdirectories
sub_directories = ["demo", "test"]

# Copy all files with the following suffixes
suffix_patterns = ["txt", "h", "hpp", "c", "cpp", "py", "xdmf", "h5"]

suffix_pattern = re.compile("(%s)," % ("|".join("[\\w-]+\\.%s" % pattern
                                                for pattern in suffix_patterns)))

script_rel_path = os.sep.join(__file__.split(os.sep)[:-1])
script_rel_path = script_rel_path or "."
dolfinx_dir = os.path.abspath(os.path.join(
    script_rel_path, os.pardir, os.pardir))


def copy_data(top_destdir, complex_mode):
    abs_destdir = top_destdir if os.path.isabs(
        top_destdir) else os.path.join(dolfinx_dir, top_destdir)

    if abs_destdir == dolfinx_dir:
        raise RuntimeError("destination directory cannot be the same as "
                           "the DOLFINx source directory")

    if not os.path.isdir(abs_destdir):
        raise RuntimeError("%s is not a directory." % abs_destdir)

    skip = set()
    if complex_mode is True:
        skip.update(["demo/hyperelasticity"])
    skip = {os.path.join(dolfinx_dir, skip_) for skip_ in skip}

    for subdir in sub_directories:
        top_dir = os.path.join(dolfinx_dir, subdir)
        for dirpath, dirnames, filenames in os.walk(top_dir):
            if dirpath not in skip:
                destdir = dirpath.replace(dolfinx_dir, abs_destdir)
                if not os.path.isdir(destdir):
                    os.makedirs(destdir)
                for f in re.findall(suffix_pattern, " ".join("%s," % f for f in filenames)):
                    srcfile = os.path.join(dirpath, f)
                    shutil.copy(srcfile, destdir)


if __name__ == "__main__":
    # Expecting a destination argument
    if len(sys.argv) == 2:
        # Call with "./copy-test-demo-data.py dest_dir" for PETSc real mode
        copy_data(sys.argv[-1], False)
    elif len(sys.argv) == 3:
        # Call with "./copy-test-demo-data.py dest_dir 1" for PETSc complex mode
        copy_data(sys.argv[-2], sys.argv[-1] == "1")
    else:
        raise RuntimeError(
            "Expecting either one or two arguments")
