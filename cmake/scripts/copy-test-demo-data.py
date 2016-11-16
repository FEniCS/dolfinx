# Copyright (C) 2013 Johan Hake
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
# Copy all data, tests and demo to a given directory relative to the top dolfin
# source directory

import os
import sys
import re
import shutil

# Subdirectories
sub_directories = ['demo', 'test', 'bench']

# Copy all files with the following suffixes
suffix_patterns = ["py", "h", "cpp", "cpp.rst", "ufl", "ufl.rst", "xml", "xml.gz", "off", "inp", \
                   "msh", "supp", "rst", "py.rst", "ele", "node", "grid", "edge", "sh"]

suffix_pattern = re.compile("(%s),"%("|".join("[\w-]+\.%s" % pattern \
                                             for pattern in suffix_patterns)))

script_rel_path = os.sep.join(__file__.split(os.sep)[:-1])
script_rel_path = script_rel_path or "."
dolfin_dir = os.path.abspath(os.path.join(script_rel_path, os.pardir, os.pardir))

def copy_data(top_destdir):

    abs_destdir = top_destdir if os.path.isabs(top_destdir) else os.path.join(dolfin_dir, top_destdir)

    if abs_destdir == dolfin_dir:
        raise RuntimeError("destination directory cannot be the same as "\
                           "the dolfin source directory")

    if not os.path.isdir(abs_destdir):
        raise RuntimeError("%s is not a directory." % abs_destdir)

    for subdir in sub_directories:

        top_dir = os.path.join(dolfin_dir, subdir)
        for dirpath, dirnames, filenames in os.walk(top_dir):
            destdir = dirpath.replace(dolfin_dir, abs_destdir)
            if not os.path.isdir(destdir):
                os.makedirs(destdir)
            for f in re.findall(suffix_pattern, " ".join(\
                "%s,"%f for f in filenames)):
                srcfile = os.path.join(dirpath, f)
                shutil.copy(srcfile, destdir)

if __name__ == "__main__":
    # Expecting a destination argument
    if len(sys.argv) != 2:
        raise RuntimeError("Expecting 1 argument with the destination directory")

    copy_data(sys.argv[-1])
