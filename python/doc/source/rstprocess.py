# -*- coding: utf-8 -*-
# Copyright (C) 2017 Garth N. Wells
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

import sys
import os
import shutil

sys.path.append('../../../utils/pylit/')
try:
    import pylit
except ImportError:
    raise ImportError("Unable to import pylit module")


def process():
    """Copy demo rst files (C++ and Python) from the DOLFIN source tree
    into the demo source tree, and process file with pylit

    """

    # Check that we can find pylint.py for converting foo.py.rst to
    # foo.py
    pylit_parser = "../../../utils/pylit/pylit.py"
    if os.path.isfile(pylit_parser):
        pass
    else:
        raise RuntimeError("Cannot find pylit.py")

    # Directories to scan
    subdirs = ["../../demo/documented"]

    # Iterate over subdirectories containing demos
    for subdir in subdirs:

        # Get list of demos (demo name , subdirectory)
        demos = [(dI, os.path.join(subdir, dI)) for dI in os.listdir(subdir) if os.path.isdir(os.path.join(subdir, dI))]

        # Iterate over demos
        for demo, path in demos:

            # Make demo doc directory
            demo_dir = os.path.join('./demos/', demo)
            if not os.path.exists(demo_dir):
                os.makedirs(demo_dir)

            #for f in rst_files_common:
            #    shutil.copy(os.path.join(path, f), demo_dir)

            # Build list of rst and png files in demo source directory
            rst_files = [f for f in os.listdir(path) if os.path.splitext(f)[1] == ".rst" ]
            other_files = [f for f in os.listdir(path) if os.path.splitext(f)[1] in (".png", ".py", ".gz")]

            # Create directory in documentation tree for demo
            demo_dir = os.path.join('./demos/', demo)
            if not os.path.exists(demo_dir):
                os.makedirs(demo_dir)

            # Copy .png and .py files into documentation demo directory
            for f in other_files:
                shutil.copy(os.path.join(path, f), demo_dir)

            # Copy rst files into documentation demo directory
            for f in rst_files:
                shutil.copy(os.path.join(path, f), demo_dir)

                # Copy rst files into documentation demo directory and
            # process with Pylit
            for f in rst_files:
                shutil.copy(os.path.join(path, f), demo_dir)

                # Run pylit on py.rst files (files with 'double
                # extensions')
                if os.path.splitext(os.path.splitext(f)[0])[1] in (".py"):
                    rst_file = os.path.join(demo_dir, f)
                    pylit.main([rst_file])


if __name__ == "__main__":
    process()
