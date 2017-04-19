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

def process():
    """Copy demo rst files (C++ and Python) from the DOLFIN source tree
    into the demo source tree, and process file with pylit

    """

    # Create demo CMakeLists.txt files
    current_dir = os.getcwd()
    os.chdir("../../")
    os.system(sys.executable + ' ./cmake/scripts/generate-cmakefiles.py')
    os.chdir(current_dir)

    # Check that we can find pylint.py for converting foo.py.rst to
    # foo.py
    pylit_parser = "../../utils/pylit/pylit.py"
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

            # Process C++ and Python versions
            for version in ("cpp", "python"):

                # Get path to demo source directory (cpp/python) and
                # check that it exists
                version_path = os.path.join(path, version)
                if not os.path.isdir(version_path):
                    continue

                # Build list of rst files in demo source directory
                rst_files = [f for f in os.listdir(version_path) if os.path.splitext(f)[1] == ".rst" ]

                # Create directory in documentation tree for demo
                demo_dir = os.path.join('./demos/', demo, version)
                if not os.path.exists(demo_dir):
                    os.makedirs(demo_dir)

                # Copy rst files into documentation demo directory and process with Pylit
                for f in rst_files:
                    shutil.copy(os.path.join(version_path, f), demo_dir)

                    # Run pylit on cpp.rst and py.rst files (file with 'double extensions')
                    if os.path.splitext(os.path.splitext(f)[0])[1] in (".py", ".cpp", ".ufl"):
                        rst_file = os.path.join(demo_dir, f)
                        command = pylit_parser + " " + rst_file
                        ret = os.system(command)
                        if not ret == 0:
                            raise RuntimeError("Unable to convert rst file to a .cpp/py ({})".format(f))

                # Copy CMakeLists.txt file (if it exists)
                cmake_file = os.path.join(version_path, "CMakeLists.txt")
                if os.path.isfile(cmake_file):
                    shutil.copy(cmake_file, demo_dir)
