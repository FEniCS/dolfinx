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
# Modified by Garth N. Wells, 2013
#
# Utility script for splitting the cpp and python demos into separate
# directory trees. Ignores cmake files for python.

from __future__ import print_function
import sys, os, shutil
import subprocess

index_template = """
Collection of documented demos
==============================

.. toctree::
   :glob:
   :numbered:
   :maxdepth: 1

   */*/%s/documentation

%s

.. note::

    You must have a working installation of FEniCS in order to run the
    demos.

"""


run_cpp_demos = """
To run the C++ demos, follow the below procedure:

* Download the source files i.e., ``main.cpp`` and ``CMakeLists.txt``,
  from the demo that you want to run. Some demos also contain UFL form
  files, e.g., ``Poisson.ufl``. Note that there may be multiple form
  files.

* Compile the form files to generate code with DOLFIN wrappers::

      $ ffc -l dolfin Poisson.ufl

  Generated .h files are usually distributed with the demos so you may
  choose to skip this step and use the provided header file directly,
  in this case ``Poisson.h``.

  If you wish to use optimized generated code, do::

      $ ffc -O -l dolfin Poisson.ufl

* Configure the demo build process::

      $ cmake .

* Compile the demo::

      $ make

* Run the created executable::

    $ ./demo

"""

run_python_demos = """
To run the Python demos, follow the below procedure:

* Download the source file, e.g., ``demo_poisson.py``, for the demo that you
  want to run.

* Use the Python interpreter to run this file::

      $ python demo.py

"""

run_instructions = {"cpp": run_cpp_demos, "python": run_python_demos}

def generate_main_index_file(output_dir, language):

    filename = os.path.join(output_dir, "index.rst")
    file = open(filename, "w")
    text = index_template % (language, run_instructions[language])
    file.write(text)
    file.close()


def copy_split_demo_doc(input_dir, cpp_output_dir, python_output_dir):

    # Get list of files in demo directories
    try:

        # Get root path of git repo
        git_root  = subprocess.check_output(["git", "rev-parse", "--show-toplevel"]).splitlines()

        # Get list of files tracked by git (relative to git root)
        git_files = subprocess.check_output(["git", "ls-files", "--full-name", input_dir]).splitlines()

        # Build list with full paths
        git_files = [os.path.join(git_root[0],  f) for f in git_files]

        if not git_files:
            # Workaround for when we're not in a git repo
            git_files = subprocess.check_output(["find", input_dir]).splitlines()
    except:
        git_files = None

    def ignore_cpp(directory, contents):
        if directory[-3:] == "cpp":
            return contents
        elif git_files is not None:
            return [c for c in contents if not in_git(directory, c, git_files, ["cpp"]) ]
        else:
            return []

    def ignore_python(directory, contents):
        if directory[-6:] == "python":
            return contents
        elif git_files is not None:
            return [c for c in contents if not in_git(directory, c, git_files, ["python"])]
        else:
            return []

    # Copy demo tree to cpp_output_dir ignoring python demos
    try:
        shutil.rmtree(cpp_output_dir)
    except:
        pass
    shutil.copytree(input_dir, cpp_output_dir, ignore=ignore_python)

    # In addition, generate main index file for navigating demos
    generate_main_index_file(cpp_output_dir, "cpp")

    # Copy demo tree to python_output_dir ignoring cpp demos
    try:
        shutil.rmtree(python_output_dir)
    except:
        pass
    shutil.copytree(input_dir, python_output_dir, ignore=ignore_cpp)

    # In addition, generate main index file for navigating demos
    generate_main_index_file(python_output_dir, "python")


def in_git(directory, name, git_files, exclude_dirs=[]):
    "Check whether file is version-controlled"

    # Make sure we have the real path (remove symlinks)
    directory = os.path.realpath(directory)

    # Get file full path
    f = os.path.join(directory, name)

    # Return true if a directory (git doesn't track directories)
    if os.path.isdir(f):
        if name in exclude_dirs:
            return False
        else:
            return True
    return f in git_files


if __name__ == "__main__":

    args = sys.argv[1:]

    if len(args) != 3:
        usage= "Usage: python copy_and_split_demo_doc.py input_dir cpp_output_dir python_output_dir"
        print(usage)
        sys.exit(2)

    copy_split_demo_doc(args[0], args[1], args[2])
