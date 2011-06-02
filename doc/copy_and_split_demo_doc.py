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
# Utility script for splitting the cpp and python demos into separate
# directory trees. Ignores cmake files for python.

import sys, os, shutil

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

    def ignore_cpp(directory, contents):
        if directory[-3:] == "cpp":
            return contents
        else:
            return [c for c in contents if "cmake" in c.lower()]

    def ignore_python(directory, contents):
        if directory[-6:] == "python":
            return contents
        else:
            return ()

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


if __name__ == "__main__":

    args = sys.argv[1:]

    if len(args) != 3:
        usage= "Usage: python copy_and_split_demo_doc.py input_dir cpp_output_dir python_output_dir"
        print usage
        sys.exit(2)

    copy_split_demo_doc(args[0], args[1], args[2])
