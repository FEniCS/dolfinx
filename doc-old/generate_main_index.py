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
# Utility script for creating main index.rst files in sphinx
# documentation projects in DOLFIN.

from __future__ import print_function
import os, sys

index_template = """

%(formatting)s
%(title)s
%(formatting)s

%(body)s

.. toctree::
   :hidden:

   programmers-reference/index
   demo/index
   quick_reference
"""

body = """
The Demos
=========

:doc:`The demos <demo/index>` (Collection of documented demo programs)

The Programmer's Reference
==========================

:ref:`Programmer's Reference Index <genindex>` (Classes, functions, terms)

"""
modules = """
DOLFIN modules
==================

:ref:`DOLFIN Module Index <modindex>` (Modules)

"""

bodies = {"C++": body, "Python": (body + modules)}

def generate_main_index_file(output_dir, language, version):

    filename = os.path.join(output_dir, "index.rst")
    file = open(filename, "w")
    title = "Documentation for DOLFIN-%s (%s)" % (version, language)
    text = index_template % {"title": title, "formatting": len(title)*"#",
                             "body": bodies[language]}
    file.write(text)
    file.close()

if __name__ == "__main__":

    args = sys.argv[1:]

    if len(args) != 3:
        usage= "Usage: python generate_main_index.py cpp_output_dir python_output_dir version"
        print(usage)
        sys.exit(2)

    generate_main_index_file(args[0], "C++", args[2])
    generate_main_index_file(args[1], "Python", args[2])


