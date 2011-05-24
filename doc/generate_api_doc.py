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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
#
# Utility script for generating .rst documentation for DOLFIN

import sys
from dolfin_utils.documentation import generate_dolfin_doc

if __name__ == "__main__":

    args = sys.argv[1:]

    if len(args) != 3:
        usage= "Usage: python generate_api_doc.py source_dir output_dir version"
        print usage
        sys.exit(2)

    generate_dolfin_doc(args[0], args[1], args[2])
