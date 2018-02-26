# Copyright (C) 2016 Garth N. Wells
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import os
import sys


# Get directory of this file
dir_path = os.path.dirname(os.path.realpath(__file__))

# Directories to scan
subdirs = [dir_path]

# Check that we can find pylint.py
parser = dir_path + "/../../utils/pylit/pylit.py"
if os.path.isfile(parser):
    pass
else:
    raise RuntimeError("Cannot find pylit.py. Are you running from the top level directory")

# Get absolute path to parser
parser = os.path.abspath(parser)

# Generate .py files from all .py.rst files
topdir = os.getcwd()
for subdir in subdirs:
    for root, dirs, files in os.walk(subdir):

        # Check for .py.rst files
        rstfiles = [f for f in files if len(f) > 7 and f[-7:] == ".py.rst"]
        if len(rstfiles) == 0:
            continue

        # Compile files
        os.chdir(root)
        print("Converting rst files in in {} ...".format(root))
        for f in rstfiles:
            command = sys.executable + " " + parser + " " + os.path.abspath(f)
            #print("  " + command)
            ret = os.system(command)
            if not ret == 0:
                raise RuntimeError("Unable to convert rst file to a .py ({})".format(f))

        os.chdir(topdir)
