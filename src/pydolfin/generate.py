#!/usr/bin/env python
#
# Generate list of include files for SWIG interface file.
# All files under src/kernel/ listed in a dolfin_foo.h are
# included in the interface.

__author__ = "Anders Logg (logg@simula.no)"
__date__ = "2007-04-12 -- 2007-04-13"
__copyright__ = "Copyright (C) 2007 Anders Logg"
__license__  = "GNU GPL Version 2"

import os

# Name of SWIG interface file to be generated
interface_file = "dolfin_headers.h"

# List of headers to exclude (add more here, just one for testing)
excludes = ["dolfin/Buffer.h"]

# Extract header files
headers = {}
for root, dirs, files in os.walk("../kernel"):
    if root[-6:] == "dolfin":
        for file in files:
            if file[:7] == "dolfin_" and file[-2:] == ".h":
                print "Processing %s..." % file
                headers[file] = []
                f = open("%s/%s" % (root, file), "r")
                for line in f:
                    if "#include " in line:
                        header = line.split()[1].replace("<", "").replace(">", "")
                        print header
                        if not header in excludes:
                            headers[file] += [header]
                f.close()

# Generate list of header files
print "Generating file %s" % interface_file
f = open(interface_file, "w")
f.write("// Generated list of include files for PyDOLFIN\n")
for key in headers:
    f.write("\n// DOLFIN headers included from %s\n" % key)
    for header in headers[key]:
        f.write("%%include \"%s\"\n" % header)
f.close()
