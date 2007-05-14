#!/usr/bin/env python
#
# Generate list of include files for SWIG interface file.

__author__ = "Anders Logg (logg@simula.no)"
__date__ = "2007-04-12 -- 2007-05-02"
__copyright__ = "Copyright (C) 2007 Anders Logg"
__license__  = "GNU LGPL Version 2.1"

import os
import re

# List of headers to exclude (add more here)
excludes = ["plot.h", "ParameterSystem.h", "ParameterList.h", \
            "ConvectionMatrix.h", "MassMatrix.h", "StiffnessMatrix.h", "LoadVector.h"]

# Name of SWIG interface file to be generated
interface_file = "dolfin_headers.h"

# Extract modules from dolfin.h
modules = []
f = open("../kernel/main/dolfin.h")
for line in f:
    if "#include <dolfin/" in line and line[:17] == "#include <dolfin/":
        module = line.split("/")[1].split(".")[0].split("_")[1]
        modules += [module]
f.close()

# Extract header files
headers = []
for module in modules:
    module_headers = []
    print "Processing dolfin_%s.h..." % module
    f = open("../kernel/%s/dolfin/dolfin_%s.h" % (module, module))
    for line in f:
        if re.search("^#include ",line):
            header = line.split()[1].replace("<", "").replace(">", "")
            if not header.split("/")[1] in excludes:
                module_headers += [header]
    f.close()
    headers += [(module, module_headers)]

# Generate list of header files
print "Generating file %s" % interface_file
f = open(interface_file, "w")
f.write("// Generated list of include files for PyDOLFIN\n")
for (module, module_headers) in headers:
    f.write("\n// DOLFIN headers included from %s\n" % module)
    for header in module_headers:
        f.write("%%include \"%s\"\n" % header)
f.close()
