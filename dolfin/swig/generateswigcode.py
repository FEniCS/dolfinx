#!/usr/bin/env python
#
# Generate SWIG files for Python interface of DOLFIN
#
# Copyright (C) 2007 Anders Logg
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
# Modified by <aasmund@simula.no>
# Modified by Johan Hake, 2009-2012
# Modified by Kristian B. Oelgaard, 2010
#
# First added:  2007-04-12
# Last changed: 2012-01-17

import os
import re

# Template code for all combined SWIG modules
swig_template = r"""
%%{
#include <dolfin/dolfin.h>
#define PY_ARRAY_UNIQUE_SYMBOL %s
#include <numpy/arrayobject.h>
%%}

%%init%%{
import_array();
%%}

// Global shared ptr declarations
%%include "dolfin/swig/shared_ptr_classes.i"

// Global typemaps
%%include "dolfin/swig/typemaps.i"
%%include "dolfin/swig/std_pair_typemaps.i"
%%include "dolfin/swig/numpy_typemaps.i"
%%include "dolfin/swig/array_typemaps.i"
%%include "dolfin/swig/std_vector_typemaps.i"
%%include "dolfin/swig/std_set_typemaps.i"
%%include "dolfin/swig/std_map_typemaps.i"

// Global exceptions
%%include <exception.i>
%%include "dolfin/swig/exceptions.i"

// Do not expand default arguments in C++ by generating two an extra 
// function in the SWIG layer. This reduces code bloat.
%%feature("compactdefaultargs");

// STL SWIG string class
%%include <std_string.i>
"""

def create_combined_module_file(combined_module):
    """
    Create and initiate the main SWIG interface file for each
    comined module file
    """
    combined_module_file = open(os.path.join("modules", \
                                             combined_module+".i"), "w")
    

def extract_module_header_files(module):
    """
    Extract header files for a module
    """
    module_headers = []
    print "Processing dolfin_%s.h..." % module
    f = open("../%s/dolfin_%s.h" % (module, module))
    for line in f:
        if re.search("^#include ",line):
            header = line.split()[1].replace("<", "").replace(">", "")
            
            # Get just the file name (after last /) and check against excludes:
            if not header.split("/")[-1] in excludes:
                module_headers.append(header)
            
    return module_headers


def write_module_code(module, combined_module, combined_module_file):
    """
    Write SWIG module code.

    1) Append %include statements for a combined module
    2) Write an import file to facilitate SWIG type import for each module
    3) Write %shared_ptr statements which each combined module reads in
    
    """

    # Generate module imports
    import_file = open(os.path.join("import", module + ".i"), "w")
    import_file.write("// Auto generated import statements for the "\
                      "SWIG module: '%s'\n\n"% module)
    combined_module_file.write("\n// DOLFIN headers included from %s\n" % module)

    # Check if there is a foo_pre.i file
    if os.path.isfile(module+"_pre.i"):
        combined_module_file.write("%%include \"dolfin/swig/%s_pre.i\"\n" % module)

    # Iterate over all headers in the module
    for header in extract_module_header_files(module):
        
        # Write header include statement to the combined %include file
        combined_module_file.write("%%include \"%s\"\n" % header)

        # Write header import statement
        import_file.write('%%import(module="dolfin.cpp.%s") "%s"\n'%\
                          (combined_module, header))
    
    # Check if there is a foo_post.i file
    if os.path.isfile(module+"_post.i"):
        combined_module_file.write("%%include \"dolfin/swig/%s_post.i\"\n" % module)


# Combined modules with sub modules
combined_modules = dict(common = ["common", "parameter", "log", "io"],
                        la = ["la", "nls"],
                        mesh = ["mesh", "intersection", "refinement", "ale"],
                        function = ["function", "plot", "math"],
                        fem = ["fem", "adaptivity", "quadrature"])

# List of headers to exclude (add more here)
excludes = ["plot.h", "IntersectionOperatorImplementation.h" ]

# Extract header files
headers = []
for combined_module, modules in combined_modules.items():

    # Create a file being the root of the combined module
    combined_module_file = create_combined_module_file(combined_module)
    
    # Iterate over modules in each combined module and extract headers
    for module in modules:
        write_module_code(module, combined_module, combined_module_file)
        

# Generate list of header files
print "Generating file %s" % interface_file
f = open(interface_file, "w")
f.write("// Generated list of include files for PyDOLFIN\n")
for (module, module_headers) in headers:
    # Generate module imports
    f_import = open(os.path.join("import", module + ".i"), "w")
    f_import.write("// Auto generated import statements for the SWIG kernel module: '%s'\n\n"% module)
    f.write("\n// DOLFIN headers included from %s\n" % module)
    if os.path.isfile(module+"_pre.i"):
        f.write("%%include \"dolfin/swig/%s_pre.i\"\n" % module)
    for header in module_headers:
        f.write("%%include \"%s\"\n" % header)
        f_import.write('%%import(module="dolfin.cpp") "%s"\n'%header)
    if os.path.isfile(module+"_post.i"):
        f.write("%%include \"dolfin/swig/%s_post.i\"\n" % module)
    f_import.close()
f.close()

# Create docstrings.i file from docstrings module (only for dolfin.cpp)
from documentation import generate_docstrings
generate_docstrings()

# Extract all shared_ptr stored classes and store them in a pyton module
# and place that under dolfin.compilemodeuls.sharedptrclasses.py
shared_ptr_classes = re.findall("%shared_ptr\(dolfin::(.+)\)", \
                                open("shared_ptr_classes.i").read())

shared_ptr_classes = filter(lambda x: "NAME" not in x, shared_ptr_classes)
template = """
'''
This module contains the names of the classes in DOLFIN that is
stored using shared_ptrs. The file is automatically generated by the
generate.py script in the dolfin/swig directory.
'''

__all__ = ['shared_ptr_classes']

shared_ptr_classes = %s
"""

par = os.path.pardir
open(os.path.join(par, par, "site-packages", "dolfin", \
                  "compilemodules", "sharedptrclasses.py"), "w").write(\
    template%(repr(shared_ptr_classes)))
