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
import glob
import time
from codesnippets import *

# Create time info for labeling generated code
_local_time = time.localtime()
_date_form = dict(year = _local_time.tm_year,
                  month = _local_time.tm_mon,
                  day = _local_time.tm_yday)

# Create form for copyright statement to a SWIG interface file
_copyright_form_swig = dict(comment = r"//")
_copyright_form_swig.update(_date_form)

# Combined modules with sub modules
combined_modules = dict(common = ["common", "parameter", "log", "io"],
                        la = ["la", "nls"],
                        mesh = ["mesh", "intersection", "refinement", "ale"],
                        function = ["function", "plot", "math"],
                        fem = ["fem", "adaptivity", "quadrature"])

# List of headers to exclude (add more here)
excludes = ["plot.h", "IntersectionOperatorImplementation.h" ]

def create_combined_module_file(combined_module):
    """
    Create and initiate the main SWIG interface file for each
    comined module file
    """

    # Open file
    combined_module_file = open(os.path.join("modules", \
                                             combined_module+".i"), "w")

    combined_module_file.write(copyright_statement%(_copyright_form_swig))

    combined_module_file.write()
    
    # FIXME: Continue here
    return combined_module_file

def generate_typemap_includes():
    """
    Generate an include file which includes all typemap files under

       dolfin/swig/typemaps
       
    """

    include_file = open(os.path.join("typemaps", "includes.i"), "w")
    include_file.write(copyright_statement%(_copyright_form_swig))
    include_file.write("""
//-----------------------------------------------------------------------------
// Include all global typemap files
//-----------------------------------------------------------------------------
""")
    for typemap_file in glob.glob("typemaps/*.i"):
        include_file.write("%%include dolfin/swig/typemaps/%s\n"%typemap_file)

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

def write_module_code(module, combinedmodule):
    """
    Write SWIG module code.

    1) Write include.i file which consist of include of dolfin header files
    2) Write an import file to facilitate SWIG type import for each module
    
    """

    def write_include_modifier(module, modifier):
        """
        Write an include statements for pre or post modifier
        """
        if os.path.isfile(os.path.join(module, modifier + ".i")):
            files[file_type].write("%%include \"dolfin/swig/%s/%s.i\"\n" % \
                                   (module, modifier))

    # Get all headers in module
    headers = extract_module_header_files(module)

    # File form
    header_forms = dict(includes="%%include \"%s\"\n",
                        imports="%%%%import(module=\"dolfin.cpp.%s\") \"%%s\"\n" %\
                        combinedmodule)
    
    # Generate files
    files = {}
    for file_type, header_form in header_forms.items():

        # Create the file
        files[file_type] = open(os.path.join(module, file_type + ".i"), "w")
        files[file_type].write(copyright_statement%(_copyright_form_swig))
    
        files[file_type].write("// Auto generated %s statements for the "\
                               "module: %s\n\n"% (file_type[:-1], module))

        # Check if there is a foo/pre.i file
        if file_type == "includes":
            write_include_modifier(module, "pre")

        # Write include or import statement for each individual file
        for header in headers:
            files[file_type].write(header_form % header)
    
        # Check if there is a foo/post.i file
        if file_type == "includes":
            write_include_modifier(module, "post")

        files[file_type].close()

def generate_swig_include_files():

    # Scan typemap directory and generate an include file
    generate_typemap_includes()

    # Iterate over all combined modules
    for combined_module, modules in combined_modules.items():

        # Create a file being the root of the combined module
        combined_module_file = create_combined_module_file(combined_module)
    
        # Iterate over modules in each combined module and extract headers
        for module in modules:
            write_module_code(module, combined_module)

## Generate list of header files
#print "Generating file %s" % interface_file
#f = open(interface_file, "w")
#f.write("// Generated list of include files for PyDOLFIN\n")
#for (module, module_headers) in headers:
#    # Generate module imports
#    f_import = open(os.path.join("import", module + ".i"), "w")
#    f_import.write("// Auto generated import statements for the SWIG kernel module: '%s'\n\n"% module)
#    f.write("\n// DOLFIN headers included from %s\n" % module)
#    if os.path.isfile(module+"_pre.i"):
#        f.write("%%include \"dolfin/swig/%s_pre.i\"\n" % module)
#    for header in module_headers:
#        f.write("%%include \"%s\"\n" % header)
#        f_import.write('%%import(module="dolfin.cpp") "%s"\n'%header)
#    if os.path.isfile(module+"_post.i"):
#        f.write("%%include \"dolfin/swig/%s_post.i\"\n" % module)
#    f_import.close()
#f.close()
#
## Create docstrings.i file from docstrings module (only for dolfin.cpp)
#from documentation import generate_docstrings
#generate_docstrings()
#
## Extract all shared_ptr stored classes and store them in a pyton module
## and place that under dolfin.compilemodeuls.sharedptrclasses.py
#shared_ptr_classes = re.findall("%shared_ptr\(dolfin::(.+)\)", \
#                                open("shared_ptr_classes.i").read())
#
#shared_ptr_classes = filter(lambda x: "NAME" not in x, shared_ptr_classes)
#template = """
#'''
#This module contains the names of the classes in DOLFIN that is
#stored using shared_ptrs. The file is automatically generated by the
#generate.py script in the dolfin/swig directory.
#'''
#
#__all__ = ['shared_ptr_classes']
#
#shared_ptr_classes = %s
#"""
#
#par = os.path.pardir
#open(os.path.join(par, par, "site-packages", "dolfin", \
#                  "compilemodules", "sharedptrclasses.py"), "w").write(\
#    template%(repr(shared_ptr_classes)))
