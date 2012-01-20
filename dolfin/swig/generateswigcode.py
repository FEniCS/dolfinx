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
import sys

from codesnippets import *
from documentation import generate_docstrings

# Create time info for labeling generated code
_local_time = time.localtime()
_date_form = dict(year = _local_time.tm_year,
                  month = _local_time.tm_mon,
                  day = _local_time.tm_yday)

# Create form for copyright statement to a SWIG interface file
copyright_form_swig = dict(comment = r"//", holder="Johan Hake")
copyright_form_swig.update(_date_form)

# Extract original modules from dolfin.h
# NOTE: We need these, in particular the order
original_modules = []
f = open("../dolfin.h")
for line in f:
    if "#include <dolfin/" in line and line[:17] == "#include <dolfin/":
        module = line.split("/")[1]
        original_modules += [module]
f.close()

# Combined modules with sub modules
# NOTE: Changes in combined modules should be done here.
combined_modules = dict(common = ["common", "parameter", "log"],
                        la = ["la", "nls"],
                        mesh = ["mesh", "intersection", "refinement", "ale",\
                                "graph"],
                        function = ["function", "math"],
                        fem = ["fem", "quadrature", "adaptivity"],
                        io = ["io", "plot"])

# Check that the directory structure of the combined modules
# corresponds to the above dict
module_dirs = []
for module_dir in glob.glob("modules/*"):
    module_dirs.append(module_dir.split(os.path.sep)[-1])

# Some sanity checks 
for module_dir in module_dirs:
    if module_dir not in combined_modules:
        raise RuntimeError("Found a subdirectory: '%s' under the 'modules' "\
                           "directory, which is not listed as a combined "\
                           "module." % module_dir)

for combined_module, modules in combined_modules.items():
    if combined_module not in module_dirs:
        raise RuntimeError("Found a combined module: '%s' which is not a "\
                           "subdirectory under the 'modules' directory." % \
                           combined_module)
    for module in modules:
        if module not in original_modules:
            raise RuntimeError("Found a module: '%s' listed in the '%s' "\
                               "combined module, which is not part of the "\
                               "original DOLFIN modules." % \
                               (module, combined_module))

# Create a map from original modules to the combined
original_to_combined = {}
for module in original_modules:
    for combined_module, modules in combined_modules.items():
        if module in modules:
            original_to_combined[module] = combined_module
            break

# List of headers to exclude (add more here)
excludes = ["plot.h", "IntersectionOperatorImplementation.h" ]

def create_combined_module_file(combined_module, modules):
    """
    Create and initiate the main SWIG interface file for each
    comined module file
    """

    # Open file
    combined_module_file = open(os.path.join("modules", \
                                             combined_module, "module.i"), "w")

    combined_module_file.write(copyright_statement%(copyright_form_swig))

    combined_module_form = dict(
        module=combined_module,
        MODULE=combined_module.upper(),
        )

    # Create includes for header files and docstrings
    for module_include in ["docstrings", "includes"]:
        includes = []
        for module in original_modules:
            # Only include files from modules of the same combined module
            if module not in modules:
                continue
            includes.append("%%include \"dolfin/swig/%s/%s.i\""%\
                            (module, module_include))
        combined_module_form[module_include] = "\n".join(includes)

    # Create includes for import of types from other modules
    # NOTE: Special case for common module. This should not be dependent
    # NOTE: on other modules. It also breaks as The base class Variable
    # NOTE: is needed by other modules and issues warnigs when these types
    # NOTE: are imported
    if combined_module != "common":
        module_imports = []
        for module in original_modules:
            # Do not import files from modules of the same combined module
            if module in modules:
                continue
            module_imports.append("%%include \"dolfin/swig/%s/imports.i\""%module)

        combined_module_form["module_imports"] = "\n".join(module_imports)
    else:
        combined_module_form["module_imports"] = ""
    
    # Write the generated code
    combined_module_file.write(combined_module_template % combined_module_form)

def extract_module_header_files(module):
    """
    Extract header files for a module
    """
    module_headers = []
    print("Processing dolfin_%s.h..." % module)
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
            interface_file = "dolfin/swig/%s/%s.i" % (module, modifier)
            files[file_type].write("%%include \"%s\"\n" % interface_file)

    # Get all headers in module
    headers = extract_module_header_files(module)

    # File form
    header_forms = dict(includes="%%include \"%s\"\n",
                        imports="%%%%import(module=\"dolfin.cpp.%s\") \"%%s\"\n" %\
                        combinedmodule)
    
    # Generate include and import files
    files = {}
    interface_files = []
    
    for file_type, header_form in header_forms.items():

        # Create the file
        files[file_type] = open(os.path.join(module, file_type + ".i"), "w")
        files[file_type].write(copyright_statement%(copyright_form_swig))
    
        files[file_type].write("// Auto generated %s statements for the "\
                               "module: %s\n\n"% (file_type[:-1], module))

        # Check if there is a foo/pre.i file
        #if file_type == "includes":
        write_include_modifier(module, "pre")

        # Write include or import statement for each individual file
        for header in headers:
            files[file_type].write(header_form % header)
    
        # Check if there is a foo/post.i file
        if file_type == "includes":
            write_include_modifier(module, "post")

        files[file_type].close()

        not_include = [os.path.join(module, "includes.i"),
                       os.path.join(module, "modules.i")]

        # Add interface files
        interface_files = ["../../%s/"%module + \
                           interface_file.split(os.path.sep)[-1] \
                           for interface_file in \
                           glob.glob(os.path.join(module, "*.i")) \
                           if interface_file not in not_include]

    # Make the header files relative to where the combined module lives
    headers = [header.replace("dolfin", "../../..") for header in headers]

    return headers, interface_files

def generate_swig_include_files():
    """
    Generate all autogenerated SWIG files for PyDOLFIN
    """

    # Generate all docstrings
    # FIXME: Might be integratable into write_module_code
    #generate_docstrings()

    global_interface_files =  ["../../typemaps/" + \
                               interface_file.split(os.path.sep)[-1] \
                               for interface_file in \
                               glob.glob(os.path.join("typemaps","*.i"))]

    global_interface_files.extend(["../../shared_ptr_classes.i",
                                   "../../exceptions.i",
                                   "../../version.i",
                                   "../../forwarddeclarations.i"])

    # Iterate over all combined modules
    for combined_module, modules in combined_modules.items():

        # Create a file being the root of the combined module
        create_combined_module_file(combined_module, modules)
    
        all_headers = []
        all_interface_files = []

        # Iterate over modules in each combined module and extract headers
        for module in modules:
            headers, interface_files = write_module_code(module, combined_module)
            all_headers.extend(headers)
            all_interface_files.extend(interface_files)

        # Add global interface files
        all_interface_files.extend(global_interface_files)

        # Add modules file
        all_interface_files.append("../../modules/%s/module.i" % combined_module)

        # Generate CMake help file
        header_files_file = open(os.path.join("modules", combined_module,
                                              "headers.txt"), "w")
        header_files_file.write(";".join(all_headers))

        interface_files_file = open(os.path.join("modules", combined_module,
                                                 "interface_files.txt"), "w")
        interface_files_file.write(";".join(all_interface_files))
        

if __name__ == "__main__":
    generate_swig_include_files()
