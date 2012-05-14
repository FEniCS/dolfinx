#!/usr/bin/env python
#
# Generate SWIG files for Python interface of DOLFIN
#
# Copyright (C) 2012 Johan Hake
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
# First added:  2012-01-17
# Last changed: 2012-01-20

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
                  day = _local_time.tm_mday)

# Create form for copyright statement to a SWIG interface file
copyright_form_swig = dict(comment = r"//", holder="Johan Hake")

# FIXME: Removed date from copyright form 
#copyright_form_swig.update(_date_form)

# Extract original modules from dolfin.h
# NOTE: We need these, in particular the order
original_submodules = []
f = open("../dolfin.h")
for line in f:
    if "#include <dolfin/" in line and line[:17] == "#include <dolfin/":
        module = line.split("/")[1]
        original_submodules += [module]
f.close()

# Combined modules with sub modules
# NOTE: Changes in combined modules should be done here.
combined_modules = dict(common   = ["common", "parameter", "log"],
                        la       = ["la", "nls"],
                        mesh     = ["mesh", "intersection", "refinement", \
                                    "ale", "graph", "generation"],
                        function = ["function", "math"],
                        fem      = ["fem", "quadrature", "adaptivity"],
                        io       = ["io", "plot"])

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
        if module not in original_submodules:
            raise RuntimeError("Found a module: '%s' listed in the '%s' "\
                               "combined module, which is not part of the "\
                               "original DOLFIN modules." % \
                               (module, combined_module))

# Create a map from original modules to the combined
original_to_combined = {}
for submodule in original_submodules:
    for combined_module, submodules in combined_modules.items():
        if submodule in submodules:
            original_to_combined[submodule] = combined_module
            break

# List of headers to exclude (add more here)
excludes = ["plot.h", "IntersectionOperatorImplementation.h" ]

def create_combined_module_file(combined_module, submodules):
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
    for submodule_include in ["docstrings", "includes"]:
        includes = []
        for submodule in original_submodules:
            # Only include files from modules of the same combined module
            if submodule not in submodules:
                continue
            includes.append("%%include \"dolfin/swig/%s/%s.i\""%\
                            (submodule, submodule_include))
        combined_module_form[submodule_include] = "\n".join(includes)

    # Create includes for import of types from other modules
    # NOTE: Special case for common module. This should not be dependent
    # NOTE: on other modules. It also breaks as The base class Variable
    # NOTE: is needed by other modules and issues warnigs when these types
    # NOTE: are imported
    if combined_module != "common":
        submodule_imports = []
        for submodule in original_submodules:
            # Do not import files from modules of the same combined module
            if submodule in submodules:
                continue
            submodule_imports.append(\
                "%%include \"dolfin/swig/%s/local_imports.i\""%submodule)

        combined_module_form["module_imports"] = "\n".join(submodule_imports)
    else:
        combined_module_form["module_imports"] = ""

    # Write the generated code
    combined_module_file.write(combined_module_template % combined_module_form)

def extract_module_header_files(submodule):
    """
    Extract header files for a submodule
    """
    module_headers = []
    print("Processing dolfin_%s.h..." % submodule)
    f = open("../%s/dolfin_%s.h" % (submodule, submodule))
    for line in f:
        if re.search("^#include ",line):
            header = line.split()[1].replace("<", "").replace(">", "")

            # Get just the file name (after last /) and check against excludes:
            if not header.split("/")[-1] in excludes:
                module_headers.append(header)

    return module_headers

def write_module_code(submodule, combinedmodule):
    """
    Write SWIG module code.

    1) Write include.i file which consist of include of dolfin header files
    2) Write an import file to facilitate SWIG type import for each module

    """

    def write_include_modifier(submodule, modifier):
        """
        Write an include statements for pre or post modifier
        """
        if os.path.isfile(os.path.join(submodule, modifier + ".i")):
            interface_file = "dolfin/swig/%s/%s.i" % (submodule, modifier)
            files[file_type].write("%%include \"%s\"\n" % interface_file)

    # Get all headers in module
    headers = extract_module_header_files(submodule)

    # File form
    header_forms = dict(includes="%%include \"%s\"\n",
                        imports="%%%%import(module=\"dolfin.cpp.%s\") \"%%s\"\n" %\
                        combinedmodule,
                        local_imports="%%%%import(module=\"%s\") \"%%s\"\n" %\
                        combinedmodule)

    # Generate include and import files
    files = {}
    interface_files = []

    for file_type, header_form in header_forms.items():

        # Create the file
        files[file_type] = open(os.path.join(submodule, file_type + ".i"), "w")
        files[file_type].write(copyright_statement%(copyright_form_swig))

        files[file_type].write("// Auto generated %s statements for the "\
                               "module: %s\n\n"% (file_type[:-1], submodule))

        # Check if there is a foo/pre.i file
        write_include_modifier(submodule, "pre")

        # Write include or import statement for each individual file
        for header in headers:
            files[file_type].write(header_form % header)

        # Check if there is a foo/post.i file
        if file_type == "includes":
            write_include_modifier(submodule, "post")

        files[file_type].close()

        not_include = [os.path.join(submodule, "imports.i"),
                       ]

        # Add interface files
        interface_files = ["../../%s/"%submodule + \
                           interface_file.split(os.path.sep)[-1] \
                           for interface_file in \
                           glob.glob(os.path.join(submodule, "*.i")) \
                           if interface_file not in not_include]

    # Make the header files relative to where the combined module lives
    headers = [header.replace("dolfin", "../../..") for header in headers]

    return headers, interface_files

def generate_swig_include_files():
    """
    Generate all autogenerated SWIG files for PyDOLFIN
    """

    # Generate all docstrings
    # FIXME: Might be integratable into write_module_code?
    generate_docstrings()

    global_interface_files =  ["../../typemaps/" + \
                               interface_file.split(os.path.sep)[-1] \
                               for interface_file in \
                               glob.glob(os.path.join("typemaps","*.i"))]

    global_interface_files.extend(["../../shared_ptr_classes.i",
                                   "../../exceptions.i",
                                   "../../version.i",
                                   "../../forwarddeclarations.i"])

    # Collect header files and interface files for each submodule
    all_headers = []
    interface_files = {}
    for submodule in original_submodules:
        headers, interface_files[submodule] = \
            write_module_code(submodule, original_to_combined[submodule])
        all_headers.extend(headers)

    # Iterate over all combined modules
    for combined_module, submodules in combined_modules.items():

        # Collect interface files specific for this module
        module_interface_files = []

        # Iterate over all submodules and collect interface files
        for submodule in original_submodules:

            # If the submodule is included in the combined module
            if submodule in submodules:
                module_interface_files.extend(interface_files[submodule])

            # Else if we are gathering interface files for the common
            # modules we continue as the common module does not import
            # other submodules
            elif combined_module == "common":
                continue

            # Else just add the local_import.i and pre.i file
            else:
                module_interface_files.append(\
                    "../../%s/local_imports.i"%submodule)
                if os.path.isfile(os.path.join(submodule, "pre.i")):
                    module_interface_files.append(\
                        "../../%s/pre.i"%submodule)

        # Create a file being the root of the combined module
        create_combined_module_file(combined_module, submodules)

        # Add global interface files
        module_interface_files.extend(global_interface_files)

        # Add modules file
        module_interface_files.append("../../modules/%s/module.i" % \
                                      combined_module)

        # Generate CMake help file
        header_files_file = open(os.path.join("modules", combined_module,
                                              "headers.txt"), "w")
        header_files_file.write(";".join(sorted(all_headers)))

        interface_files_file = open(os.path.join("modules", combined_module,
                                                 "interface_files.txt"), "w")
        interface_files_file.write(";".join(sorted(module_interface_files)))


if __name__ == "__main__":
    generate_swig_include_files()
