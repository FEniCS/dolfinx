#!/usr/bin/env python
"""Simple functions to update the docstrings for the Python interface from the
FEniCS Documentation."""

__author__ = "Kristian B. Oelgaard (k.b.oelgaard@gmail.com)"
__date__ = "2010-08-19"
__copyright__ = "Copyright (C) 2010 Kristian B. Oelgaard"
__license__  = "GNU LGPL Version 2.1"

# Last changed: 2010-08-20

import os
import shutil
import types

if os.path.isfile("docstrings.i"):
    os.remove("docstrings.i")
output_file = open("docstrings.i", "a")
docstring = '%%feature("docstring")  %s "\n%s\n";\n\n'
# Might need more names here (operators??).
name_map = {"__init__" : lambda n: n}

def write_docstring(name, function, doc):
    """Write docstring for function. Assuming namespace 'dolfin' for all
    functions."""
    if name != "":
        output_file.write(docstring % ("::".join(["dolfin", name, function]), doc))
    else:
        output_file.write(docstring % ("::".join(["dolfin", function]), doc))

def handle_functions(mod, name):
    "Extract functions/methods from module or class."
    # Get all function types.
    functions = [v for k, v in mod.__dict__.items()\
                 if isinstance(v, (types.FunctionType, types.MethodType))]

    for func in functions:
        n = func.__name__
        if n in name_map:
            n = name_map[n](name)
        # Skip methods for now.
        if n[:2] == "__":
            continue
        write_docstring(name, n, func.__doc__)

# Simply dump docstrings from all classes and functions in the cpp module
# assuming that they are defined in the dolfin name space.
def generate_docstrings(docstrings):
    print "Generating docstrings.i from documentation module..."
    for key, val in docstrings.dolfin.cpp.__dict__.items():
        if isinstance(val, types.TypeType):
            # Write class docstring and handle member functions.
            write_docstring("", key, val.__doc__)
            handle_functions(val, key)

    # Write docstrings for all functions defined in the cpp module.
    handle_functions(docstrings.dolfin.cpp, "")
    output_file.close()

# Copy docstrings module from FEniCS Documentation to local dolfin/site-packages.
def copy_docstrings_module(docstrings):
    print "Copy docstrings module to dolfin/site-packages..."
    # Get absolute path to docstrings module.
    src = os.path.abspath(docstrings.__path__[0])
    # Create destination name and delete iff it exists.
    dst = os.path.join(os.pardir, os.pardir, "site-packages", "dolfin", "docstrings")
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    # Dump module in the site-packages (dolfin Python layer).
    shutil.copytree(src, dst)

