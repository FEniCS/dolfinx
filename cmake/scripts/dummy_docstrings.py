"""
Generate SWIG docstrings when doxygen is not installed
 
SWIG generates different Python modules depending on SWIG docstrings
being present. See dolfin issue 

 https://bitbucket.org/fenics-project/dolfin/issues/834/
    swig-generates-different-code-when-having
    
This script is based on dolfin's previous docstring generation script
and uses the dolfin_utils C++ parser and produces dummy docstrings.
It can be removed when SWIG no longer has this strange behaviour.

We currently (spring 2017) use a SWIG docstring generation process based
on running the doxygen C++ documentation tool. This gives us a more
standard documentation pipeline and we also get warnings if docstrings
fall out of sync with the function signatures.

This file is a heavily feature restricted version of the old implementation
of generate-swig-doctrings.py which can last be found in dolfin commit
7f40b0ecf687e3c56ce0fec7c7594d8ea2476784 and can be seen on bitbucket:

 https://bitbucket.org/fenics-project/dolfin/src/
    7f40b0ecf687e3c56ce0fec7c7594d8ea2476784/cmake/scripts/
    generate-swig-docstrings.py?fileviewer=file-view-default

This script assumes that all functions and classes lives in the dolfin namespace.
"""

# Copyright (C) 2010 Kristian B. Oelgaard
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
# Modified by Johan Hake 2010
# Modified by Anders E. Johansen 2011
# Modified by Anders Logg 2014
# Modified by Chris Richardson 2016
# Modified by Tormod Landet 2017

from __future__ import print_function
import os, shutil, types, sys

# Add path to dolfin_utils and import the documentation extractor.
doc_dir = os.path.abspath("site-packages")
sys.path.append(doc_dir)

from dolfin_utils.documentation import extract_doc_representation

docstring = '%%feature("docstring")  %s "\nDummy docstring. Missing doxygen\n";\n\n'


def get_function_name(signature):
    "Extract function name from signature."
    words = signature.split("(")[0].split()
    # Special handling of operator since Swig needs 'operator double', not just
    # 'double', which is different from _normal_ operators like 'operator='
    if len(words) > 1 and words[-2] == "operator":
        return " ".join(words[-2:])
    return words[-1]


def group_overloaded_functions(docs):
    """Group functions with same name, but different signature.
    Assuming that overloaded functions in the dolfin namespace are defined
    in the same header file."""

    new_docs = []
    for (classname, parent, comment, function_documentation) in docs:
        func_doc = {}
        order = []
        # Iterate over class functions
        for (signature, comm) in function_documentation:
            # No need to put empty docstrings in the docstrings.i file!
            if comm is None:
                continue
            name = get_function_name(signature)
            if not name in order:
                order.append(name)
            if not name in func_doc:
                func_doc[name] = [(signature, comm)]
            else:
                func_doc[name].append((signature, comm))
        new_docs.append((classname, parent, comment, func_doc, order))

    return new_docs


def write_docstrings(output_file, module, header, docs, classnames):
    """Write docstrings from a header file."""

    output_file.write("// Documentation extracted from: (module=%s, header=%s)\n" % (module, header))

    documentation = group_overloaded_functions(docs)
    for (classname, parent, comment, func_docs, order) in documentation:
        # Create class documentation (if any) and write.
        if classname is not None and comment is not None:
            output_file.write(docstring % ("dolfin::%s" % classname))
        # Handle functions in the correct order (according to definition in the
        # header file).
        for name in order:
            func_name = "dolfin::%s::%s" % (classname, name)
            if classname is None:
                func_name = "dolfin::%s" % name

            functions = func_docs[name]
            if not functions:
                continue
            output_file.write(docstring % func_name)


def generate_dummy_docstrings(top_destdir):
    """
    Generate docstring files for each module
    """
    # Get top DOLFIN directory.
    script_rel_path = os.sep.join(__file__.split(os.sep)[:-1])
    script_rel_path = script_rel_path or "."
    dolfin_dir = os.path.abspath(os.path.join(script_rel_path, os.pardir, os.pardir))
    top_destdir = top_destdir or dolfin_dir
    abs_destdir = top_destdir if os.path.isabs(top_destdir) else os.path.join(dolfin_dir, top_destdir)

    if not os.path.isdir(abs_destdir):
        raise RuntimeError("%s is not a directory." % abs_destdir)

    # Extract documentation
    documentation, classnames = extract_doc_representation(dolfin_dir)

    print("Generating dummy docstrings...")
    for module in documentation:
        if not os.path.isdir(os.path.join(abs_destdir, "dolfin", "swig", module)):
            os.mkdir(os.path.join(abs_destdir, "dolfin", "swig", module))
        outpath = os.path.join(abs_destdir, "dolfin", "swig", module, "docstrings.i")
        output_file = open(outpath, "wt")
        output_file.write("// Dummy docstrings file, extracted from the DOLFIN source C++ files.\n\n")

        print("  Writing ", outpath)
        for header, docs in documentation[module]:
            write_docstrings(output_file, module, header, docs, classnames)

        output_file.close()


if __name__ == "__main__":
    dest_dir = sys.argv[1] if len(sys.argv) > 2 else ""
    generate_dummy_docstrings(dest_dir)
