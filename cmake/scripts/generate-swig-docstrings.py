"""Simple functions to update the docstrings.i file for the Python interface
from the intermediate representation of the documentation which is extracted
from the C++ source code of DOLFIN.

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
#
# First added:  2010-08-19
# Last changed: 2014-03-03

from __future__ import print_function
import os, shutil, types, sys

# Add path to dolfin_utils and import the documentation extractor.
doc_dir = os.path.abspath("site-packages")
sys.path.append(doc_dir)

from dolfin_utils.documentation import extract_doc_representation
from dolfin_utils.documentation import indent, add_links
from codeexamples import codesnippets

debug_output = False

def output(out):
    global debug_output
    if debug_output:
        print(out)

docstring = '%%feature("docstring")  %s "\n%s\n";\n\n'

# Dictionary for mapping C++ types to Python types.
# NOTE: KBO: The dictionary is not complete and is only tested for the Mesh.h class
cpp_to_python = {
"std::string": "str",
"string": "str",

"enum": "int",

"int": "int",
"unsigned int": "int",
"uint": "int",
"dolfin::uint": "int",
"std::size_t": "int",
"uint*": "numpy.array(uint)",
"dolfin::uint*": "numpy.array(uint)",
"std::size_t*": "numpy.array(uintp)",

"double": "float",
"double*": "numpy.array(float)",
"real": "float",
"dolfin::real": "float",

"bool": "bool",
}

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
#        print("cls: ", classname)
        # Iterate over class functions
        for (signature, comm) in function_documentation:
            # No need to put empty docstrings in the docstrings.i file!
            if comm is None:
                continue
#            print("sig: ", signature)
            name = get_function_name(signature)
            if not name in order:
                order.append(name)
#            print("name: '%s'" % name)
            if not name in func_doc:
                func_doc[name] = [(signature, comm)]
            else:
                func_doc[name].append((signature, comm))
        new_docs.append((classname, parent, comment, func_doc, order))

    return new_docs

def replace_example(text, classname, signature):
    """Replace the C++ example code with the Python equivalent.
    Currently we can only handle one block/section of example code per function.
    """
    # Check if we need to manipulate comment.
    if not "*Example*" in text:
        return text
    # Check if we have example code in the dictionary
    examplecode = ".. note::\n\n    No example code available for this function."
    if not classname in codesnippets:
        output(" "*6 + "No example code for class: '%s'" % classname)
    elif not signature in codesnippets[classname]:
        output(" "*6 + "No example code for (class, function): ('%s', '%s')" % (classname, signature))
    else:
        examplecode = codesnippets[classname][signature]

    # Remove leading and trailing new lines in example code.
    lines = examplecode.split("\n")
    while lines and not lines[0].strip():
        del lines[0]
    while lines and not lines[-1].strip():
        del lines[-1]
    examplecode = "\n".join(lines)

    # NOTE: KBO: We currently only handle 1 example block
    new_text = []
    example = False
    indentation = 0
    # Loop comment lines
    for l in text.split("\n"):
        # When we get to the lines containing the example, add the header and
        # codeblock.
        if not example and "*Example*" in l:
            example = True
            indentation = len(l) - len(l.lstrip())
            new_text.append(l)
            new_text += indent(examplecode, indentation + 4).split("\n")
        elif example and l.strip() and len(l) - len(l.lstrip()) <= indentation:
            example = False
            new_text.append(l)
        # Skip lines as long as we're inside the example block.
        elif example:
            continue
        else:
            new_text.append(l)
    return "\n".join(new_text)

def handle_std_pair(cpp_type, classnames):
    """Map std::pair to Python object."""

    args = cpp_type.split(">")[0].split("<")[1].split(",")
    if (len(args) != 2):
        output("No typemap handler implemented for %s" % cpp_type)
        return cpp_type

    arg1, arg2 = args
    arg1 = arg1.strip()
    arg2 = arg2.strip()

    if arg1 in cpp_to_python:
        arg1 = cpp_to_python[arg1]
        if not arg2 in cpp_to_python:
            output("No type map for '%s'!" % cpp_type)
            return cpp_type
        arg2 = cpp_to_python[arg2]
        return "(%s, %s)" % (arg1, arg2)

    elif arg1[0] == "_" and arg1[-1] == "_" and arg1[1:-1] in classnames:
        if not arg2 in cpp_to_python:
            output("No type map for '%s'!" % cpp_type)
            return cpp_type
        arg2 = cpp_to_python[arg2]
        return "Swig Object< std::pair<%s, %s> >" % (arg1, arg2)

    else:
        return None

def handle_std_vector(cpp_type, classnames):
    """Map std::vector to Python object (numpy.array)."""

    # Special case: vector of pairs
    if "std::pair" in cpp_type and not cpp_type.startswith("std::pair"):
        try:
            arg1, arg2 = cpp_type.split("<")[2].split(">")[0].split(",")
            pair = "std::pair<%s,%s>" % (arg1, arg2)
            if handle_std_pair(pair, classnames) is not None:
                return "numpy.array(%s)" % handle_std_pair(pair, classnames)
            else:
                return None
        except:
            # Failed to handle complex type, fail gracefully
            return None
    else:
        arg = cpp_type.split("<")[1].split(">")[0].strip()
        if not arg in cpp_to_python:
            if arg[0] == "_" and arg[-1] == "_" and arg[1:-1] in classnames:
                return "list of %s" % arg
            else:
                return None
        return "numpy.array(%s)" % cpp_to_python[arg]

# NOTE: KBO: This function is not complete and is only tested for the Mesh.h class
def map_cpp_type(cpp_type, classnames):
    "Map a C++ type to a Python type."

    if cpp_type in cpp_to_python:
        return cpp_to_python[cpp_type]

    # std::vector --> numpy.array or list
    elif "std::vector" in cpp_type:
        pobject = handle_std_vector(cpp_type, classnames)
        if pobject is not None:
            return pobject
        else:
            output("No type map for '%s'!" % cpp_type)
            return cpp_type

    # Special handling of std::pair
    elif "std::pair" in cpp_type:
        pobject = handle_std_pair(cpp_type, classnames)
        if pobject is not None:
            return pobject
        else:
            output("No type map for '%s'!" % cpp_type)
            return cpp_type

    # dolfin::Array --> numpy.array (primitives only)
    elif "_Array_" in cpp_type:
        arg = cpp_type.split("<")[1].split(">")[0].strip()
        if not arg in cpp_to_python:
            output("No type map for '%s'!" % arg)
            return "numpy.array(%s)" % arg
        return "numpy.array(%s)" % cpp_to_python[arg]

    # std::set --> set
    elif "std::set" in cpp_type:
        arg = cpp_type.split("<")[1].split(">")[0].strip()
        if not arg in cpp_to_python:
            output("No type map for '%s'!" % cpp_type)
            return cpp_type
        return "set of %s" % cpp_to_python[arg]

    # Handle links to classes defined in DOLFIN.
    elif cpp_type[0] == "_" and cpp_type[-1] == "_" and cpp_type[1:-1] in classnames:
        return cpp_type

    # Special case, e.g. cpp_type = boost::shared_ptr<_FunctionSpace_>
    elif "_" in cpp_type:
        args = cpp_type.split("_")
        for arg in args:
            if arg in classnames:
                return "_" + arg + "_"
        output("No type map for '%s'!" % cpp_type)

    else:
        output("No type map for '%s'!" % cpp_type)

    return cpp_type

def map_argument_and_return_types(text, classnames):
    """Map C++ types in the *Arguments* and *Returns* sections to corresponding
    Python types using a simple dictionary.

    Current implementation assumes the following format:

    *Returns*
         type
             description

    *Arguments*
         name0 (type)
             description
         name1 (type)
             description

    possibly separated with blank lines.
    """

    new_text = text

    # Could perhaps be handled more elegantly if we rely on the formatting?
    if "*Returns*" in new_text:
        # Get lines and find line number with *Returns*
        lines = new_text.split("\n")
        r_index = ["*Returns*" in l for l in lines].index(True)
        arg = False
        for e, l in enumerate(lines):
            if e > r_index and not arg:
                # First none blank line contains our argument
                if l.strip():
                    arg = True
                    indentation = len(l) - len(l.lstrip())
                    lines[e] = indent(map_cpp_type(l.strip(), classnames), indentation)
        new_text = "\n".join(lines)

    if "*Arguments*" in new_text:
        # Get lines and find line number with *Arguments*
        lines = new_text.split("\n")
        a_index = ["*Arguments*" in l for l in lines].index(True)
        a_indent = len(lines[a_index]) - len(lines[a_index].lstrip())
        n_indent = 0
        for e, l in enumerate(lines):
            if e > a_index and l.strip():
                indentation = len(l) - len(l.lstrip())
                # End of argument block
                if indentation <= a_indent:
                    break
                # Determine indentation of lines with argument names
                # first non blank line determines this
                if n_indent == 0:
                    n_indent = indentation
                # Get type of arguments defined in lines with name and type
                if indentation == n_indent:
                    n, t = l.split("(")
                    n = n.strip()
                    t = t.split(")")[0]
                    lines[e] = indent("%s (%s)" % (n, map_cpp_type(t.strip(), classnames)), n_indent)

        new_text = "\n".join(lines)

    return new_text

def modify_doc(text, classnames, classname, signature):
    "Add links, translate C++ to Python and change C++ types."

    # Replace C++ example code with Python example code
    text = replace_example(text, classname, signature)

    # Map C++ types to corresponding Python types
    text = map_argument_and_return_types(text, classnames)

    # Add links
    text = add_links(text, classnames, ":py:class:")

    # Escape '"' otherwise will SWIG complain
    text = text.replace('\"',r'\"')

    return text

def get_args(signature):
    "Get argument names (for Python) from signature."
#    print("sig: ", signature)
    arg_string = signature.split("(")[-1].split(")")[0]
#    print("arg_string: '%s'" % arg_string)
    args = []
    if arg_string:
        # This does not handle ',' inside type declaration,
        # e.g. std::pair<uint, uint>.
        # args = [a.split()[-1] for a in arg_string.split(",")]
        for a in arg_string.split(","):
            arg = a.split()[-1]
            # Assuming '::' is never in a name, but always
            # present when dealing with e.g. 'std::pair'
            # or boost::unordered_map.
            if not "::" in arg:
                args.append(arg)
#    print("args: '%s'" % args)
    return args


def write_docstrings(output_file, module, header, docs, classnames):
    """Write docstrings from a header file."""

    output_file.write("// Documentation extracted from: (module=%s, header=%s)\n" % (module, header))

    documentation = group_overloaded_functions(docs)
    for (classname, parent, comment, func_docs, order) in documentation:
        # Create class documentation (if any) and write.
        if classname is not None and comment is not None:
            cls_doc = modify_doc(comment, classnames, classname, classname)
            output_file.write(docstring % ("dolfin::%s" % classname, cls_doc))
        # Handle functions in the correct order (according to definition in the
        # header file).
        for name in order:
            func_name = "dolfin::%s::%s" % (classname, name)
            if classname is None:
                func_name = "dolfin::%s" % name

            functions = func_docs[name]
            if not functions:
                continue
            # We've got overloaded functions.
            if len(functions) > 1:
                func_doc = "**Overloaded versions**"
                for signature, doc in functions:
                    args = get_args(signature)
                    doc = "\n\n* %s\ (%s)\n\n" % (name, ", ".join(args)) +\
                          indent(doc, 2)
                    func_doc += modify_doc(doc, classnames, classname, signature)
                output_file.write(docstring % (func_name, func_doc))
            # Single function
            else:
                # Get function (only one)
                signature, func_doc = functions[0]
                func_doc = modify_doc(func_doc, classnames, classname, signature)
                output_file.write(docstring % (func_name, func_doc))

def generate_docstrings(top_destdir):
    """
    Generate docstring files for each module
    """

    from codesnippets import copyright_statement

    # Get top DOLFIN directory.
    script_rel_path = os.sep.join(__file__.split(os.sep)[:-1])
    script_rel_path = script_rel_path or "."
    dolfin_dir = os.path.abspath(os.path.join(script_rel_path, os.pardir, os.pardir))

    top_destdir = top_destdir or dolfin_dir

    abs_destdir = top_destdir if os.path.isabs(top_destdir) else os.path.join(dolfin_dir, top_destdir)

    if not os.path.isdir(abs_destdir):
        raise RuntimeError("%s is not a directory." % abs_destdir)

    # Set copyright form
    copyright_form_swig = dict(comment = r"//", holder="Kristian B. Oelgaard")

    # Extract documentation
    documentation, classnames = extract_doc_representation(dolfin_dir)

    output("Generating docstrings from intermediate representation module...")
    for module in documentation:
        if not os.path.isdir(os.path.join(abs_destdir, "dolfin", "swig", module)):
            os.mkdir(os.path.join(abs_destdir, "dolfin", "swig", module))
        output_file = open(os.path.join(abs_destdir, "dolfin", "swig", \
                                        module, "docstrings.i"), "w")
        output_file.write(copyright_statement%(copyright_form_swig))
        output_file.write("// Autogenerated docstrings file, extracted from the DOLFIN source C++ files.\n\n")

        output(" "*2 + module)
        for header, docs in documentation[module]:
            output(" "*4 + header)
            write_docstrings(output_file, module, header, docs, classnames)

        output_file.close()

if __name__ == "__main__":

    if len(sys.argv) not in [1,2,3]:
        raise RuntimeError("expected 0, 1 or 2 arguments")

    dest_dir = sys.argv[1] if len(sys.argv) > 2 else ""
    debug_output = len(sys.argv) > 2 and sys.argv[2] == "DEBUG"

    generate_docstrings(dest_dir)
