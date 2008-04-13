"Generate finite elements for DOLFIN library of precompiled elements"

__author__ = "Anders Logg (logg@simula.no)"
__date__ = "2007-04-12 -- 2008-04-13"
__copyright__ = "Copyright (C) 2007-2008 Anders Logg"
__license__  = "GNU LGPL Version 2.1"

from ffc import *
from ffc.common.constants import FFC_OPTIONS

# Fancy import of list of elements from elements.py
from elements import __doc__ as elements
elements = [eval(element) for element in elements.split("\n")[1:-1]]

# Iterate over elements and compile
signatures = []
for i in range(len(elements)):
    
    # Don't generate all functions
    OPTIONS = FFC_OPTIONS.copy()
    OPTIONS["no-evaluate_basis"] = True
    OPTIONS["no-evaluate_basis_derivatives"] = True

    # Generate code
    print "Compiling element %d out of %d..." % (i, len(elements))
    element = elements[i]
    name = "ffc_%.2d" % i
    compile(element, name, options=OPTIONS)

    # Save signatures of elements and dof maps
    dof_map = DofMap(element)
    signatures += [(name, element.signature(), dof_map.signature())]
    
# Generate code for elementmap.cpp
filename = "element_library.inc"
print "Generating file " + filename
file = open(filename, "w")
file.write("// Automatically generated code mapping element and dof map signatures\n")
file.write("// to the corresponding ufc::finite_element and ufc::dof_map classes\n")
file.write("\n")
file.write("#include <cstring>\n")
file.write("\n")
for (name, element_signature, dof_map_signature) in signatures:
    file.write("#include \"%s.h\"\n" % name)
file.write("\n")
file.write("#include \"ElementLibrary.h\"\n")
file.write("\n")
file.write("ufc::finite_element* dolfin::ElementLibrary::create_finite_element(const char* signature)\n")
file.write("{\n")
for (name, element_signature, dof_map_signature) in signatures:
    file.write("  if (strcmp(signature, \"%s\") == 0)\n" % element_signature)
    file.write("    return new %s_finite_element_0();\n" % name)
file.write("  return 0;\n")
file.write("}\n")
file.write("\n")
file.write("ufc::dof_map* dolfin::ElementLibrary::create_dof_map(const char* signature)\n")
file.write("{\n")
for (name, element_signature, dof_map_signature) in signatures:
    file.write("  if (strcmp(signature, \"%s\") == 0)\n" % dof_map_signature)
    file.write("    return new %s_dof_map_0();\n" % name)
file.write("  return 0;\n")
file.write("}\n")
file.close()
