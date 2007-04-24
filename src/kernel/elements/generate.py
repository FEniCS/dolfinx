"Generate finite elements for DOLFIN library of precompiled elements"

__author__ = "Anders Logg (logg@simula.no)"
__date__ = "2007-04-12 -- 2007-04-24"
__copyright__ = "Copyright (C) 2007 Anders Logg"
__license__  = "GNU LGPL Version 2.1"

from ffc import *

# Fancy import of list of elements from elements.py
from elements import __doc__ as elements
elements = [eval(element) for element in elements.split("\n")[1:-1]]

# Iterate over elements and compile
signatures = []
for element in elements:

    shape = "hej"
    degree = 1
    
    # Set name of element
    name = "%s_%s_%d" % (element.family().replace(" ", "_"),
                         shape_to_string[element.cell_shape()],
                         element.degree())
    
    # Generate code
    print "Compiling element: " + name
    compile(element, name)
    
    # Save signatures of elements and dof maps
    dof_map = DofMap(element)
    signatures += [(name, element.signature(), dof_map.signature())]
    
# Generate code for elementmap.cpp
filename = "ElementLibrary.cpp"
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
file.write("#include <dolfin/ElementLibrary.h>\n")
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
