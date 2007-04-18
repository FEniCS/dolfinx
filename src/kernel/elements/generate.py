"Generate finite elements for DOLFIN library of precompiled elements"

__author__ = "Anders Logg (logg@simula.no)"
__date__ = "2007-04-12 -- 2007-04-13"
__copyright__ = "Copyright (C) 2007 Anders Logg"
__license__  = "GNU GPL Version 2"

from ffc import *

# Elements to generate
elements = [("Lagrange", (1, 3)), ("Discontinuous Lagrange", (0, 3))]
shapes = ["triangle", "tetrahedron"]

# Iterate over elements and compile
signatures = []
for (family, (qmin, qmax)) in elements:
    for shape in shapes:
        for degree in range(qmin, qmax + 1):

            # Create element
            element = FiniteElement(family, shape, degree)

            # Set name of element
            name = "%s_%s_%d" % (family.replace(" ", "_"), shape, degree)

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
