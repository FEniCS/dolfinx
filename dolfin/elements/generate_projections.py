"Generate pre-compiled L2 projections for DOLFIN"

__author__ = "Anders Logg (logg@simula.no), Johan Jansson (jjan@csc.kth.se)"
__date__ = "2008-03-18 -- 2008-10-23"
__copyright__ = "Copyright (C) 2008 Anders Logg, Johan Jansson"
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
    OPTIONS["language"] = "dolfin"
    
    # Generate code
    print "Compiling projection %d out of %d..." % (i, len(elements))
    element = elements[i]

    v = TestFunction(element)
    Pf = TrialFunction(element)
    f = Function(element)
    a = dot(Pf, v) * dx
    L = dot(f, v) * dx

    name = "ffc_L2proj_%.2d" % i
    compile([a, L], name, options=OPTIONS)

    # Save signatures of elements and dof maps
    dof_map = DofMap(element)
    signatures += [(name, element.signature(), dof_map.signature())]
    
# Generate code for projections
filename = "projection_library.inc"
print "Generating file " + filename
file = open(filename, "w")
file.write("// Automatically generated code mapping element signatures\n")
file.write("// to the corresponding Form classes representing projection\n")
file.write("\n")
file.write("#include <cstring>\n")
file.write("\n")
for (name, element_signature, dof_map_signature) in signatures:
    file.write("#include \"%s.h\"\n" % name)
file.write("\n")
file.write("#include \"ProjectionLibrary.h\"\n")
file.write("\n")
file.write("dolfin::Form* dolfin::ProjectionLibrary::create_projection_a(const char* signature)\n")
file.write("{\n")
for (name, element_signature, dof_map_signature) in signatures:
    file.write("  if (strcmp(signature, \"%s\") == 0)\n" % element_signature)
    file.write("    return new %sBilinearForm;\n" % name)
file.write("  return 0;\n")
file.write("}\n")
file.write("\n")
file.write("dolfin::Form* dolfin::ProjectionLibrary::create_projection_L(const char* signature, Function& f)\n")
file.write("{\n")
for (name, element_signature, dof_map_signature) in signatures:
    file.write("  if (strcmp(signature, \"%s\") == 0)\n" % element_signature)
    file.write("    return new %sLinearForm(f);\n" % name)
file.write("  return 0;\n")
file.write("}\n")
file.close()
