__author__ = "Anders Logg (logg@simula.no)"
__date__ = "2007-08-15 -- 2007-08-16"
__copyright__ = "Copyright (C) 2007 Anders Logg"
__license__  = "GNU LGPL Version 2.1"

from ffc import *
from dolfin import *

# JIT assembler
def assemble(form, mesh, coefficients=None):
    "Assemble form over mesh and return tensor"
    
    # Compile form
    (compiled_form, module, form_data) = jit(form)

    # Extract coefficients
    coefficients = ArrayFunctionPtr()
    #for f in form_data[0].coefficients:
    #    coefficients.append(f)
    #print "Found coefficients: " + str(coefficients)

    # Create dummy arguments (not yet supported)
    coefficients = ArrayFunctionPtr()
    cell_domains = MeshFunction("uint")
    exterior_facet_domains = MeshFunction("uint")
    interior_facet_domains = MeshFunction("uint")

    # Assemble compiled form
    rank = compiled_form.rank()
    if rank == 0:
        s = Scalar()
        cpp_assemble(s, compiled_form, mesh, coefficients,
                       cell_domains, exterior_facet_domains, interior_facet_domains,
                       True)
        return s
    elif rank == 1:
        b = Vector()
        cpp_assemble(b, compiled_form, mesh, coefficients,
                       cell_domains, exterior_facet_domains, interior_facet_domains,
                       True)
        return b
    elif rank == 2:
        A = Matrix()
        cpp_assemble(A, compiled_form, mesh, coefficients,
                       cell_domains, exterior_facet_domains, interior_facet_domains,
                       True)
        return A
    else:
        raise RuntimeError, "Unable to assemble tensors of rank %d." % rank

# Rename FFC Function
ffc_Function = Function

# Create new class inheriting from both FFC and DOLFIN Function
class Function(ffc_Function, cpp_Function):

    def __init__(self, element, *others):
        ffc_Function.__init__(self, element)
        cpp_Function.__init__(self, *others)
