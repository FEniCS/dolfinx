"""This module provides functionality for form assembly in Python,
corresponding to the C++ assembly and PDE classes.

The C++ assemble function (renamed to cpp_assemble) is wrapped with
an additional preprocessing step where code is generated using the
FFC JIT compiler.

The C++ PDE classes are reimplemented in Python since the C++ classes
rely on the dolfin::Form class which is not used on the Python side."""

__author__ = "Anders Logg (logg@simula.no)"
__date__ = "2007-08-15 -- 2007-08-16"
__copyright__ = "Copyright (C) 2007 Anders Logg"
__license__  = "GNU LGPL Version 2.1"

from ffc import *
from dolfin import *

# JIT assembler
def assemble(form, mesh):
    "Assemble form over mesh and return tensor"
    
    # Compile form
    (compiled_form, module, form_data) = jit(form)

    # Extract coefficients
    coefficients = ArrayFunctionPtr()
    for c in form_data[0].coefficients:
        coefficients.push_back(c.f)

    # Create dummy arguments (not yet supported)
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
        "Create Function"

        # Element is given to constructor of FFC Function (if any)
        if isinstance(element, FiniteElement):
            ffc_Function.__init__(self, element)
            cpp_Function.__init__(self, *others)
        else:
            cpp_Function.__init__(self, *((element,) + others))

# LinearPDE class
class LinearPDE:
    """A LinearPDE represents a (system of) linear partial differential
    equation(s) in variational form: Find u in V such that
    
        a(v, u) = L(v) for all v in V',

    where a is a bilinear form and L is a linear form."""

    def __init__(self, a, L, mesh, bcs=[]):
        "Create LinearPDE"

        self.a = a
        self.L = L
        self.mesh = mesh
        self.bcs = bcs
        self.x = Vector()

        # Make sure we have a list
        if not isinstance(self.bcs, list):
            self.bcs = [self.bcs]

    def solve(self):
        "Solve PDE and return solution"

        begin("Solving linear PDE.");

        # Assemble linear system
        A = assemble(self.a, self.mesh)
        b = assemble(self.L, self.mesh)

        # FIXME: Maybe there is a better solution?
        # Compile form, needed to create discrete function
        (compiled_form, module, form_data) = jit(self.a)

        # Apply boundary conditions
        for bc in self.bcs:
            bc.apply(A, b, compiled_form)

        #message("Matrix:")
        #A.disp()

        #message("Vector:")
        #b.disp()

        # Choose linear solver
        solver_type = get("PDE linear solver")
        if solver_type == "direct":
            message("Using direct solver.")
            solver = LUSolver()
            #solver.set("parent", self)
        elif solve_type == "iterative":
            message("Using iterative solver (GMRES).")
            solver = KrylovSolver(gmres)
            #solver.set("parent", self)
        else:
            error("Unknown solver type \"%s\"." % solver_type)

        # Solver linear system
        solver.solve(A, self.x, b)

        #message("Solution vector:")
        #self.x.disp()
  
        # Create Function
        u = Function(self.mesh, self.x, compiled_form)

        end()

        return u
