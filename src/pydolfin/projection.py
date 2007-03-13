def projection(K, name):

    from import_form import *
    from ffc.compiler.compiler import *

    # Construct projection form in FFC representation

    v = TestFunction(K)
    U = TrialFunction(K)

    f = Function(K)

    a = dot(v, U) * dx
    L = dot(v, f) * dx

    # Compile form and return as module

    form = import_form([a, L, None], name)
    
    # Indicate memory ownership
    form.thisown = False

    return form

def project(f, K, mesh):

    from dolfin import *
    
    Pforms = projection(K, "Projection")

    a = Pforms.ProjectionBilinearForm()
    L = Pforms.ProjectionLinearForm(f)

    # Assemble linear system
    M = Matrix()
    b = Vector()
    FEM_assemble(a, L, M, b, mesh)

    x = Vector(b.size())

    # Solve linear system
    solver = KrylovSolver()
    solver.solve(M, x, b)

    # Lump
    #m = Vector()
    #FEM_lump(M, m)

    #x.copy(b, 0, 0, x.size())
    #x.div(m)

    # Define a function from computed degrees of freedom
    Pf = Function(x, mesh, a.trial())

    # Indicate memory ownership
    Pf.thisown = False
    Pforms.thisown = False
    a.thisown = False
    L.thisown = False
    x.thisown = False

    return Pf
