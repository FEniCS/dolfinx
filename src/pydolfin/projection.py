def projection(Kdolfin, name):

    from import_form import *
    from ffc.compiler.compiler import *

    # Construct projection form in FFC representation

    s = Kdolfin.spec()

    K = FiniteElement(s.type(), s.shape(), s.degree(), s.vectordim())
    
    v = TestFunction(K)
    U = TrialFunction(K)

    f = Function(K)

    a = dot(v, U) * dx
    L = dot(v, f) * dx

    # Compile form and return as module

    form = import_form([a, L, None], name)
    
    return form
