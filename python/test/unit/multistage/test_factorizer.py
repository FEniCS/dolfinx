
from dolfin.multistage.factorize import extract_tested_expressions, ScalarFactorizer, map_expr_dag
from ufl import *

def test_scalar_factorizer():
    U = FiniteElement("CG", triangle, 1)
    V = VectorElement("CG", triangle, 1)

    v = TestFunction(V)

    expressions = [
        # Terminals and indexed terminals
        as_ufl(0.0),
        SpatialCoordinate(triangle)[0],
        Coefficient(U),
        Coefficient(V)[0],
        # Operators without arguments:
        SpatialCoordinate(triangle)[1] + as_ufl(1.0),
        cos(SpatialCoordinate(triangle)[1]),
        abs(Coefficient(U)),
        exp(Coefficient(V)[1]),
        Coefficient(U) * SpatialCoordinate(triangle)[1],
        Coefficient(U) / SpatialCoordinate(triangle)[1],
        Coefficient(U) + SpatialCoordinate(triangle)[1],
        # Test functions and indexed test functions:
        TestFunction(U),
        TestFunction(V)[0],
        TestFunction(V)[1],
        # Operators applied to arguments:
        2*TestFunction(U),
        2*TestFunction(V)[0],
        2*TestFunction(V)[1],
        TestFunction(U)/5,
        TestFunction(V)[0]/5,
        TestFunction(V)[1]/5,
        sin(SpatialCoordinate(triangle)[1])*TestFunction(U)/5,
        v[0]/4 + v[0]/5,
        v[0]/4 + v[1]/5,
        ]

    errors = [
        # Nonscalar
        SpatialCoordinate(triangle),
        TestFunction(V),
        # Adding test function and other expression
        TestFunction(U) + 1.0,
        # Multiplying test functions:
        TestFunction(U) * TestFunction(U),
        TestFunction(U) * TestFunction(V)[0],
        # Nonlinear operator applied to test function
        sin(TestFunction(U)),
        sin(TestFunction(V)[1]),
        ]

    from ufl.classes import Sum, Operator, Expr, Argument

    for expr in expressions:
        func = ScalarFactorizer()

        e = map_expr_dag(func, expr, compress=False)
        assert isinstance(e, dict) or expr == e

        #if isinstance(e, dict):
        #    print(str(expr), " = ", ",  ".join("%s[%d]: %s" % (func._arg, k, e[k]) for k in e))
        #else:
        #    print(str(expr), " = ", str(e))

    for expr in errors:
        func = ScalarFactorizer()
        try:
            e = map_expr_dag(func, expr, compress=False)
            ok = False
        except:
            ok = True
        assert ok
