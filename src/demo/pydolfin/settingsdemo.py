from dolfin import *

tol = get("ODE tolerance")
print "tolerance:",
print tol
method = get("ODE method")
print "method:",
print method
order = get("ODE order")
print "order:",
print order
fixed = get("ODE fixed time step")
print "fixed time step:",
print fixed
print

set("ODE tolerance", 0.0001)
set("ODE method", "dg")
set("ODE order", 0)
set("ODE fixed time step", True)

tol = get("ODE tolerance")
print "tolerance:",
print tol
method = get("ODE method")
print "method:",
print method
order = get("ODE order")
print "order:",
print order
fixed = get("ODE fixed time step")
print "fixed time step:",
print fixed
print
