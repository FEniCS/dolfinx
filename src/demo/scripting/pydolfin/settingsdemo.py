from dolfin import *

tol = dolfin_get("tolerance")
print "tolerance:",
print tol
method = dolfin_get("method")
print "method:",
print method
order = dolfin_get("order")
print "order:",
print order
fixed = dolfin_get("fixed time step")
print "fixed time step:",
print fixed
print

dolfin_set("tolerance", 0.0001)
dolfin_set("method", "dg")
dolfin_set("order", 0)
dolfin_set("fixed time step", True)

tol = dolfin_get("tolerance")
print "tolerance:",
print tol
method = dolfin_get("method")
print "method:",
print method
order = dolfin_get("order")
print "order:",
print order
fixed = dolfin_get("fixed time step")
print "fixed time step:",
print fixed
print
