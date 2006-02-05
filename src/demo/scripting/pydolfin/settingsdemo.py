from dolfin import *

tol = get("tolerance")
print "tolerance:",
print tol
method = get("method")
print "method:",
print method
order = get("order")
print "order:",
print order
fixed = get("fixed time step")
print "fixed time step:",
print fixed
print

set("tolerance", 0.0001)
set("method", "dg")
set("order", 0)
set("fixed time step", True)

tol = get("tolerance")
print "tolerance:",
print tol
method = get("method")
print "method:",
print method
order = get("order")
print "order:",
print order
fixed = get("fixed time step")
print "fixed time step:",
print fixed
print
