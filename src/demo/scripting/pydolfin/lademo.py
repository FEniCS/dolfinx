from dolfin import *

x = Vector(10)
y = Vector(10)

x.copy(3)
y.copy(0)

y.axpy(3, x)

print "x:"
x.disp()

x[4] = 6

print "x:"
x.disp()

print "y:"
y.disp()

x.div(y)

print "x:"
x.disp()

