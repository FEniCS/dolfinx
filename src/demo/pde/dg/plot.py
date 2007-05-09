from dolfin import *

u = Function("poisson.xml")
plot(u)
