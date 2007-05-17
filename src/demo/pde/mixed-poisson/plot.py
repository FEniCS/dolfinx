from dolfin import *

sigma = Function("sigma.xml")
plot(sigma)

u = Function("u.xml")
plot(u)
