from dolfin import *

# Plot velocity
u = Function("velocity.xml")
plot(u)

# Plot pressure
p = Function("pressure.xml")
plot(p)
