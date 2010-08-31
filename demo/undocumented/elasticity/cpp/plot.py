from dolfin import *

u = Function("elasticity.xml")
plot(u, mode="displacement", lutfile="VIF.lut")
