from dolfin import *
from math import *

u = Function()
ufile = File("poisson.xml")
ufile >> u

vtkfile = File("output.pvd")
vtkfile << u

