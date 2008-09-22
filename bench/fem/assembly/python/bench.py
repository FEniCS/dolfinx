__author__ = "Ilmar Wilbers (ilmarw@simula.no)"
__date__ = "2008-06-04 -- 2008-07-21"
__copyright__ = "Copyright (C) 2008 Ilmar Wilbers"
__license__  = "GNU LGPL Version 2.1"

# Modified by Anders Logg, 2008.

# Be careful to first run the benchmark once in order for
# the Python extensions to be created, or else the timings
# will not be correct.

from dolfin import *
from time import time
import sys

def make_form(name, mesh):
    globals()['mesh'] = mesh
    execfile("../forms/" + name + '.form', globals())
    try:
        return a
    except:
        print "No object 'a' to return in file %s.form" %name
        return None

def bench_form(form, mesh, reps=1):
    totaltime = 0.0
    t0 = time()
    A = assemble(form, mesh)
    totaltime += time() - t0
    for i in range(reps - 1):
        t0 = time()
        assemble(form, mesh, tensor=A, reset_tensor=False)
        totaltime += time() - t0
    return totaltime / float(reps)

def make_mesh(name, dim):
    if dim == 3:
        N = 32
        mesh = UnitCube(N, N, N)
        return mesh
    else:
        N = 256
        mesh = UnitSquare(N, N)
        return mesh

if __name__ == "__main__":
    try:
        reps = int(sys.argv[1])
    except:
        print 'Usage: %s number_of_repetitions [1]' %sys.argv[0]
        reps = 1

# Backends
backends = ["uBLAS", "PETSc", "Epetra", "MTL4", "Assembly"]

# Forms
forms = ["Elasticity3D",
         "PoissonP1", 
         "PoissonP2", 
         "PoissonP3", 
         "THStokes2D",
         "NSEMomentum3D",
         "StabStokes2D"]

dolfin_set("output destination", "silent")
results = Table("Assembly benchmark")

for backend in backends:
    dolfin_set("linear algebra backend", backend)
    for form in forms:
        dim = 2 if not form.find("3D") > -1 else 3
        m = make_mesh(form, dim)
        a = make_form(form, m)
        print "Assembling %s with %s" % (form, backend)
        t = bench_form(a, m, reps=reps)
        results.set(backend, form, t)

dolfin_set("output destination", "terminal")
print ""

results.disp()
summary()
