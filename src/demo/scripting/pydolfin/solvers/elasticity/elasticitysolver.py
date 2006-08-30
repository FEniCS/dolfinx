import datetime

from dolfin import *
from math import *

def save(counter, u, k, vtkfile):

    savefrequency = 33.0

    if(counter % int((1.0 / savefrequency / k)) == 0):
        vtkfile << u

class Source(Function):
    def eval(self, point, i):
        if(i == 1):
            return -2.0
        else:
            return 0.0

class InitialDisplacement(Function):
    def eval(self, point, i):
        return 0.0

class InitialVelocity(Function):
    def eval(self, point, i):
        if(i == 1 and point.x > 0.0):
            return 1.0
        else:
            return 0.0
        
class SimpleBC(BoundaryCondition):
    def eval(self, value, point, i):
        if point.x == 0.0:
            value.set(0.0)
        return value
    
f = Source()
u0 = InitialDisplacement()
v0 = InitialVelocity()

bc = SimpleBC()

mesh = Mesh("tetmesh-4.xml.gz")

E = 20.0 # Young's modulus
nu = 0.3 # Poisson's ratio

lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
mu = E / (2 * (1 + nu))

elast_forms = import_formfile("Elasticity.form")

# FIXME: Need to use Constant to use coefficients with FFC and PyDOLFIN

#a = elast_forms.ElasticityBilinearForm(lmbda, mu)
a = elast_forms.ElasticityBilinearForm()
L = elast_forms.ElasticityLinearForm(f)
Lu0 = elast_forms.ElasticityLinearForm(u0)
Lv0 = elast_forms.ElasticityLinearForm(v0)

mass_forms = import_formfile("ElasticityMass.form")

amass = mass_forms.ElasticityMassBilinearForm()

element = elast_forms.ElasticityBilinearFormTrialElement()

A = Matrix()
M = Matrix()
xu0 = Vector()
xv0 = Vector()
xu1 = Vector()
xv1 = Vector()
xu1old = Vector()
xv1old = Vector()
stepresidual = Vector()
m = Vector()
b = Vector()
xtmp1 = Vector()
xtmp2 = Vector()


t = 0.0  # time
T = 5.0  # final time
k = 0.01 # time step
counter = 0 # step counter

FEM_assemble(a, A, mesh)
FEM_assemble(L, b, mesh)

FEM_applyBC(A, b, mesh, element, bc)

#print "A: "
#A.disp(False)


FEM_assemble(amass, M, mesh)

# Lump mass matrix

FEM_lump(M, m)

# Assemble initial values

FEM_assemble(Lu0, xu1, mesh)
FEM_assemble(Lv0, xv1, mesh)

# Solve for initial values

xu1.div(m)
xv1.div(m)

# Initialize vectors

xtmp1.init(xu1.size())

# Solution functions
u = Function(xu1, mesh, element);
v = Function(xv1, mesh, element);

vtkfile = File("elasticity.pvd")

# Save
save(counter, u, k, vtkfile)

t1 = datetime.datetime.now()


while(t < T):
    
    # Make time step
    
    # Copy values from previous time step
    xu0.copy(xu1)
    xv0.copy(xv1)
    
    dolfin_log(False)
    # Assemble load vector
    FEM_assemble(L, b, mesh)
    
    # Set boundary conditions
    FEM_applyBC(A, b, mesh, element, bc)
    dolfin_log(True)
    
    # Fixed-point iteration
    for fpiter in range(0, 50):
        
        # Copy values from previous iteration
        xu1old.copy(xu1)
        xv1old.copy(xv1)
        
        # Evaluate right-hand side of ODE: u' = f(u)
        A.mult(xu1old, xtmp1);

        # Compute residual of time step equation (discrete residual)
        stepresidual.copy(b)
        stepresidual -= xtmp1
        stepresidual *= k
        stepresidual.div(m)
        stepresidual.axpy(1.0, xv0)
        stepresidual.axpy(-1.0, xv1)
            
        xv1.axpy(1, stepresidual)
            
        xu1.copy(xu0)
        xu1.axpy(k, xv1old)
        
        # Compute increments
        xtmp1.copy(xu1)
        xtmp1.axpy(-1, xu1old)
        
        xtmp2.copy(xv1)
        xtmp2.axpy(-1, xv1old)
        
#        print "Increments: ", xtmp1.norm(xtmp1.linf), \
#              " ", xtmp2.norm(xtmp1.linf)

        # Check convergence
        if(max(xtmp1.norm(xtmp1.linf), xtmp2.norm(xtmp2.linf)) < 1e-8):
            print "Fixed-point iteration converged. Iterations: ", fpiter
            break



    # Increment t and counter
    t += k
    counter += 1

    print "t: ", t
    save(counter, u, k, vtkfile)


t2 = datetime.datetime.now()

print "measured time: ", t2 - t1


# Plot with Mayavi

# Load mayavi
from mayavi import *

# Plot solution
v = mayavi()
d = v.open_vtk_xml("elasticity000000.vtu")

f = v.load_filter('WarpVector', config=0) 

m = v.load_module("BandedSurfaceMap")
m.actor.GetProperty().SetColor((0.8, 0.8, 1.0))

camera = v.renwin.camera
camera.Zoom(1.0)
camera.SetPosition(2.5, 1.5, 5.5)
camera.SetFocalPoint(1.0, 0.0, 0.0)
camera.SetRoll(0.0)

# Turn on sweeping
d.sweep_delay.set(0.01)
d.sweep_var.set(1)
d.do_sweep()

# Wait until window is closed
v.master.wait_window()
