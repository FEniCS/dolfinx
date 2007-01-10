import sys
import datetime
import Numeric
import LinearAlgebra
from dolfin import *
import transform

class DirectPDE(TimeDependentPDE):
    def __init__(self, mesh, f, E, nu, nuvval, bc, T, v0, rho):

        self.U = Function(Vector(), mesh)
        self.V = Function(Vector(), mesh)
        self.B = Function(Vector(), mesh)

        lmbdaval = E * nu / ((1 + nu) * (1 - 2 * nu))
        muval = E / (2 * (1 + nu))

        self.lmbda = Function(lmbdaval)
        self.mu = Function(muval)
        self.nuv = Function(nuvval)

        direct_forms = import_formfile("ElasticityDirect.form")
        #import elasticitydirect as direct_forms

        load_forms = import_formfile("ElasticityLoad.form")
        #import elasticityload as load_forms

        self.amass = direct_forms.ElasticityDirectBilinearForm(rho)

        self.Ldirect = direct_forms.ElasticityDirectLinearForm(self.B, self.V,
#                                                               self.lmbda,
                                                               self.mu,
                                                               self.nuv)

        Lload = load_forms.ElasticityLoadLinearForm(f)

        self.U.init(mesh, self.amass.trial())
        self.V.init(mesh, self.amass.trial())
        self.B.init(mesh, self.Ldirect.element(0))

        N = self.U.vector().size() + self.V.vector().size()

        TimeDependentPDE.__init__(self, self.amass, self.Ldirect,
                                  mesh, bc, N, T)

        self.v0 = v0
        
        self.E = E
        self.nu = nu
        
        self.solutionfile = File("solution.pvd")
        self.counter = 0
        self.filecounter = 0
        self.fcount = 0

        self.xu = self.U.vector()
        self.xv = self.V.vector()
        self.xB = self.B.vector()
    
        self.dotxu = Vector(self.xu.size())
        self.dotxv = Vector(self.xv.size())
        
        self.M = Matrix()
        self.m = Vector()
        
        self.bload = Vector()
        self.xtmp = Vector(self.xu.size())
    
        self.xF = Vector(self.xB.size())
        self.xF0 = Vector(self.xB.size())
        self.xF1 = Vector(self.xB.size())
        
        # Initialize and compute coefficients

        # Mass matrix
        FEM_assemble(self.a(), self.M, self.mesh())
        FEM_lump(self.M, self.m)

        print "m:"
        self.m.disp()

        # U
        ElasticityUpdatedSolver_initu0(self.xu, self.U.element(), self.mesh())

        # V
        ElasticityUpdatedSolver_finterpolate(self.v0, self.V, self.mesh())
        dolfin_log(False)
        FEM_applyBC(self.xv, self.mesh(), self.V.element(), self.bc())
        dolfin_log(True)

        #print "xv:"
        #self.xv.disp()

        # B
        ElasticityUpdatedSolver_initF0Euler(self.xF0, self.B.element(),
                                            self.mesh())
        ElasticityUpdatedSolver_computeFBEuler(self.xF, self.xB, self.xF0,
                                               self.xF1, self.B.element(),
                                               self.mesh())

        dolfin_log(False)
        FEM_assemble(Lload, self.bload, self.mesh())
        FEM_applyBC(self.bload, self.mesh(), self.a().trial(), self.bc())
        dolfin_log(True)

        #print "bload:"
        #self.bload.disp()

        # Initial values for ODE
        # Gather into x
        self.x.copy(self.xu, 0, 0, self.xu.size())
        self.x.copy(self.xv, self.xu.size(), 0, self.xv.size())

        print "xu:"
        self.xu.disp()

        print "xv:"
        self.xv.disp()

    def init(self, U):
        print "Python init"
        
    def preparestep(self):
        1

    def prepareiteration(self):
        1

    def fu(self, x, dotx, t):
        #print "Python fu"

        self.xu.copy(self.x, 0, 0, self.dotxu.size())
        self.xv.copy(self.x, 0, self.dotxu.size(), self.dotxv.size())

        # Mesh

        ElasticityUpdatedSolver_deform(self.mesh(), self.U)

        ElasticityUpdatedSolver_computeFBEuler(self.xF, self.xB,
                                               self.xF0, self.xF1,
                                               self.B.element(),
                                               self.mesh())


        # U
        
        self.dotxu.copy(self.xv, 0, 0, self.xv.size())

        # V

        dolfin_log(False)
        FEM_assemble(self.L(), self.dotxv, self.mesh())
        FEM_applyBC(self.dotxv, self.mesh(), self.V.element(), self.bc())
        dolfin_log(True)

        #print "dotxv: "
        #self.dotxv.disp()

        #print "mesh: "
        #self.mesh().disp()

        self.dotxv.axpy(1.0, self.xtmp)

        self.dotxv.axpy(1.0, self.bload)

        self.dotxv.div(self.m)

        # Gather into dotx
        self.dotx.copy(self.dotxu, 0, 0, self.dotxu.size())
        self.dotx.copy(self.dotxv, self.dotxu.size(), 0, self.dotxv.size())

        self.fcount += 1

    def save(self, U, t):

        if((self.counter % (1)) == 0):
            print "t: ", t
            self.solutionfile << U
            self.filecounter += 1

        self.counter += 1
