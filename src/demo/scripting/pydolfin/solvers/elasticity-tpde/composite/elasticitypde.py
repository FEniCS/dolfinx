from dolfin import *

class ElasticityPDE(TimeDependentPDE):
    def __init__(self, mesh, f, bc, T):
        
        self.U = Function(Vector(), mesh)
        #self.V = Function(Vector(), mesh)

        forms = import_formfile("Elasticity.form")
        #import elasticityform as forms

        self.aelast = forms.ElasticityBilinearForm()
        self.Lelast = forms.ElasticityLinearForm(self.U, f)

        self.U.init(mesh, self.aelast.trial())

        self.N = self.U.vector().size()

        TimeDependentPDE.__init__(self, self.aelast, self.Lelast, mesh,
                                  bc, self.N, T)

        self.xtmp = Vector(self.U.vector().size())

        # Initial values

        #self.U.interpolate(u0)
        #self.V.interpolate(v0)


        self.M = Matrix()
        self.m = Vector()

        FEM_assemble(self.a(), self.M, mesh)
        FEM_applyBC(self.M, self.mesh(), self.a().trial(), self.bc())

        FEM_lump(self.M, self.m)

        self.solutionfile = File("solution.pvd")
        self.sampleperiod = T / 100.0
        self.lastsample = 0.0

    def preparestep(self):
        1
        #print "step"

    def save(self, U, t):
        1

        U_0 = U[0]

        if(t == 0.0):
            U.vector().copy(self.x, 0, 0, self.U.vector().size())
            self.solutionfile << U_0

        while(self.lastsample + self.sampleperiod < t):
            self.lastsample = min(t, self.lastsample + self.sampleperiod)
            self.U.vector().copy(self.x, 0, 0, self.U.vector().size())
            self.solutionfile << U_0

    def fu(self, x, dotx, t):

        self.U.vector().copy(x, 0, 0, self.U.vector().size())

        dolfin_log(False)
        FEM_assemble(self.L(), self.xtmp, self.mesh())
        FEM_applyBC(self.xtmp, self.mesh(), self.a().trial(), self.bc())
        dolfin_log(True)

        self.xtmp.div(self.m)

        dotx.copy(self.xtmp, 0, 0, self.xtmp.size())

        #print "dotx: "
        #dotx.disp()
