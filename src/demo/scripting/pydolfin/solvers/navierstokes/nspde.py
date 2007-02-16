from dolfin import *

class NSPDE(TimeDependentPDE):
    def __init__(self, mesh, f, nuval, d1val, d2val, vbc, pbc, T):
        
        self.pbc = pbc

        self.U = Function(Vector(), mesh)
        self.P = Function(Vector(), mesh)

        self.nuval = nuval
        self.nu = Function(nuval)

        self.d1 = Function(Vector(), mesh)
        self.d2 = Function(Vector(), mesh)

        forms = import_formfile("NavierStokes.form")

        pforms = import_formfile("NavierStokesPressure.form")

        self.ans = forms.NavierStokesBilinearForm()
        self.Lns = forms.NavierStokesLinearForm(self.U, self.P, f,
                                                self.nu, self.d1, self.d2)

        self.ap = pforms.NavierStokesPressureBilinearForm(self.d1)
        self.Lp = pforms.NavierStokesPressureLinearForm(self.U)

        self.U.init(mesh, self.ans.trial())
        self.P.init(mesh, self.ap.trial())
        self.d1.init(mesh, self.Lns.element(2))
        self.d2.init(mesh, self.Lns.element(2))

        self.N = self.U.vector().size()

        TimeDependentPDE.__init__(self, self.ans, self.Lns, mesh,
                                  vbc, self.N, T)

        self.U.attach(self.x)

        self.xtmp = Vector(self.U.vector().size())

        self.Ap = Matrix()
        self.bp = Vector()

        self.xp = self.P.vector()

        self.linsolver = KrylovSolver()

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
        print "t: ", t
        print "x: "
        self.x.disp()
        print "p: "
        self.P.vector().disp()

        if(t == 0.0):
            self.solutionfile << U

        while(self.lastsample + self.sampleperiod < t):
            self.lastsample = min(t, self.lastsample + self.sampleperiod)
            self.solutionfile << U

    def u0(self, x):

        print "x0: "
        x.disp()

    def fu(self, x, dotx, t):

        FEM_applyBC(self.x, self.mesh(), self.a().trial(), self.bc())

        

        

        #print "self.x: "
        #self.x.disp()
        #print "x: "
        #x.disp()
        NSESolver_ComputeStabilization(self.mesh(), self.U, self.nuval, 1.0e-3,
                                       self.d1.vector(), self.d2.vector())

        # P

        dolfin_log(False)
        FEM_assemble(self.ap, self.Ap, self.mesh())
        FEM_assemble(self.Lp, self.bp, self.mesh())
        FEM_applyBC(self.Ap, self.mesh(), self.P.element(), self.pbc)
        FEM_applyBC(self.bp, self.mesh(), self.P.element(), self.pbc)
        dolfin_log(True)

        self.linsolver.solve(self.Ap, self.P.vector(), self.bp)

        # U

        dolfin_log(False)
        FEM_assemble(self.L(), self.xtmp, self.mesh())
        FEM_applyBC(self.xtmp, self.mesh(), self.a().trial(), self.bc())
        dolfin_log(True)

        self.xtmp.div(self.m)

        dotx.copy(self.xtmp, 0, 0, self.xtmp.size())

