from dolfin import *

class HeatPDE(TimeDependentPDE):
    def __init__(self, mesh, f, bc, k, T, t):
        
        self.t = t

        self.U = Function(Vector(), mesh)

        forms = import_formfile("HeatTD.form")

        self.aheat = forms.HeatTDBilinearForm()
        self.Lheat = forms.HeatTDLinearForm(self.U, f)

        self.U.init(mesh, self.aheat.trial())

        N = self.U.vector().size()

        TimeDependentPDE.__init__(self, self.aheat, self.Lheat, mesh,
                                  bc, N, k, T)

        #self.x.copy(0.0)

        self.M = Matrix()
        self.m = Vector()

        FEM_assemble(self.a(), self.M, mesh)
        FEM_applyBC(self.M, self.mesh(), self.a().trial(), self.bc())

        FEM_lump(self.M, self.m)

        self.solutionfile = File("solution.pvd")
        self.sampleperiod = T / 100.0
        self.lastsample = 0.0

    def save(self, U, t):

        if(t == 0.0):
            self.U.vector().copy(self.x)
            self.solutionfile << U

        while(self.lastsample + self.sampleperiod < t):
            self.lastsample = min(t, self.lastsample + self.sampleperiod)
            self.U.vector().copy(self.x)
            self.solutionfile << U

    def fu(self, x, dotx, t):

        self.t.assign(t)

        self.U.vector().copy(self.x)

        #print "x: "
        #x.disp()

        dolfin_log(False)
        FEM_assemble(self.L(), self.dotx, self.mesh())
        FEM_applyBC(self.dotx, self.mesh(), self.a().trial(), self.bc())
        dolfin_log(True)

        dotx.div(self.m)
