#include <dolfin.h>
#include "Poisson.h"

#include <dolfin/fem/PETScDMCollection.h>

#include <petscdmshell.h>
#include <petscksp.h>

using namespace dolfin;

// Source term (right-hand side)
class Source : public Expression
{
  void eval(Array<double>& values, const Array<double>& x) const
  {
    //double dx = x[0] - 0.5;
    //double dy = x[1] - 0.5;
    //values[0] = 10*exp(-(dx*dx + dy*dy) / 0.02);
    values[0] = 1.0;
  }
};

// Normal derivative (Neumann boundary condition)
class dUdN : public Expression
{
  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0] = sin(5*x[0]);
  }
};

// Sub domain for Dirichlet boundary condition
class DirichletBoundary : public SubDomain
{
  bool inside(const Array<double>& x, bool on_boundary) const
  {
    return on_boundary;
    //return x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS;
  }
};

int main()
{
  set_log_level(DEBUG);

  // Create meshes and function spaces
  auto mesh0 = std::make_shared<UnitSquareMesh>(16, 16);
  auto V0 = std::make_shared<Poisson::FunctionSpace>(mesh0);

  auto mesh1 = std::make_shared<UnitSquareMesh>(32, 32);
  auto V1 = std::make_shared<Poisson::FunctionSpace>(mesh1);

  auto mesh2 = std::make_shared<UnitSquareMesh>(64, 64);
  auto V2 = std::make_shared<Poisson::FunctionSpace>(mesh2);

  // Define boundary condition
  auto ubc = std::make_shared<Constant>(0.0);
  auto boundary = std::make_shared<DirichletBoundary>();
  auto bc = std::make_shared<DirichletBC>(V2, ubc, boundary);

  // Define variational forms
  Poisson::BilinearForm a(V2, V2);
  Poisson::LinearForm L(V2);
  auto f = std::make_shared<Source>();
  //auto g = std::make_shared<dUdN>();
  //auto g = std::make_shared<Constant>(0.0);
  //L.f = f;
  //L.g = g;

  // Compute solution
  //Function u(V);
  //solve(a == L, u, bc);

  PETScMatrix A;
  PETScVector b;
  assemble_system(A, b, a, L, {bc});

  PetscErrorCode ierr;

  KSP ksp;
  KSPCreate(MPI_COMM_WORLD, &ksp);
  KSPSetOperators(ksp, A.mat(), A.mat());
  KSPSetType(ksp, "preonly");

  PC pc;
  KSPGetPC(ksp, &pc);
  //PCSetType(pc, "lu");

  std::vector<std::shared_ptr<const FunctionSpace>> spaces = {V0, V1, V2};
  {
    PETScDMCollection dm_collection(spaces);

  DM dm = dm_collection.dm();

  KSPSetType(ksp, "richardson");
  PCSetType(pc, "mg");
  PCMGSetLevels(pc, 3, NULL);
  PCMGSetGalerkin(pc, PC_MG_GALERKIN_BOTH);
  PETScOptions::set("ksp_monitor_true_residual");
  //PETScOptions::set("mg_levels_ksp_monitor_true_residual");
  PETScOptions::set("ksp_atol", 1.0e-10);
  PETScOptions::set("ksp_rtol", 1.0e-10);
  KSPSetFromOptions(ksp);

  Function u(V2);
  PETScVector& x = u.vector()->down_cast<PETScVector>();
  KSPSetDM(ksp, dm);
  KSPSetDMActive(ksp, PETSC_FALSE);
  ierr = KSPSolve(ksp, b.vec(), x.vec());CHKERRQ(ierr);

  std::cout << "Soln vector norm: " << x.norm("l2") << std::endl;

  //KSPView(ksp, PETSC_VIEWER_STDOUT_SELF);
  }

  KSPDestroy(&ksp);

  return 0;
}
