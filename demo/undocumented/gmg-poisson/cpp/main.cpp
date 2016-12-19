#include <dolfin.h>
#include "Poisson.h"
#include "interpolation.h"

#include <petscdmshell.h>

using namespace dolfin;

// Source term (right-hand side)
class Source : public Expression
{
  void eval(Array<double>& values, const Array<double>& x) const
  {
    double dx = x[0] - 0.5;
    double dy = x[1] - 0.5;
    values[0] = 10*exp(-(dx*dx + dy*dy) / 0.02);
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
    return x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS;
  }
};

PetscErrorCode create_interpolation(DM dmc, DM dmf, Mat *mat, Vec *vec)
{
  std::shared_ptr<FunctionSpace> *Vc, *Vf;
  DMShellGetContext(dmc, (void**)&Vc);
  DMShellGetContext(dmf, (void**)&Vf);

  std::shared_ptr<PETScMatrix> P = create_transfer_matrix(*Vc, *Vf);

  *mat = P->mat();
  *vec = NULL;

  PetscObjectReference((PetscObject)P->mat());

  return 0;
}



int main()
{
  // Create mesh and function space
  auto mesh = std::make_shared<UnitSquareMesh>(32, 32);
  auto V = std::make_shared<Poisson::FunctionSpace>(mesh);

  // Define boundary condition
  auto ubc = std::make_shared<Constant>(0.0);
  auto boundary = std::make_shared<DirichletBoundary>();
  auto bc = std::make_shared<DirichletBC>(V, ubc, boundary);

  // Define variational forms
  Poisson::BilinearForm a(V, V);
  Poisson::LinearForm L(V);
  auto f = std::make_shared<Source>();
  auto g = std::make_shared<dUdN>();
  L.f = f;
  L.g = g;

  // Compute solution
  Function u(V);
  //solve(a == L, u, bc);

  PETScMatrix A;
  PETScVector b;
  assemble_system(A, b, a, L, {bc});

  KSP ksp;
  KSPCreate(MPI_COMM_WORLD, &ksp);
  KSPSetOperators(ksp, A.mat(), A.mat());
  KSPSetType(ksp, "preonly");

  PC pc;
  KSPGetPC(ksp, &pc);
  PCSetType(pc, "lu");

  PETScVector& x = u.vector()->down_cast<PETScVector>();
  //KSPSolve(ksp, b.vec(), x.vec());

  // Gine grid
  DM dm1;
  DMShellCreate(MPI_COMM_WORLD, &dm1);
  DMShellSetGlobalVector(dm1, x.vec());
  DMShellSetContext(dm1, (void*)&V);

  // Coarse grid
  auto mesh0 = std::make_shared<UnitSquareMesh>(16, 16);
  auto V0 = std::make_shared<Poisson::FunctionSpace>(mesh0);
  Function u0(V0);
  PETScVector& x0 = u0.vector()->down_cast<PETScVector>();

  DM dm0;
  DMShellCreate(MPI_COMM_WORLD, &dm0);
  DMShellSetGlobalVector(dm0, x0.vec());
  DMShellSetContext(dm0, (void*)&V0);

  // Set grids
  DMSetCoarseDM(dm1, dm0);
  DMSetFineDM(dm0, dm1);

  // Set interpolation matrix
  DMShellSetCreateInterpolation(dm1, create_interpolation);
  DMShellSetCreateInterpolation(dm0, create_interpolation);

  KSPSetType(ksp, "richardson");
  PCSetType(pc, "mg");
  PCMGSetGalerkin(pc, PC_MG_GALERKIN_BOTH);

  KSPSolve(ksp, b.vec(), x.vec());

  return 0;
}
