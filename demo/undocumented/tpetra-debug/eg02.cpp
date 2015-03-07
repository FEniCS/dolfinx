
#include <dolfin.h>
#include "Poisson.h"

using namespace dolfin;

// Sub domain for Dirichlet boundary condition
class DirichletBoundary : public SubDomain
{
  bool inside(const Array<double>& x, bool on_boundary) const
  {
    return x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS;
  }
};

// Source term (right-hand side)
class Source : public Expression
{
  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0] = x[0];
  }
};

int main(int argc, char *argv[])
{
  if (argc != 4)
    error("Usage %s backend solver preconditioner\n", argv[0]);

  const std::string backend = argv[1];
  const std::string sol = argv[2];
  const std::string pc = argv[3];
  parameters["linear_algebra_backend"] = backend;

  unsigned int n = 400;
  UnitSquareMesh mesh(n, n);
  Poisson::FunctionSpace V(mesh);
  Poisson::LinearForm L(V);
  Poisson::BilinearForm a(V, V);

  std::size_t rank = dolfin::MPI::rank(mesh.mpi_comm());

  Source f;
  L.f = f;

  std::cout << mesh.num_vertices() << "\n";

  DirichletBoundary boundary;
  Constant u0(0.0);
  DirichletBC bc(V, u0, boundary);

  std::shared_ptr<Matrix> A(new Matrix);
  Vector b;
  assemble_system(*A, b, a, L, bc);

  list_krylov_solver_methods();
  list_krylov_solver_preconditioners();

  Function u(V);

  if (backend == "Tpetra" and n < 10)
  {
    TpetraVector& bt = as_type<TpetraVector>(b);
    TpetraVector& ut = as_type<TpetraVector>(*u.vector());

    ut.mapdump("x");
    bt.mapdump("b");

    TpetraVector::mapdump(bt.vec()->getMap(), "bMap");
    TpetraVector::mapdump(ut.vec()->getMap(), "xMap");

    TpetraMatrix& At = as_type<TpetraMatrix>(*A);
    TpetraVector::mapdump(At.mat()->getRowMap(), "Arow");
    TpetraVector::mapdump(At.mat()->getRangeMap(), "Arange");
    TpetraVector::mapdump(At.mat()->getDomainMap(), "Adomain");
    TpetraVector::mapdump(At.mat()->getColMap(), "Acol");
  }

  KrylovSolver solver(sol, pc);

  std::cout << solver.parameters("preconditioner").str(true);

  solver.parameters["monitor_convergence"] = true;
  solver.set_operator(A);

  solver.solve(*u.vector(), b);

  File xdmf1("solve.xdmf");
  xdmf1 << u;

  list_timings();

  return 0;
}
