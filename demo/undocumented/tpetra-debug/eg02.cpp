
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

  UnitSquareMesh mesh(5, 5);
  Poisson::FunctionSpace V(mesh);
  Poisson::LinearForm L(V);
  Poisson::BilinearForm a(V, V);

  std::size_t rank = MPI::rank(mesh.mpi_comm());

  Source f;
  L.f = f;

  std::cout << mesh.num_vertices() << "\n";

  DirichletBoundary boundary;
  Constant u0(0.0);
  DirichletBC bc(V, u0, boundary);

  std::shared_ptr<Matrix> A(new Matrix);
  Vector b;
  assemble_system(*A, b, a, L, bc);

  Function u(V);
  //  A->init_vector(*u.vector(), 1);
  if (backend == "Tpetra")
  {
    as_type<TpetraVector>(*u.vector()).mapdump("x");
    as_type<TpetraVector>(b).mapdump("b");
  }

  KrylovSolver solver(sol, pc);
  solver.parameters["monitor_convergence"] = true;
  solver.set_operator(A);

  solver.solve(*u.vector(), b);

  // Terrible name
  // Create ghost values
  if (backend == "Tpetra")
    as_type<TpetraVector>(*u.vector()).update_ghost_values();

  File xdmf1("solve.xdmf");
  xdmf1 << u;

  //  plot(u);
  //  interactive();

  return 0;
}
