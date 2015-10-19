#include <cmath>
#include <dolfin.h>
#include "MultiMeshPoisson.h"
#include "MultiMeshL2Norm.h"

using namespace dolfin;

// Exact solution
class ExactSolution: public Expression
{
  void eval(Array<double>& v, const Array<double>& x) const
  {
    v[0] = sin(DOLFIN_PI*x[0])*sin(DOLFIN_PI*x[1]);
  }
};

// Source term
class Source : public Expression
{
  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0] = 2*DOLFIN_PI*DOLFIN_PI*sin(DOLFIN_PI*x[0])*sin(DOLFIN_PI*x[1]);
  }
};

// Sub domain for Dirichlet boundary condition
class DirichletBoundary : public SubDomain
{
  bool inside(const Array<double>& x, bool on_boundary) const
  {
    return on_boundary and (x[0] < DOLFIN_EPS or x[0] > 1-DOLFIN_EPS or
			    x[1] < DOLFIN_EPS or x[1] > 1-DOLFIN_EPS);
  }
};

double MultiMeshL2Error(const MultiMeshFunctionSpace& V,
			const MultiMeshFunction& uh)
{
  std::cout << "Compute L2 error" << std::endl;

  ExactSolution u;
  MultiMeshForm mmf(V);

  for (std::size_t i = 0; i < V.num_parts(); ++i)
  {
    std::shared_ptr<MultiMeshL2Norm::Functional> M(new MultiMeshL2Norm::Functional(V.multimesh()->part(i)));
    M->uh = *uh.part(i);
    M->u = u;

    mmf.add(M);
  }
  mmf.build();

  MultiMeshAssembler mma;
  Scalar m;
  mma.assemble(m, mmf);
  return std::sqrt(m.get_scalar_value());
}

std::string pvd_filename(const std::string& basename,
			       std::size_t part,
			       double h)
{
  std::stringstream ss;
  ss << basename << "-part" << part << "-h" << h;
}


void solve_random_meshes(std::size_t Nmeshes,
			 double h,
			 std::string& filename,
			 bool do_plot,
			 bool do_errors,
			 double& L2_error)
{
  std::cout << "Solve Poisson on " << Nmeshes << " random meshes on top of background mesh" << std::endl;

  const std::size_t N = static_cast<std::size_t>(std::round(1. / h));

  MultiMeshFunctionSpace V;
  V.parameters("multimesh")["quadrature_order"] = 2;

  // Background mesh
  UnitSquareMesh usm(N, N);
  MultiMeshPoisson::FunctionSpace V_part(usm);
  V.add(V_part);

  const std::size_t mn = 1;

  // Build function space
  for (std::size_t i = 0; i < Nmeshes; ++i)
  {
    // Create random rectangle mesh
    double x0 = dolfin::rand();
    double x1 = dolfin::rand();
    if (x0 > x1) std::swap(x0, x1);
    double y0 = dolfin::rand();
    double y1 = dolfin::rand();
    if (y0 > y1) std::swap(y0, y1);

    const std::size_t m = static_cast<std::size_t>(std::round((x1-x0) / h));
    const std::size_t n = static_cast<std::size_t>(std::round((y1-y0) / h));

    std::shared_ptr<Mesh> rm(new RectangleMesh(Point(x0, y0), Point(x1, y1),
					       std::max(m, mn), std::max(n, mn)));

    // // Rotate
    // const double v = dolfin::rand()*90;
    // rm->rotate(v);

    std::shared_ptr<MultiMeshPoisson::FunctionSpace> V_part(new MultiMeshPoisson::FunctionSpace(rm));

    V.add(V_part);
  }
  V.build();

  MultiMeshForm a(V, V);
  MultiMeshForm L(V);
  Source f;

  // Build forms
  for (std::size_t i = 0; i < V.num_parts(); ++i)
  {
    std::shared_ptr<const FunctionSpace> V_part = V.part(i);
    std::shared_ptr<MultiMeshPoisson::BilinearForm> a_part(new MultiMeshPoisson::BilinearForm(V_part, V_part));
    std::shared_ptr<MultiMeshPoisson::LinearForm> L_part(new MultiMeshPoisson::LinearForm(V_part));
    L_part->f = f;
    a.add(a_part);
    L.add(L_part);
  }
  a.build();
  L.build();

  // BC
  Constant zero(0);
  DirichletBoundary bdry;
  MultiMeshDirichletBC bc(V, zero, bdry);

  // Assemble
  Matrix A;
  Vector b;
  MultiMeshAssembler mma;
  mma.assemble(A, a);
  mma.assemble(b, L);
  bc.apply(A, b);

  // Solve
  MultiMeshFunction uh(V);
  solve(A, *uh.vector(), b);

  // Save
  for (std::size_t i = 0; i < V.num_parts(); ++i)
  {
    std::stringstream ss;
    ss << i;
    File(filename + ss.str() + ".pvd") << *uh.part(i);
  }

  // Plot
  if (do_plot)
  {
    for (std::size_t i = 0; i < V.num_parts(); ++i)
      plot(uh.part(i));
    plot(V.multimesh());
    interactive();
  }

  // Error
  if (do_errors)
  {
    L2_error = MultiMeshL2Error(V, uh);
    //H1_error = MultiMeshLH1Error(V, uh);
  }
}


int main(int argc, char* argv[])
{
  if (dolfin::MPI::size(MPI_COMM_WORLD) > 1)
  {
    info("Sorry, this demo does not (yet) run in parallel.");
    return 0;
  }

  // DOLFIN parameters
  parameters.parse(argc, argv);
  parameters["reorder_dofs_serial"] = false;

  // Application parameters
  Parameters p("my_own_params");

  // Debug or not
  p.add("debug", false);

  // Set number of random meshes on top of background unit square mesh
  p.add("Nmeshes", 1);

  // Set start stop for mesh size
  p.add("start", 5);
  p.add("stop", 5);

  // Set pvd filename base
  p.add("filename", "uh");

  // Plot or not
  p.add("plot", false);

  // Errors or not
  p.add("errors", true);

  // Read parameters
  p.parse(argc, argv);
  if (p["debug"])
    set_log_level(DBG);
  const std::size_t Nmeshes = p["Nmeshes"];
  const std::size_t start = p["start"];
  const std::size_t stop = p["stop"];
  std::string filename = p["filename"];
  const bool do_plot = p["plot"];
  const bool do_errors = p["errors"];

  for (std::size_t i = start; i <= stop; ++i)
  {
    const double h = std::pow(2, -i);
    double L2_error;
    solve_random_meshes(Nmeshes, h, filename, do_plot, do_errors, L2_error);
    std::cout << h << ' ' << L2_error << std::endl;
  }

}
