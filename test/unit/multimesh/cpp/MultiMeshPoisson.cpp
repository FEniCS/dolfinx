#include <cmath>
#include <dolfin.h>
#include "MultiMeshPoisson.h"

using namespace dolfin;

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
    return on_boundary and (x[0] < DOLFIN_EPS or
			    x[0] > 1-DOLFIN_EPS or
			    x[1] < DOLFIN_EPS or
			    x[1] > 1-DOLFIN_EPS);
  }
};


MultiMeshFunction solve_random_meshes(std::size_t N_meshes,
				      double h,
				      std::string& filename)
{
  const std::size_t N = static_cast<std::size_t>(std::round(1. / h));

  MultiMeshFunctionSpace V;
  V.parameters("multimesh")["quadrature_order"] = 2;

  // Background mesh
  UnitSquareMesh usm(N, N);
  MultiMeshPoisson::FunctionSpace V_part(usm);
  V.add(V_part);

  const std::size_t mn = 1;

  // Build function space
  for (std::size_t i = 0; i < N_meshes; ++i)
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
  MultiMeshFunction u(V);
  solve(A, *u.vector(), b);

  // Save
  for (std::size_t i = 0; i < V.num_parts(); ++i)
  {
    std::stringstream ss;
    ss << i;
    File(filename + ss.str() + ".pvd") << *u.part(i);

    plot(u.part(i), ss.str());
    interactive();
  }

}


// Compute solution for given mesh configuration
void solve(double t,
           double x1, double y1,
           double x2, double y2,
           bool plot_solution,
           File& u0_file, File& u1_file, File& u2_file)
{
  // Create meshes
  double r = 0.5;
  RectangleMesh mesh_0(Point(-r, -r), Point(r, r), 16, 16);
  RectangleMesh mesh_1(Point(x1 - r, y1 - r), Point(x1 + r, y1 + r), 8, 8);
  RectangleMesh mesh_2(Point(x2 - r, y2 - r), Point(x2 + r, y2 + r), 8, 8);
  mesh_1.rotate(70*t);
  mesh_2.rotate(-70*t);

  // Create function spaces
  MultiMeshPoisson::FunctionSpace V0(mesh_0);
  MultiMeshPoisson::FunctionSpace V1(mesh_1);
  MultiMeshPoisson::FunctionSpace V2(mesh_2);

  // FIXME: Some of this stuff may be wrapped or automated later to
  // avoid needing to explicitly call add() and build()

  // Create forms
  MultiMeshPoisson::BilinearForm a0(V0, V0);
  MultiMeshPoisson::BilinearForm a1(V1, V1);
  MultiMeshPoisson::BilinearForm a2(V2, V2);
  MultiMeshPoisson::LinearForm L0(V0);
  MultiMeshPoisson::LinearForm L1(V1);
  MultiMeshPoisson::LinearForm L2(V2);

  // Build multimesh function space
  MultiMeshFunctionSpace V;
  V.parameters("multimesh")["quadrature_order"] = 2;
  V.add(V0);
  V.add(V1);
  V.add(V2);
  V.build();

  // Set coefficients
  Source f;
  L0.f = f;
  L1.f = f;
  L2.f = f;

  // Build multimesh forms
  MultiMeshForm a(V, V);
  MultiMeshForm L(V);
  a.add(a0);
  a.add(a1);
  a.add(a2);
  L.add(L0);
  L.add(L1);
  L.add(L2);
  a.build();
  L.build();

  // Create boundary condition
  Constant zero(0);
  DirichletBoundary boundary;
  MultiMeshDirichletBC bc(V, zero, boundary);

  // Assemble linear system
  Matrix A;
  Vector b;
  MultiMeshAssembler assembler;
  assembler.assemble(A, a);
  assembler.assemble(b, L);

  // Apply boundary condition
  bc.apply(A, b);

  // Compute solution
  MultiMeshFunction u(V);
  solve(A, *u.vector(), b);

  // Save to file
  u0_file << *u.part(0);
  u1_file << *u.part(1);
  u2_file << *u.part(2);

  // Plot solution (last time)
  if (plot_solution)
  {
    plot(V.multimesh());
    plot(u.part(0), "u_0");
    plot(u.part(1), "u_1");
    plot(u.part(2), "u_2");
    interactive();
  }
}

int main(int argc, char* argv[])
{
  set_log_level(DBG);

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

  // Set number of random meshes on top of background unit square mesh
  p.add("N_meshes", 1);

  // Set (appriximate) mesh size
  p.add("h", 0.01);

  // Set pvd filename base
  p.add("filename", "uh");

  const std::size_t N_meshes = p["N_meshes"];
  const double h = p["h"];
  std::string filename = p["filename"];

  const auto uh = solve_random_meshes(N_meshes, h, filename);


  return 0;
}
