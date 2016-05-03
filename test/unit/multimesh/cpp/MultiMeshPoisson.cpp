#include <cmath>
#include <dolfin.h>
#include "MultiMeshPoisson.h"
#include "MultiMeshL2Norm.h"

#include <dolfin/geometry/dolfin_simplex_tools.h>

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

std::string fullname(const std::string& basename,
		     std::size_t part,
		     double h)
{
  std::stringstream ss;
  ss << basename << "-part" << part << "-h" << h;
  std::string str = ss.str();
  std::replace(str.begin(), str.end(), '.', '_');
  return ss.str();
}


void solve_random_meshes(std::size_t Nmeshes,
			 std::size_t start,
			 std::size_t stop,
			 std::string& filename,
			 bool do_plot,
			 bool do_errors,
			 std::vector<double>& hh,
			 std::vector<double>& L2_errors)
{
  std::cout << "Solve Poisson on " << Nmeshes << "+1 meshes" << std::endl;

  hh.resize(Nmeshes);
  L2_errors.resize(Nmeshes);

  std::vector<Point> xy0(Nmeshes), xy1(Nmeshes);

  // Create random rectangle mesh
  // for (std::size_t part = 0; part < Nmeshes; ++part)
  // {
  //   // Define random rectangle mesh
  //   xy0[part] = Point(dolfin::rand(), dolfin::rand());
  //   xy1[part] = Point(dolfin::rand(), dolfin::rand());
  //   if (xy0[part][0] > xy1[part][0]) std::swap(xy0[part][0], xy1[part][0]);
  //   if (xy0[part][1] > xy1[part][1]) std::swap(xy0[part][1], xy1[part][1]);
  //   std::cout << xy0[part] << " " << xy1[part] << std::endl;
  // }

  // Create meshes not aligned with each other
  const double hmin = 1. / std::pow(2.0, stop+1);
  for (std::size_t part = 0; part < Nmeshes; ++part)
  {
    xy0[part] = Point((part+0.5)*hmin, (part+1)*hmin);
    xy1[part] = Point(1, 1) - xy0[part];
  }

  for (std::size_t s = start; s <= stop; ++s)
  {
    // Mesh size
    const double h = hh[s-start] = 1./std::pow(2.0, s);

    std::cout << "run " << s << " mesh size " << h << std::endl;

    // Data
    Constant zero(0);
    DirichletBoundary bdry;
    Source f;

    MultiMeshFunctionSpace V;
    V.parameters("multimesh")["quadrature_order"] = 2;

    MultiMesh mm; // only for plotting

    // Background mesh
    const std::size_t min_elements = 1;
    const std::size_t N = std::max(min_elements, static_cast<std::size_t>(std::round(1. / h)));
    UnitSquareMesh usm(N, N);

    mm.add(usm);

    // Add up
    MultiMeshPoisson::FunctionSpace V_part(usm);
    V.add(V_part);

    for (std::size_t part = 0; part < Nmeshes; ++part)
    {
      // Create meshes
      const std::size_t m = static_cast<std::size_t>(std::round((xy1[part][0]-xy0[part][0]) / h));
      const std::size_t n = static_cast<std::size_t>(std::round((xy1[part][1]-xy0[part][1]) / h));
      std::shared_ptr<Mesh> rm(new RectangleMesh(xy0[part], xy1[part],
						 std::max(m, min_elements),
						 std::max(n, min_elements)));
      std::shared_ptr<MultiMeshPoisson::FunctionSpace> V_part(new MultiMeshPoisson::FunctionSpace(rm));
      V.add(V_part);

      mm.add(rm);
    }
    V.build();

    mm.build();
    tools::dolfin_write_medit_triangles("multimesh", mm, s);

    // Create forms
    MultiMeshForm a(V, V);
    MultiMeshForm L(V);

    for (std::size_t part = 0; part < V.num_parts(); ++part)
    {
      //std::shared_ptr<const FunctionSpace> V_part = V.part(part);
      auto V_part = V.part(part);
      std::shared_ptr<MultiMeshPoisson::BilinearForm> a_part(new MultiMeshPoisson::BilinearForm(V_part, V_part));
      std::shared_ptr<MultiMeshPoisson::LinearForm> L_part(new MultiMeshPoisson::LinearForm(V_part));
      L_part->f = f;
      a.add(a_part);
      L.add(L_part);
    }
    a.build();
    L.build();

    // Assemble
    Matrix A;
    Vector b;
    MultiMeshAssembler mma;
    mma.assemble(A, a);
    mma.assemble(b, L);

    // BC
    MultiMeshDirichletBC bc(V, zero, bdry);
    bc.apply(A, b);

    // Solve
    MultiMeshFunction uh(V);
    solve(A, *uh.vector(), b);

    // Save
    for (std::size_t part = 0; part < V.num_parts(); ++part)
      File(fullname(filename, part, h) + ".pvd") << *uh.part(part);

    // Plot
    if (do_plot)
    {
      for (std::size_t part = 0; part < V.num_parts(); ++part)
	plot(uh.part(part), fullname(filename, part, h));
      plot(V.multimesh());
      interactive();
    }

    // Error
    if (do_errors)
    {
      L2_errors[s-start] = MultiMeshL2Error(V, uh);
      //H1_error = MultiMeshLH1Error(V, uh);
    }
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

  // // Application parameters
  // Parameters p("my_own_params");

  // // Debug or not
  // p.add("debug", false);

  // // Set number of random meshes on top of background unit square mesh
  // p.add("Nmeshes", 1);

  // // Set start stop for mesh size
  // p.add("start", 2);
  // p.add("stop", 6);

  // // Set pvd filename base
  // p.add("filename", "uh");

  // // Plot or not
  // p.add("plot", false);

  // // Errors or not
  // p.add("errors", true);

  // // Read parameters
  // p.parse(argc, argv);
  // if (p["debug"])
  // set_log_level(DBG);
  const std::size_t Nmeshes = 1;//p["Nmeshes"];
  const std::size_t start = 2;//p["start"];
  const std::size_t stop = 6;///p["stop"];
  std::string filename = "uh";//p["filename"];
  const bool do_plot = false;//p["plot"];
  const bool do_errors = true;//p["errors"];

  std::vector<double> hh, L2_errors;
  solve_random_meshes(Nmeshes, start, stop,
		      filename, do_plot, do_errors,
		      hh, L2_errors);

  for (std::size_t i = 0; i <= stop-start; ++i)
    std::cout << hh[i] << ' ' << L2_errors[i] << std::endl;

}
