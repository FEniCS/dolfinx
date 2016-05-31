#include <cmath>
#include <dolfin.h>
#include "MultiMeshPoisson.h"
#include "MultiMeshArea.h"

#include "P1.h"
#include <dolfin/geometry/dolfin_simplex_tools.h>
#include "mmtools.h"
#include "fem.h"

using namespace dolfin;

// Templated function for computation of multimesh L2 norm of error
template<class TFunctional>
double MultiMeshL2Error(const MultiMeshFunctionSpace& V,
                        const MultiMeshFunction& uh,
                        const Expression& u)
{
  TFunctional M0(V.multimesh()->part(0));
  TFunctional M1(V.multimesh()->part(1));

  M0.uexact = u;
  M0.uh = *uh.part(0);

  M1.uexact = u;
  M1.uh = *uh.part(1);

  MultiMeshForm M(V);
  M.add(M0);
  M.add(M1);
  M.build();

  MultiMeshAssembler assembler;
  Scalar m;

  assembler.assemble(m, M);

  return std::sqrt(m.get_scalar_value());
}

double area(const MultiMeshFunctionSpace& V)
{
  auto mmfunctionals = std::vector<std::make_shared<const MultiMeshArea::Functional> >(V.multimesh()->num_parts());

  MultiMeshArea::Functional M0(V.multimesh()->part(0));
  MultiMeshArea::Functional M1(V.multimesh()->part(1));

  Constant ones(1.);
  M0.ones = ones;
  M1.ones = ones;

  MultiMeshForm M(V);
  M.add(M0);
  M.add(M1);
  M.build();

  MultiMeshAssembler assembler;
  Scalar m;

  assembler.assemble(m, M);

  return m.get_scalar_value();
}


void solve_poisson(std::size_t N_meshes,
		   const std::size_t Nx,
		   const std::size_t Ny,
		   std::size_t quadrature_order)
{
  // Create random meshes inside UnitSquare (see matlab script in multimesh-2015)
  std::vector<std::shared_ptr<const Mesh>> meshes(11);
  meshes[0] = std::make_shared<UnitSquareMesh>(Nx, Ny);
  meshes[1] = std::make_shared<RectangleMesh>(Point(0.489594, 0.353142), Point(0.503781, 0.877049), 1, 5);
  meshes[2] = std::make_shared<RectangleMesh>(Point(0.586440, 0.361022), Point(0.675112, 0.620278), 1, 3);
  meshes[3] = std::make_shared<RectangleMesh>(Point(0.019257, 0.651350), Point(0.083874, 0.974802), 1, 3);
  meshes[4] = std::make_shared<RectangleMesh>(Point(0.122021, 0.257846), Point(0.403491, 0.268439), 3, 1);
  meshes[5] = std::make_shared<RectangleMesh>(Point(0.152234, 0.121658), Point(0.348008, 0.884153), 2, 8);
  meshes[6] = std::make_shared<RectangleMesh>(Point(0.399020, 0.047401), Point(0.930041, 0.342374), 5, 3);
  meshes[7] = std::make_shared<RectangleMesh>(Point(0.544906, 0.686223), Point(0.794682, 0.893633), 2, 2);
  meshes[8] = std::make_shared<RectangleMesh>(Point(0.046192, 0.195477), Point(0.303661, 0.720166), 3, 5);
  meshes[9] = std::make_shared<RectangleMesh>(Point(0.433261, 0.269092), Point(0.470625, 0.560713), 1, 3);
  meshes[10] = std::make_shared<RectangleMesh>(Point(0.503888, 0.138725), Point(0.646810, 0.307746), 1, 2);

  // Build multimesh
  auto multimesh = std::make_shared<MultiMesh>();
  for (std::size_t i = 0; i < N_meshes; ++i)
    multimesh->add(meshes[i]);
  multimesh->build(quadrature_order);


  {
    double volume = mmtools::compute_volume(*multimesh, 0);
    double area = mmtools::compute_interface_area(*multimesh, 0);
    std::cout << "volume " << volume << std::endl
	      << "area " << area << std::endl;
  }

  mmtools::writemarkers(0, *multimesh);

  // Sub domain for Dirichlet boundary condition
  class DirichletBoundary : public SubDomain
  {
    bool inside(const Array<double>& x, bool on_boundary) const
    {
      return on_boundary;
    }
  };
  auto boundary = std::make_shared<DirichletBoundary>();

  // Source
  class Source: public Expression
  {
    void eval(Array<double>& values, const Array<double>& xx) const
    {
      const double x=xx[0], y=xx[1];
      values[0] = DOLFIN_PI*DOLFIN_PI*sin(DOLFIN_PI*x)*sin(DOLFIN_PI*y);
    }
  };
  auto source = std::make_shared<Source>();

  auto u = fem::solve<MultiMeshPoisson::MultiMeshFunctionSpace,
		      MultiMeshPoisson::MultiMeshBilinearForm,
		      MultiMeshPoisson::MultiMeshLinearForm>
    (multimesh,
     boundary,
     source);

  // Files for storing solution
  File u0_file("u0.pvd");
  File u1_file("u1.pvd");
  File u2_file("u2.pvd");
  File u3_file("u3.pvd");
  File u4_file("u4.pvd");
  File u5_file("u5.pvd");
  File u6_file("u6.pvd");
  File u7_file("u7.pvd");
  File u8_file("u8.pvd");
  File u9_file("u9.pvd");
  File u10_file("u10.pvd");
  // std::vector<File> files = {{ u0_file, u1_file, u2_file, u3_file, u4_file, u5_file, u6_file,
  // 			       u7_file, u8_file, u9_file, u10_file, u11_file }};
  // for (std::size_t i = 0; i < N_meshes; ++i)
  //   files[i] << *u->part(i);

  u0_file << *u->part(0);
  u1_file << *u->part(1);
  u2_file << *u->part(2);
  u3_file << *u->part(3);
  u4_file << *u->part(4);
  u5_file << *u->part(5);
  u6_file << *u->part(6);
  u7_file << *u->part(7);
  u8_file << *u->part(8);
  u9_file << *u->part(9);
  u10_file << *u->part(10);


  std::vector<double> maxvals;
  tools::find_max<P1::FunctionSpace>(0, *multimesh, *u, maxvals);
  for (std::size_t i = 0; i < multimesh->num_parts(); ++i)
    std::cout << "part i " << i << " maxval " << maxvals[i] << std::endl;

  const double glob_max = *std::max_element(maxvals.begin(), maxvals.end());
  const double h = 0.5*(1./Nx + 1./Ny);

  std::cout << h <<' '<< glob_max<<std::endl;

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
  //parameters["allow_extrapolation"] = true; // FIXME: Why is this needed?

  // Application specific parameters
  Parameters p("my_parameters");
  p.add("debug", false);
  p.add("N_meshes", 11);
  p.add("quadrature_order", 3);
  p.add("Nx", 10);
  p.parse(argc, argv);

  // Read parameters
  if (p["debug"])
    set_log_level(DBG);
  const std::size_t N_meshes = p["N_meshes"];
  const std::size_t quadrature_order = p["quadrature_order"];
  const std::size_t Nx = p["Nx"];

  solve_poisson(N_meshes, Nx, Nx, quadrature_order);

}
