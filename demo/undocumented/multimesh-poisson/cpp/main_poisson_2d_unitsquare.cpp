#include <cmath>
#include <dolfin.h>
#include "MultiMeshPoisson.h"

#include "P1.h"
#include <dolfin/geometry/dolfin_simplex_tools.h>
#include "mmtools.h"
#include "fem.h"

using namespace dolfin;

void solve_poisson()
{

  // Create meshes
  const std::size_t N = 8;
  auto mesh_0 = std::make_shared<UnitSquareMesh>(N, N);

  const double L1 = 0.4;
  auto mesh_1 = std::make_shared<RectangleMesh>(Point(0.05, 0.95-L1), Point(0.05+L1, 0.95), N, N);
  const double L2 = 0.1;
  auto mesh_2 = std::make_shared<RectangleMesh>(Point(0.1, 0.9-L2), Point(0.1+L2, 0.9), N, N);

  // Build multimesh
  auto multimesh = std::make_shared<MultiMesh>();
  multimesh->add(mesh_0);
  multimesh->add(mesh_1);
  // multimesh->add(mesh_2);
  multimesh->build(); // qr generated here

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

  auto u = fem::solve<MultiMeshPoisson::MultiMeshFunctionSpace,
		      MultiMeshPoisson::MultiMeshBilinearForm,
		      MultiMeshPoisson::MultiMeshLinearForm>
    (multimesh,
     boundary);

  // Files for storing solution
  File u0_file("u0.pvd");
  File u1_file("u1.pvd");
  File u2_file("u2.pvd");

  // Save to file
  u0_file << *u->part(0);
  u1_file << *u->part(1);

}


int main(int argc, char* argv[])
{
  if (dolfin::MPI::size(MPI_COMM_WORLD) > 1)
  {
    info("Sorry, this demo does not (yet) run in parallel.");
    return 0;
  }


  solve_poisson();

}
