#include <cmath>
#include <dolfin.h>
//#include "dolfin_simplex_tools.h"
using namespace dolfin;
//-----------------------------------------------------------------------------

int main()
{
  const std::size_t N = 1;
  auto background_mesh = std::make_shared<Mesh>(UnitCubeMesh(N, N, N));
  auto box_mesh = std::make_shared<Mesh>(BoxMesh(Point(0.1, 0.1, 0.1),
                                                 Point(0.7, 0.6, 0.5),
                                                 N, N, N));
  // tools::dolfin_write_medit("mesh0", *background_mesh);
  // tools::dolfin_write_medit("box_mesh", *box_mesh);

  std::vector<std::shared_ptr<const Mesh> > meshes;
  meshes.push_back(background_mesh);
  meshes.push_back(box_mesh);

  const std::size_t quadrature_order = 1;
  std::shared_ptr<MultiMesh> multimesh(new MultiMesh(meshes, quadrature_order));
  std::cout << "MultiMesh done"<<std::endl;

  // tools::dolfin_write_medit("mm", *multimesh);
  // tools::writemarkers("mm", *multimesh);

  return 0;
}
