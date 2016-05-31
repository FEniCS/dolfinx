#include <cmath>
#include <fstream>
#include <dolfin.h>

#include <dolfin/geometry/dolfin_simplex_tools.h>

using namespace dolfin;

int main(int argc, char* argv[])
{
  const std::size_t N = 1;
  auto mesh_0 = std::make_shared<UnitSquareMesh>(N, N);

  // double exact_area = 0;
  // const double L1 = 0.4;
  // auto mesh_1 = std::make_shared<RectangleMesh>(Point(0.05, 0.95-L1), Point(0.05+L1, 0.95), N, N);
  // exact_area += 4*L1;
  // const double L2 = 0.1;
  // auto mesh_2 = std::make_shared<RectangleMesh>(Point(0.1, 0.9-L2), Point(0.1+L2, 0.9), N, N);


  // double exact_area = 0;
  // const Point p0(0.1, 0.2);
  // const double L1 = 0.4;
  // auto mesh_1 = std::make_shared<RectangleMesh>(p0, p0 + Point(L1,L1), N, N);
  // exact_area += 4*L1;
  // const Point p1(0.2, 0.4);
  // const double L2 = L1;
  // auto mesh_2 = std::make_shared<RectangleMesh>(p1, p1 + Point(L2,L2), N, N);
  // exact_area += 4*L2 - 0.5*L1 - 0.75*L1;


  // double exact_area = 0;
  // auto mesh_1 = std::make_shared<RectangleMesh>(Point(0.1, 0.1), Point(0.9, 0.9), N, N);
  // mesh_1->translate(Point(-0.05, 0.05));
  // exact_area += 4*0.8;
  // auto mesh_2 = std::make_shared<RectangleMesh>(Point(0.2, 0.2), Point(0.8, 0.8), N, N);
  // mesh_2->translate(Point(-0.025, 0.025));
  // exact_area += 4*0.6;
  // auto mesh_3 = std::make_shared<RectangleMesh>(Point(0.3, 0.3), Point(0.7, 0.7), N, N);
  // mesh_3->translate(Point(-0.0125, 0.0125));
  // exact_area += 4*0.4;
  // auto mesh_4 = std::make_shared<RectangleMesh>(Point(0.4, 0.4), Point(0.6, 0.6), N, N);
  // mesh_4->translate(Point(-0.0125/2, 0.0125/2));
  // exact_area += 4*0.2;


  const std::size_t Nx=1, Ny=1;
  std::vector<std::shared_ptr<const Mesh>> meshes(11);
  //auto mesh_0 = meshes[0] = std::make_shared<UnitSquareMesh>(Nx, Ny);
  auto mesh_1 = meshes[1] = std::make_shared<RectangleMesh>(Point(0.489594, 0.353142), Point(0.503781, 0.877049), 1, 1);
  const double exact_area = std::abs(0.489594-0.503781)*2 + std::abs(0.353142- 0.877049)*2;



  // tools::dolfin_write_medit_triangles("mesh0",*mesh_0);
  // tools::dolfin_write_medit_triangles("mesh1",*mesh_1);
  // tools::dolfin_write_medit_triangles("mesh2",*mesh_2);

  const double exact_volume = 1.;

  // Build multimesh
  auto multimesh = std::make_shared<MultiMesh>();
  multimesh->add(mesh_0);
  multimesh->add(mesh_1);
  // multimesh->add(mesh_2);
  // multimesh->add(mesh_3);
  // multimesh->add(mesh_4);
  multimesh->build(1);
  tools::dolfin_write_medit_triangles("multimesh",*multimesh);

//   const double volume = tools::compute_volume(*multimesh);
//   std::cout << "volume " << volume << ' ' << exact_volume <<" error="<< std::abs(volume-exact_volume) << std::endl;

//   const double volume_overlap = tools::compute_volume_overlap(*multimesh);
//   std::cout << "volume_overlap = " << volume_overlap << std::endl;

  const double area = tools::compute_interface_area(*multimesh);
  std::cout << "area " << area << ' ' << exact_area << " error="<<std::abs(area-exact_area) << std::endl;

}
