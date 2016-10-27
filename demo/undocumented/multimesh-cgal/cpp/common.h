#include <dolfin/generation/UnitSquareMesh.h>
#include <dolfin/generation/RectangleMesh.h>

namespace dolfin
{
void build_failing_case(MultiMesh& multimesh)
{
  const double h = 0.5;
  
  // Background
  {
    std::shared_ptr<Mesh> background_mesh(new UnitSquareMesh((int)std::round(1./h),
                                                             (int)std::round(1./h)));
    multimesh.add(background_mesh);
  }

  // Mesh 1
  {
    const double x0 = 0.35404867974764142602;
    const double y0 = 0.16597416632155614913;
    const double x1 = 0.63997881656511634851;
    const double y1 = 0.68786139026650294781;
    std::shared_ptr<Mesh> mesh(new RectangleMesh(Point(x0, y0), Point(x1, y1),
                                                 std::max((int)std::round((x1-x0)/h), 1),
                                                 std::max((int)std::round((y1-y0)/h), 1)));
    mesh->rotate(39.609407484349517858);
    multimesh.add(mesh);
  }

  // Mesh 2
  {
    const double x0 = 0.33033712968711609337;
    const double y0 = 0.22896817104377231722;
    const double x1 = 0.82920109332967595339;
    const double y1 = 0.89337241458397931293;
    std::shared_ptr<Mesh> mesh(new RectangleMesh(Point(x0, y0), Point(x1, y1),
                                                 std::max((int)std::round((x1-x0)/h), 1),
                                                 std::max((int)std::round((y1-y0)/h), 1)));
    mesh->rotate(31.532416069662392744);
    multimesh.add(mesh);
  }

  // Mesh 3
  {
    const double x0 = 0.28105941241656401397;
    const double y0 = 0.30745787374091237965;
    const double x1 = 0.61959648394007071914;
    const double y1 = 0.78600209801737319637;
    std::shared_ptr<Mesh> mesh(new RectangleMesh(Point(x0, y0), Point(x1, y1),
                                                 std::max((int)std::round((x1-x0)/h), 1),
                                                 std::max((int)std::round((y1-y0)/h), 1)));
    mesh->rotate(40.233022128340330426);
    multimesh.add(mesh);
  }

  multimesh.build();
}
}
