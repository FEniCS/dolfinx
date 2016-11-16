// Copyright (C) 2014 August Johansson and Anders Logg
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2014-03-10
// Last changed: 2016-11-16
//


//#define CGAL_HEADER_ONLY 1
#include <dolfin/mesh/MultiMesh.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/math/basic.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/generation/UnitSquareMesh.h>

// We need to use epeck here. Qoutient<MP_FLOAT> as number type gives overflow
#include <CGAL/Exact_predicates_exact_constructions_kernel.h>

#include <CGAL/Triangle_2.h>
#include <CGAL/intersection_2.h>
#include <CGAL/Boolean_set_operations_2.h>
#include <CGAL/Polygon_set_2.h>

#include "common.h"

typedef CGAL::Epeck ExactKernel;
typedef ExactKernel::FT FT;
typedef ExactKernel::Point_2                      Point_2;
typedef ExactKernel::Triangle_2                   Triangle_2;
typedef ExactKernel::Line_2                       Line_2;
typedef CGAL::Polygon_2<ExactKernel>              Polygon_2;
typedef Polygon_2::Vertex_const_iterator          Vertex_const_iterator;
typedef CGAL::Polygon_with_holes_2<ExactKernel>   Polygon_with_holes_2;
typedef Polygon_with_holes_2::Hole_const_iterator Hole_const_iterator;
typedef CGAL::Polygon_set_2<ExactKernel>          Polygon_set_2;


#define MULTIMESH_DEBUG_OUTPUT 0

using namespace dolfin;

//------------------------------------------------------------------------------
double rotate(double x, double y, double cx, double cy, double w,
              double& xr, double& yr)
{
  // std::cout << "rotate:\n"
  // 	      << "\t"
  // 	      << "plot("<<x<<','<<y<<",'b.');plot("<<cx<<','<<cy<<",'o');";

  const double v = w*DOLFIN_PI/180.;
  const double dx = x-cx;
  const double dy = y-cy;
  xr = cx + dx*cos(v) - dy*sin(v);
  yr = cy + dx*sin(v) + dy*cos(v);
  //std::cout << "plot("<<xr<<','<<yr<<",'r.');"<<std::endl;
}
//------------------------------------------------------------------------------
bool rotation_inside(double x,double y, double cx, double cy, double w,
                     double& xr, double& yr)
{
  rotate(x,y,cx,cy,w, xr,yr);
  if (xr>0 and xr<1 and yr>0 and yr<1) return true;
  else return false;
}
//------------------------------------------------------------------------------
std::shared_ptr<MultiMesh> get_test_case(std::size_t num_parts,
                                         std::size_t Nx)
{
  const double h = 1. / Nx;

  std::vector<std::vector<Point>> points =
    {
      { Point(0.747427, 0.186781), Point(0.849659, 0.417130) },
      { Point(0.152716, 0.471681), Point(0.455943, 0.741585) },
      { Point(0.464473, 0.251876), Point(0.585051, 0.533569) },
      { Point(0.230112, 0.511897), Point(0.646974, 0.892193) },
      { Point(0.080362, 0.422675), Point(0.580151, 0.454286) },
      { Point(0.054755, 0.534186), Point(0.444096, 0.743028) },
      { Point(0.246347, 0.643033), Point(0.611211, 0.644119) },
      { Point(0.205060, 0.610043), Point(0.873978, 0.716453) },
      { Point(0.315601, 0.609597), Point(0.660068, 0.889642) },
      { Point(0.510442, 0.227960), Point(0.547733, 0.681651) },
      { Point(0.631782, 0.625248), Point(0.939937, 0.734633) },
      { Point(0.630514, 0.130426), Point(0.789759, 0.605254) },
      { Point(0.366874, 0.077819), Point(0.799717, 0.247002) },
      { Point(0.634452, 0.166130), Point(0.720499, 0.957903) },
      { Point(0.310176, 0.376406), Point(0.733204, 0.559494) },
      { Point(0.457794, 0.195561), Point(0.669801, 0.814364) },
      { Point(0.507921, 0.543829), Point(0.604354, 0.637256) },
      { Point(0.042519, 0.138396), Point(0.289660, 0.698498) },
      { Point(0.531146, 0.112000), Point(0.786345, 0.496440) },
      { Point(0.755547, 0.804976), Point(0.783852, 0.905749) },
      { Point(0.135990, 0.421423), Point(0.721689, 0.598717) },
      { Point(0.294942, 0.314007), Point(0.823500, 0.606792) },
      { Point(0.384000, 0.479182), Point(0.974647, 0.613010) },
      { Point(0.356791, 0.728380), Point(0.770159, 0.802342) },
      { Point(0.371233, 0.542066), Point(0.978624, 0.648035) },
      { Point(0.719941, 0.326586), Point(0.878289, 0.517651) },
      { Point(0.810330, 0.554044), Point(0.865130, 0.586726) },
      { Point(0.715135, 0.517116), Point(0.773488, 0.867894) },
      { Point(0.063054, 0.556346), Point(0.385853, 0.739599) },
      { Point(0.416569, 0.045984), Point(0.691243, 0.529570) },
      { Point(0.415423, 0.634109), Point(0.566714, 0.768729) },
      { Point(0.765750, 0.205375), Point(0.933004, 0.430380) },
      { Point(0.584073, 0.423089), Point(0.598574, 0.677415) },
      { Point(0.202824, 0.350436), Point(0.602496, 0.350622) },
      { Point(0.166122, 0.147433), Point(0.584457, 0.873023) },
      { Point(0.276331, 0.306927), Point(0.756209, 0.962107) },
      { Point(0.148461, 0.046180), Point(0.490417, 0.947779) },
      { Point(0.072863, 0.633958), Point(0.289068, 0.825418) },
      { Point(0.782015, 0.454994), Point(0.912484, 0.721990) },
      { Point(0.305546, 0.449086), Point(0.838901, 0.454008) },
      { Point(0.375838, 0.409314), Point(0.645968, 0.761390) },
      { Point(0.169756, 0.500825), Point(0.550513, 0.613456) },
      { Point(0.217436, 0.170923), Point(0.408271, 0.211427) },
      { Point(0.393828, 0.373691), Point(0.472135, 0.735436) },
      { Point(0.523821, 0.426144), Point(0.902130, 0.426845) },
      { Point(0.273991, 0.113629), Point(0.635717, 0.519508) },
      { Point(0.524145, 0.159222), Point(0.876302, 0.369831) },
      { Point(0.535785, 0.284029), Point(0.864152, 0.895106) },
      { Point(0.464676, 0.083968), Point(0.671590, 0.211442) },
      { Point(0.594766, 0.011844), Point(0.707712, 0.265403) }
    };

  std::vector<double> angles =
    {{ 88.339755, 94.547259, 144.366564, 172.579922, 95.439692, 106.697958, 175.788281, 172.468177, 40.363410, 103.866765, 143.106588, 98.869318, 20.516877, 35.108539, 137.423965, 90.249864, 34.446790, 4.621397, 72.857255, 159.991224, 178.510861, 55.788859, 28.163059, 132.222868, 29.606199, 174.993928, 148.036367, 19.177764, 168.827333, 168.008844, 94.710245, 46.129366, 111.622092, 13.585448, 150.515846, 6.340156, 13.178734, 159.027957, 64.313903, 77.979669, 138.651353, 18.916756, 39.967938, 71.345030, 76.804783, 167.944421, 18.516992, 17.648271, 104.164880, 30.083616}};

  dolfin_assert(num_parts <= angles.size());
  std::vector<std::shared_ptr<const Mesh>> meshes;
  meshes.reserve(num_parts);

  meshes.push_back(std::make_shared<UnitSquareMesh>(std::round(1./h), std::round(1./h)));

  for (std::size_t i = 0; i < num_parts-1; ++i)
  {
    std::shared_ptr<Mesh> m =
      std::make_shared<RectangleMesh>(points[i][0], points[i][1],
                                      std::max<std::size_t>(std::round(std::abs(points[i][0].x()-points[i][1].x()) / h), 1),
                                      std::max<std::size_t>(std::round(std::abs(points[i][0].y()-points[i][1].y()) / h), 1));
    m->rotate(angles[i]);
    meshes.push_back(m);
  }

  const std::size_t quadrature_order = 1;
  std::shared_ptr<MultiMesh> multimesh(new MultiMesh(meshes, quadrature_order));
  return multimesh;
}
//-----------------------------------------------------------------------------
std::shared_ptr<MultiMesh> test_volume_2d_rot(std::size_t /*Nx*/,
					      std::size_t num_meshes)
{
  // Background mesh
  auto mesh_0  = std::make_shared<UnitSquareMesh>(1, 1);
  mesh_0->scale(10.0);
  mesh_0->translate(Point(-5,-5));

  // List of meshes
  std::vector<std::shared_ptr<const Mesh> > meshes;
  meshes.reserve(num_meshes + 1);
  meshes.push_back(mesh_0);

  for (std::size_t i = 0; i < num_meshes; ++i)
  {
    auto mesh = std::make_shared<UnitSquareMesh>(1, 1);
    const double angle = 2*DOLFIN_PI*i / num_meshes;
    std::cout << i<<' '<<angle << std::endl;
    mesh->translate(Point(-0.5, -0.5));
    mesh->scale(2.0);
    mesh->rotate(180.0*angle / DOLFIN_PI);
    mesh->translate(Point(cos(angle), sin(angle)));
    meshes.push_back(mesh);
  }

  // // Save meshes to file so we can examine them
  // File('background_mesh.pvd') << mesh_0;
  // File vtkfile('meshes.pvd');
  // for (const auto mesh: meshes)
  //   vtkfile << mesh;

  // Create multimesh
  const std::size_t quadrature_order = 1;
  std::shared_ptr<MultiMesh> multimesh(new MultiMesh(meshes, quadrature_order));

  return multimesh;
}

//-----------------------------------------------------------------------------
int main(int argc, char** argv)
{
  // set_log_level(TRACE);

  for (std::size_t Nx = 1; Nx < 2; Nx++)
  {
    for (std::size_t parts = 8; parts < 9; parts++)
    {
      std::cout << "\n\nNx = " << Nx << ", numparts = " << parts << std::endl;

      //std::shared_ptr<MultiMesh> m = get_test_case(Nx, parts);
      std::shared_ptr<MultiMesh> m = test_volume_2d_rot(Nx, parts);

      MultiMesh& multimesh = *m;

      // std::cout << multimesh.plot_matplotlib() << std::endl;
      std::cout << "Done building multimesh" << std::endl;
      /* ---------------- Done creating multimesh ----------------------- */

      // Compute volume of each cell using cgal
      std::vector<std::vector<std::pair<CELL_STATUS, FT>>> cell_status_cgal;
      get_cells_status_cgal(multimesh, cell_status_cgal);
      std::cout << "Done computing volumes with cgal" << std::endl;

      // Compute volume of each cell using dolfin::MultiMesh
      std::vector<std::vector<std::pair<CELL_STATUS, double> > > cell_status_multimesh;
      compute_volume(multimesh, cell_status_multimesh);
      std::cout << "Done computing volumes with multimesh" << std::endl;

      FT cgal_volume = 0.;
      double multimesh_volume = 0.;

      dolfin_assert(cell_status_cgal.size() == cell_status_multimesh.size());
      for (std::size_t i = 0; i < cell_status_cgal.size(); i++)
      {
        const std::vector<std::pair<CELL_STATUS, FT> >& current_cgal = cell_status_cgal[i];
        const std::vector<std::pair<CELL_STATUS, double> >& current_multimesh = cell_status_multimesh[i];

        dolfin_assert(current_cgal.size() == current_multimesh.size());

	std::cout << "Cells in part " << i << ": " << std::endl;
        for (std::size_t j = 0; j < current_cgal.size(); j++)
        {
          std::cout << "  Cell " << j << std::endl;
          std::cout << "    Multimesh: " << cell_status_str(current_multimesh[j].first) << " (" << current_multimesh[j].second << ")" << std::endl;
          std::cout << "    CGAL:      " << cell_status_str(current_cgal[j].first) << " (" << current_cgal[j].second << ")" << std::endl;
          std::cout << "      Diff:    " << (current_cgal[j].second - current_multimesh[j].second) << std::endl;
          cgal_volume += current_cgal[j].second;
          multimesh_volume += current_multimesh[j].second;
          dolfin_assert(near(CGAL::to_double(current_cgal[j].second), current_multimesh[j].second, DOLFIN_EPS_LARGE));
          // dolfin_assert(current_cgal[j].first == current_multimesh[j].first);
        }
        std::cout << std::endl;
      }

      // Exact volume is known
      const FT exact_volume = 100;

      std::cout << "Total volume" << std::endl;
      std::cout << "------------" << std::endl;
      std::cout << "Multimesh: " << multimesh_volume << ", error: " << (exact_volume-multimesh_volume) << std::endl;
      std::cout << "CGAL:      " << cgal_volume << ", error: " << (exact_volume-cgal_volume) << std::endl;

      dolfin_assert(near(CGAL::to_double(exact_volume), multimesh_volume, DOLFIN_EPS_LARGE));
    }
  }
}
