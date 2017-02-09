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
// Last changed: 2017-02-09
//


//#define CGAL_HEADER_ONLY 1
#include <dolfin/mesh/MultiMesh.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/math/basic.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/generation/UnitCubeMesh.h>
#include <dolfin/generation/BoxMesh.h>

#ifdef DOLFIN_ENABLE_CGAL_EXACT_ARITHMETIC
// We need to use epeck here. Qoutient<MP_FLOAT> as number type gives overflow
#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Triangle_2.h>
#include <CGAL/intersection_2.h>
#include <CGAL/Boolean_set_operations_2.h>
#include <CGAL/Polygon_set_2.h>
#endif

#include "common.h"

#define MULTIMESH_DEBUG_OUTPUT 0

using namespace dolfin;
using namespace std;

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
      { Point(0.317099, 0.034446, 0.381558), Point(0.694829, 0.950222, 0.438744) },
      { Point(0.445586, 0.646313, 0.276025), Point(0.489764, 0.709365, 0.754687) },
      { Point(0.136069, 0.549860, 0.144955), Point(0.869292, 0.579705, 0.853031) },
      { Point(0.075967, 0.123319, 0.183908), Point(0.401808, 0.239916, 0.239953) },
      { Point(0.296321, 0.188955, 0.183511), Point(0.547009, 0.744693, 0.686775) },
      { Point(0.081126, 0.486792, 0.435859), Point(0.929386, 0.775713, 0.446784) },
      { Point(0.794831, 0.378609, 0.532826), Point(0.817628, 0.644318, 0.811580) },
      { Point(0.550156, 0.207742, 0.301246), Point(0.622475, 0.587045, 0.470923) },
      { Point(0.438870, 0.111119, 0.408720), Point(0.979748, 0.258065, 0.594896) },
      { Point(0.117418, 0.296676, 0.424167), Point(0.221747, 0.318778, 0.507858) },
      { Point(0.231594, 0.488898, 0.395515), Point(0.521136, 0.624060, 0.679136) },
      { Point(0.500472, 0.059619, 0.042431), Point(0.712694, 0.471088, 0.681972) },
      { Point(0.817547, 0.149865, 0.518595), Point(0.818149, 0.722440, 0.659605) },
      { Point(0.060471, 0.416799, 0.627973), Point(0.399258, 0.526876, 0.656860) },
      { Point(0.666339, 0.539126, 0.178132), Point(0.701099, 0.698106, 0.666528) },
      { Point(0.032601, 0.669175, 0.190433), Point(0.561200, 0.881867, 0.368917) },
      { Point(0.644765, 0.190924, 0.428253), Point(0.855523, 0.376272, 0.482022) },
      { Point(0.384619, 0.251806, 0.265281), Point(0.582986, 0.290441, 0.617091) },
      { Point(0.312719, 0.161485, 0.094229), Point(0.425259, 0.178766, 0.422886) },
      { Point(0.153657, 0.281005, 0.457424), Point(0.266471, 0.440085, 0.527143) },
      { Point(0.425729, 0.644443, 0.635787), Point(0.546593, 0.647618, 0.679017) },
      { Point(0.119396, 0.450138, 0.458725), Point(0.236231, 0.607304, 0.661945) },
      { Point(0.416159, 0.256441, 0.582249), Point(0.841929, 0.832917, 0.613461) },
      { Point(0.119215, 0.645552, 0.479463), Point(0.318074, 0.939829, 0.639317) },
      { Point(0.093820, 0.530344, 0.393456), Point(0.525404, 0.861140, 0.484853) },
      { Point(0.359228, 0.394707, 0.442305), Point(0.736340, 0.683416, 0.704047) },
      { Point(0.377396, 0.216019, 0.327565), Point(0.755077, 0.790407, 0.949304) },
      { Point(0.406955, 0.789963, 0.318524), Point(0.748706, 0.825584, 0.534064) },
      { Point(0.479523, 0.227843, 0.498094), Point(0.527680, 0.801348, 0.900852) },
      { Point(0.246735, 0.083483, 0.625960), Point(0.585987, 0.666416, 0.660945) },
      { Point(0.499116, 0.123932, 0.490357), Point(0.535801, 0.445183, 0.852998) },
      { Point(0.564980, 0.205976, 0.082071), Point(0.640312, 0.417029, 0.947933) },
      { Point(0.573710, 0.052078, 0.728662), Point(0.620959, 0.931201, 0.737842) },
      { Point(0.347879, 0.054239, 0.330829), Point(0.446027, 0.177108, 0.662808) },
      { Point(0.539982, 0.287849, 0.414523), Point(0.706917, 0.999492, 0.464840) },
      { Point(0.178117, 0.056705, 0.175669), Point(0.359635, 0.521886, 0.335849) },
      { Point(0.468468, 0.104012, 0.561861), Point(0.912132, 0.745546, 0.736267) },
      { Point(0.526102, 0.707253, 0.287977), Point(0.603468, 0.729709, 0.781377) },
      { Point(0.061591, 0.337584, 0.104813), Point(0.780176, 0.607866, 0.741254) },
      { Point(0.487604, 0.272939, 0.037235), Point(0.768958, 0.396007, 0.673295) },
      { Point(0.059403, 0.696433, 0.125332), Point(0.315811, 0.772722, 0.130151) },
      { Point(0.655573, 0.108818, 0.126500), Point(0.722923, 0.531209, 0.631766) },
      { Point(0.168251, 0.316429, 0.217563), Point(0.196249, 0.317480, 0.251042) },
      { Point(0.530629, 0.335311, 0.299225), Point(0.832423, 0.597490, 0.452593) },
      { Point(0.325834, 0.398881, 0.180738), Point(0.525045, 0.546449, 0.415093) },
      { Point(0.064187, 0.671202, 0.642061), Point(0.228669, 0.767330, 0.715213) },
      { Point(0.317428, 0.789074, 0.505637), Point(0.814540, 0.852264, 0.635661) },
      { Point(0.383306, 0.575495, 0.275070), Point(0.813113, 0.617279, 0.530052) },
      { Point(0.411594, 0.583533, 0.551793), Point(0.602638, 0.750520, 0.583571) },
      { Point(0.378186, 0.224277, 0.269055), Point(0.704340, 0.729513, 0.673031) }
    };

  std::vector<double> angles =
    {{ 143.135982, 117.917641, 63.171429, 8.937797, 112.611341, 91.531558, 169.020281, 151.975583, 108.511756, 47.246802, 177.836761, 93.896972, 116.818469, 77.697211, 179.834471, 176.694831, 106.111347, 176.879412, 84.766366, 93.249380, 37.608286, 63.039242, 156.589386, 116.516066, 133.426430, 59.554418, 78.956097, 20.107034, 152.132073, 160.335381, 48.652980, 25.567402, 154.879301, 21.267936, 147.276727, 162.927641, 107.498043, 100.200570, 98.917219, 81.313062, 1.407653, 17.746937, 126.580180, 64.729137, 3.696439, 70.337175, 79.913548, 81.294979, 14.866691, 112.268954}};

  dolfin_assert(num_parts <= angles.size());
  std::vector<std::shared_ptr<const Mesh>> meshes(num_parts+1);

  meshes[0] = std::make_shared<UnitCubeMesh>(std::round(1./h), std::round(1./h), std::round(1./h));

  for (std::size_t i = 0; i < num_parts; ++i)
  {
    meshes[i+1]
      = std::make_shared<BoxMesh>(points[i][0], points[i][1],
				  std::max<std::size_t>(std::round(std::abs(points[i][0].x()-points[i][1].x()) / h), 1),
				  std::max<std::size_t>(std::round(std::abs(points[i][0].y()-points[i][1].y()) / h), 1),
				  std::max<std::size_t>(std::round(std::abs(points[i][0].z()-points[i][1].z()) / h), 1));
    // meshes[i+1]->rotate(angles[i]);
  }

  const std::size_t quadrature_order = 1;
  std::shared_ptr<MultiMesh> multimesh(new MultiMesh(meshes, quadrature_order));
  return multimesh;
}

int main(int argc, char** argv)
{
  set_log_level(DBG);

  // Simple test case to check 3D implementation that hangs strangely
  const std::size_t N = 1;
  auto m0 = make_shared<UnitCubeMesh>(N, N, N);
  auto m1 = make_shared<UnitCubeMesh>(N, N, N);

  Point p(0.25, 0.25);
  m1->scale(0.5);
  m1->translate(p);

  MultiMesh multimesh;
  multimesh.add(m0);
  multimesh.add(m1);
  multimesh.build();

  return 0;

  // Iterate over mesh sizes and numbers of parts
  for (std::size_t N = 2; N < 4; N++)
  {
    for (std::size_t num_parts = 2; num_parts < 10; num_parts++)
    {
      // Create test case and compute volume
      std::shared_ptr<MultiMesh> multimesh = get_test_case(N, num_parts);
      const double volume = multimesh->compute_volume();
      const double error = std::abs(1.0 - volume);

      info("N = %d num_parts = %d: volume = %g error = %g", N, num_parts, volume, error);
    }
  }

  return 0;
}
