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
// Last changed: 2015-06-08
//

#define CGAL_HEADER_ONLY

#include <dolfin/geometry/Point.h>
#include <dolfin/geometry/IntersectionTriangulation.h>


#include <CGAL/MP_Float.h>
#include <CGAL/Quotient.h>
#include <CGAL/Cartesian.h>

typedef CGAL::Quotient<CGAL::MP_Float>       FT;
typedef CGAL::Cartesian<FT>                  ExactKernel;
typedef ExactKernel::Point_2                 Point_2;
typedef ExactKernel::Triangle_2              Triangle_2;
typedef ExactKernel::Segment_2               Segment_2;


using namespace dolfin;

int main(int argc, char** argv)
{
  std::cout << std::setprecision(20);

  {
    // Triangle_2 t1(Point_2(0.5441393035168431424608571, 0.1820158210510630092393569),
    //               Point_2(0.2746075928361219053996933, 0.5815739727225808231025894),
    //               Point_2(0.4407905340183594367076125, 0.7190984237966810965758668));

    // Triangle_2 t2(Point_2(0.5225164146556802169385492, 0.2943855907617997091918483),
    //               Point_2(0.4756496317473773993711461, 0.254733764692555708641919),
    //               Point_2(0.4357425571052688795248287, 0.7149209778502514378573096));
    // //std::cout << t1[0] << " " << t1[1] << " " << t1[2] << std::endl;

    // std::cout << tools::drawtriangle(cgaltools::convert(t1))<<tools::drawtriangle(cgaltools::convert(t2))<<std::endl;

    // auto cell_intersection = CGAL::intersection(t1, t2);
    // dolfin_assert(cell_intersection);
    // const std::vector<Point_2>* polygon = boost::get<std::vector<Point_2>>(&*cell_intersection);
    // dolfin_assert(polygon);
    // std::cout << "Polygon: " << std::endl;
    // for (const Point_2& p : *polygon)
    // {
    //   std::cout << p << std::endl;
    // }
    // std::cout << std::endl;



    Triangle_2 t1(Point_2(0., 0.),
		  Point_2(1., 0.),
		  Point_2(1., 1.));
    Segment_2 t2(Point_2(0.5, 0.2),
		 Point_2(0.5, -0.5));

    // std::cout << tools::drawtriangle(cgaltools::convert(t1))<<tools::drawtriangle(cgaltools::convert(t2))<<std::endl;

  }

//   std::vector<Point> tri1(3);
//   tri1.push_back(Point(0.5441393035168431424608571, 0.1820158210510630092393569));
//   tri1.push_back(Point(0.2746075928361219053996933, 0.5815739727225808231025894));
//   tri1.push_back(Point(0.4407905340183594367076125, 0.7190984237966810965758668));

//   std::vector<Point> tri2(3);
//   tri2.push_back(Point(0.5225164146556802169385492, 0.2943855907617997091918483));
//   tri2.push_back(Point(0.4756496317473773993711461, 0.254733764692555708641919));
//   tri2.push_back(Point(0.4357425571052688795248287, 0.7149209778502514378573096));

//   const std::vector<double> triangulation = IntersectionTriangulation::triangulate_intersection_triangle_triangle(tri1, tri2);

//   std::cout << "Size of triangulation: " << triangulation.size() << std::endl;
//

}
