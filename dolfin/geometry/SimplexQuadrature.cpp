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
// First added:  2014-02-24
// Last changed: 2016-11-17

#include <dolfin/log/log.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshGeometry.h>
#include "SimplexQuadrature.h"
#include "predicates.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
std::pair<std::vector<double>, std::vector<double>>
  SimplexQuadrature::compute_quadrature_rule(const Cell& cell,
                                             std::size_t order)
{
  // Extract dimensions
  const std::size_t tdim = cell.mesh().topology().dim();
  const std::size_t gdim = cell.mesh().geometry().dim();

  // Get vertex coordinates
  std::vector<double> x;
  cell.get_vertex_coordinates(x);

  // Convert to std::vector<Point>
  std::vector<Point> s(tdim + 1);
  for (std::size_t t = 0; t < tdim + 1; ++t)
    for (std::size_t d = 0; d < gdim; ++d)
      s[t][d] = x[gdim*t + d];

  // Call function to compute quadrature rule
  return compute_quadrature_rule(s, gdim, order);
}
//-----------------------------------------------------------------------------
std::pair<std::vector<double>, std::vector<double>>
  SimplexQuadrature::compute_quadrature_rule(const std::vector<Point>& coordinates,
                                             std::size_t gdim,
                                             std::size_t order)
{
  std::size_t tdim = coordinates.size() - 1;

  switch (tdim)
  {
  case 0:
    // FIXME: should we return empty qr or should we have detected this earlier?
    break;
  case 1:
    return compute_quadrature_rule_interval(coordinates, gdim, order);
    break;
  case 2:
    return compute_quadrature_rule_triangle(coordinates, gdim, order);
    break;
  case 3:
    return compute_quadrature_rule_tetrahedron(coordinates, gdim, order);
    break;
  default:
    dolfin_error("SimplexQuadrature.cpp",
                 "compute quadrature rule for simplex",
                 "Only implemented for topological dimension 1, 2, 3");
  };

  std::pair<std::vector<double>, std::vector<double>> quadrature_rule;
  return quadrature_rule;
}
//-----------------------------------------------------------------------------
std::pair<std::vector<double>, std::vector<double>>
SimplexQuadrature::compute_quadrature_rule_interval(const std::vector<Point>& coordinates,
                                                    std::size_t gdim,
                                                    std::size_t order)
{
  std::pair<std::vector<double>, std::vector<double>> quadrature_rule;

  // Weights and points in local coordinates on [-1, 1]
  std::vector<double> w, p;

  switch (order)
  {
  case 1:
    // Assign weight 2, point 0
    w.assign(1, 2.);
    p.assign(1, 0.);

    break;
  case 2:
    // Assign weights 1.
    w.assign(2, 1.);

    // Assign points corresponding to -1/sqrt(3) and 1/sqrt(3)
    p = { -1./std::sqrt(3), 1./std::sqrt(3) };

    break;
  case 3:
    // Assign weights
    w = { 5./9, 8./9, 5./9 };

    // Assign points
    p = { -std::sqrt(3./5), 0., std::sqrt(3./5) };

    break;
  case 4:
    // Assign weights
    w.resize(4);
    w[0] = (18 - std::sqrt(30)) / 36;
    w[1] = (18 + std::sqrt(30)) / 36;
    w[2] = w[1];
    w[3] = w[0];

    // Assign points
    p.resize(4);
    p[0] = -std::sqrt(3./7 + 2./7*std::sqrt(6./5));
    p[1] = -std::sqrt(3./7 - 2./7*std::sqrt(6./5));
    p[2] = -p[1];
    p[3] = -p[0];

    break;
  case 5:
    // Assign weights
    w = { 0.2369268850561890875142640,
          0.4786286704993664680412915,
          0.5688888888888888888888889,
          0.4786286704993664680412915,
          0.2369268850561890875142640 };

    // Assign points
    p = { -0.9061798459386639927976269,
          -0.5384693101056830910363144,
          0.0000000000000000000000000,
          0.5384693101056830910363144,
          0.9061798459386639927976269 };

    break;
  case 6:
    // Assign weights
    w = { 0.1713244923791703450402961,
          0.3607615730481386075698335,
          0.4679139345726910473898703,
          0.4679139345726910473898703,
          0.3607615730481386075698335,
          0.1713244923791703450402961};

    // Assign points
    p = { -0.9324695142031520278123016,
          -0.6612093864662645136613996,
          -0.2386191860831969086305017,
          0.2386191860831969086305017,
          0.6612093864662645136613996,
          0.9324695142031520278123016 };

    break;
  default:
    dolfin_error("SimplexQuadrature.cpp",
                 "compute quadrature rule for interval",
                 "Not implemented for order ",order);
  }

  // Find the determinant of the Jacobian (inspired by ufc_geometry.h)
  double det = -1;

  switch (gdim)
  {
  case 1:
    det = coordinates[1].x() - coordinates[0].x();
    break;

  case 2:
    {
      const double J[] = {coordinates[1].x() - coordinates[0].x(),
                          coordinates[1].y() - coordinates[0].y()};
      const double det2 = J[0]*J[0] + J[1]*J[1];
      det = std::sqrt(det2);
      break;
    }
  case 3:
    {
      const double J[] = {coordinates[1].x() - coordinates[0].x(),
                          coordinates[1].y() - coordinates[0].y(),
			  coordinates[1].z() - coordinates[0].z()};
      const double det2 = J[0]*J[0] + J[1]*J[1];
      det = std::sqrt(det2);
      break;
    }
  default:
    dolfin_error("SimplexQuadrature.cpp",
                 "compute quadrature rule for interval",
                 "Not implemented for dimension %d", gdim);
  }

  // Map (local) quadrature points
  quadrature_rule.first.resize(gdim*p.size());
  for (std::size_t i = 0; i < p.size(); ++i)
  {
    for (std::size_t d = 0; d < gdim; ++d)
    {
      quadrature_rule.first[d + i*gdim]
        = 0.5*(coordinates[0][d]*(1 - p[i]) + coordinates[1][d]*(1 + p[i]));
      dolfin_assert(std::isfinite(quadrature_rule.first[d + i*gdim]));
    }
  }

  dolfin_assert(det >= 0);

  // Store weights
  quadrature_rule.second.assign(w.size(), 0.5*std::abs(det));
  for (std::size_t i = 0; i < w.size(); ++i)
  {
    quadrature_rule.second[i] *= w[i];
    dolfin_assert(std::isfinite(quadrature_rule.second[i]));
  }

  return quadrature_rule;
}
//-----------------------------------------------------------------------------
std::pair<std::vector<double>, std::vector<double>>
SimplexQuadrature::compute_quadrature_rule_triangle(const std::vector<Point>& coordinates,
                                                    std::size_t gdim,
                                                    std::size_t order)
{
  std::pair<std::vector<double>, std::vector<double>> quadrature_rule;

  // Weights and points in local coordinates on triangle [0,0], [1,0]
  // and [0,1]
  std::vector<double> w;
  std::vector<std::vector<double>> p;

  switch (order)
  {
  case 1:
    // Assign weight 1 and midpoint
    w.assign(1, 1.);
    p.assign(1, std::vector<double>(3, 1./3));

    break;
  case 2:
    // Assign weight 1/3
    w.assign(3, 1./3);

    // Assign points corresponding to 2/3, 1/6, 1/6
    p.assign(3, std::vector<double>(3, 1./6));
    p[0][0] = p[1][1] = p[2][2] = 2./3;

    break;
    // case 3: We do not include this case due to negative weights
    // // Assign weights
    // w.resize(4);
    // w[0] = -27./48;
    // w[1] = w[2] = w[3] = 25./48;
    // // Assign points
    // p.resize(4);
    // p[0] = { 1./3, 1./3, 1./3 };
    // p[1] = { 0.2, 0.2, 0.6 };
    // p[2] = { 0.2, 0.6, 0.2 };
    // p[3] = { 0.6, 0.2, 0.2 };
    // break;
  case 4:
    // Assign weights
    w = { 0.223381589678011,
	  0.223381589678011,
	  0.223381589678011,
	  0.109951743655322,
	  0.109951743655322,
	  0.109951743655322 };

    // Assign points
    p.resize(6);
    p[0] = { 0.445948490915965, 0.445948490915965, 0.10810301816807 };
    p[1] = { 0.445948490915965, 0.10810301816807,  0.445948490915965 };
    p[2] = { 0.10810301816807,  0.445948490915965, 0.445948490915965 };
    p[3] = { 0.091576213509771, 0.091576213509771, 0.816847572980458 };
    p[4] = { 0.091576213509771, 0.816847572980459, 0.09157621350977 };
    p[5] = { 0.816847572980459, 0.091576213509771, 0.09157621350977 };

    break;
  case 5:
    // Assign weights
    w = { 0.225,
          0.132394152788506,
          0.132394152788506,
          0.132394152788506,
          0.125939180544827,
          0.125939180544827,
          0.125939180544827 };

    // Assign points
    p.resize(7);
    p[0] = { 0.3333333333333335, 0.3333333333333335, 0.3333333333333330 };
    p[1] = { 0.4701420641051150, 0.4701420641051150, 0.0597158717897700 };
    p[2] = { 0.4701420641051150, 0.0597158717897700, 0.4701420641051151 };
    p[3] = { 0.0597158717897700, 0.4701420641051150, 0.4701420641051151 };
    p[4] = { 0.1012865073234560, 0.1012865073234560, 0.7974269853530880 };
    p[5] = { 0.1012865073234560, 0.7974269853530870, 0.1012865073234570 };
    p[6] = { 0.7974269853530870, 0.1012865073234560, 0.1012865073234570 };

    break;
  case 6:
    // Assign weights
    w = { 0.1167862757263790,
	  0.1167862757263790,
	  0.1167862757263790,
	  0.0508449063702070,
	  0.0508449063702070,
	  0.0508449063702070,
	  0.0828510756183740,
	  0.0828510756183740,
	  0.0828510756183740,
	  0.0828510756183740,
	  0.0828510756183740,
	  0.0828510756183740 };

    // Assign points
    p.resize(12);
    p[0] = { 0.2492867451709100, 0.2492867451709100, 0.5014265096581800 };
    p[1] = { 0.2492867451709100, 0.5014265096581790, 0.2492867451709110 };
    p[2] = { 0.5014265096581790, 0.2492867451709100, 0.2492867451709110 };
    p[3] = { 0.0630890144915020, 0.0630890144915020, 0.8738219710169960 };
    p[4] = { 0.0630890144915020, 0.8738219710169960, 0.0630890144915019 };
    p[5] = { 0.8738219710169960, 0.0630890144915020, 0.0630890144915019 };
    p[6] = { 0.3103524510337840, 0.6365024991213990, 0.0531450498448169 };
    p[7] = { 0.6365024991213990, 0.0531450498448170, 0.3103524510337841 };
    p[8] = { 0.0531450498448170, 0.3103524510337840, 0.6365024991213990 };
    p[9] = { 0.3103524510337840, 0.0531450498448170, 0.6365024991213990 };
    p[10] = { 0.6365024991213990, 0.3103524510337840, 0.0531450498448169 };
    p[11] = { 0.0531450498448170, 0.6365024991213990, 0.3103524510337841 };

    break;
  default:
    dolfin_error("SimplexQuadrature.cpp",
                 "compute quadrature rule for triangle",
                 "Not implemented for order ", order);
  }

  // Find the determinant of the Jacobian (inspired by ufc_geometry.h)
  double det = 0; // To keep compiler happy

  switch (gdim)
  {
  case 2:
    {
      double a[] = {coordinates[0].x(), coordinates[0].y()};
      double b[] = {coordinates[1].x(), coordinates[1].y()};
      double c[] = {coordinates[2].x(), coordinates[2].y()};
      det = orient2d(a,b,c);

      break;
    }
  case 3:
    {
      const double J[] = {coordinates[1].x() - coordinates[0].x(),
                          coordinates[2].x() - coordinates[0].x(),
                          coordinates[1].y() - coordinates[0].y(),
                          coordinates[2].y() - coordinates[0].y(),
                          coordinates[1].z() - coordinates[0].z(),
                          coordinates[2].z() - coordinates[0].z()};
      const double d_0 = J[2]*J[5] - J[4]*J[3];
      const double d_1 = J[4]*J[1] - J[0]*J[5];
      const double d_2 = J[0]*J[3] - J[2]*J[1];
      const double det2 = d_0*d_0 + d_1*d_1 + d_2*d_2;
      det = std::sqrt(det2);

      break;
    }
  default:
    dolfin_error("SimplexQuadrature.cpp",
                 "compute quadrature rule for triangle",
                 "Not implemented for dimension ", gdim);
  }

  // Store points
  quadrature_rule.first.resize(gdim*p.size());
  for (std::size_t i = 0; i < p.size(); ++i)
    for (std::size_t d = 0; d < gdim; ++d)
    {
      quadrature_rule.first[d + i*gdim]
        = p[i][0]*coordinates[0][d]
        + p[i][1]*coordinates[1][d]
        + p[i][2]*coordinates[2][d];
      dolfin_assert(std::isfinite(quadrature_rule.first[d + i*gdim]));
    }

  // Store weights
  quadrature_rule.second.assign(w.size(), 0.5*std::abs(det));
  for (std::size_t i = 0; i < w.size(); ++i)
  {
    quadrature_rule.second[i] *= w[i];
    dolfin_assert(std::isfinite(quadrature_rule.second[i]));
  }

  return quadrature_rule;
}
//-----------------------------------------------------------------------------
std::pair<std::vector<double>, std::vector<double>>
SimplexQuadrature::compute_quadrature_rule_tetrahedron(const std::vector<Point>& coordinates,
                                                       std::size_t gdim,
                                                       std::size_t order)
{
  std::pair<std::vector<double>, std::vector<double>> quadrature_rule;

  // Weights and points in local coordinates on tetrahedron [0,0,0],
  // [1,0,0], [0,1,0] and [0,0,1]
  std::vector<double> w;
  std::vector<std::vector<double>> p;

  switch (order)
  {
  case 1:
    // Assign weight 1 and midpoint
    w.assign(1, 1.);
    p.assign(1, std::vector<double>(4, 0.25));

    break;
  case 2:
    // Assign weight 0.25
    w.assign(4, 0.25);

    // Assign points corresponding to 0.585410196624969,
    // 0.138196601125011, 0.138196601125011 and 0.138196601125011
    p.assign(4, std::vector<double>(4, 0.138196601125011));
    p[0][0] = p[1][1] = p[2][2] = p[3][3] = 0.585410196624969;

    break;
  case 3:
    // Assign weights
    w = { -4./5,
	   9./20,
	   9./20,
	   9./20,
	   9./20 };

    // Assign points
    p.resize(5);
    p[0] = { 0.25, 0.25, 0.25, 0.25 };
    p[1] = { 1./6, 1./6, 1./6, 0.5 };
    p[2] = { 1./6, 1./6, 0.5,  1./6 };
    p[3] = { 1./6, 0.5,  1./6, 1./6 };
    p[4] = { 0.5,  1./6, 1./6, 1./6 };

    break;
  case 4:
    // Assign weights
    w = { -0.0789333333333330,
	  0.0457333333333335,
	  0.0457333333333335,
	  0.0457333333333335,
	  0.0457333333333335,
	  0.1493333333333332,
	  0.1493333333333332,
	  0.1493333333333332,
	  0.1493333333333332,
	  0.1493333333333332,
	  0.1493333333333332 };

    // Assign points
    p.resize(11);
    p[0] = { 0.2500000000000000, 0.2500000000000000, 0.2500000000000000, 0.2500000000000000 };
    p[1] = { 0.0714285714285715, 0.0714285714285715, 0.0714285714285715, 0.7857142857142855 };
    p[2] = { 0.0714285714285715, 0.0714285714285715, 0.7857142857142855, 0.0714285714285715 };
    p[3] = { 0.0714285714285715, 0.7857142857142855, 0.0714285714285715, 0.0714285714285715 };
    p[4] = { 0.7857142857142855, 0.0714285714285715, 0.0714285714285715, 0.0714285714285715 };
    p[5] = { 0.3994035761667990, 0.3994035761667990, 0.1005964238332010, 0.1005964238332010 };
    p[6] = { 0.3994035761667990, 0.1005964238332010, 0.3994035761667990, 0.1005964238332010 };
    p[7] = { 0.1005964238332010, 0.3994035761667990, 0.3994035761667990, 0.1005964238332010 };
    p[8] = { 0.3994035761667990, 0.1005964238332010, 0.1005964238332010, 0.3994035761667990 };
    p[9] = { 0.1005964238332010, 0.3994035761667990, 0.1005964238332010, 0.3994035761667990 };
    p[10] = { 0.1005964238332010, 0.1005964238332010, 0.3994035761667990, 0.3994035761667990 };

    break;
  case 5:
    // Assign weights
    w = { 0.0734930431163618,
	  0.0734930431163618,
	  0.0734930431163618,
	  0.0734930431163618,
	  0.1126879257180158,
	  0.1126879257180158,
	  0.1126879257180158,
	  0.1126879257180158,
	  0.0425460207770813,
	  0.0425460207770813,
	  0.0425460207770813,
	  0.0425460207770813,
	  0.0425460207770813,
	  0.0425460207770813 };

    // Assign points
    p.resize(14);
    p[0] = { 0.0927352503108910, 0.0927352503108910, 0.0927352503108910, 0.7217942490673269 };
    p[1] = { 0.7217942490673265, 0.0927352503108910, 0.0927352503108910, 0.0927352503108915 };
    p[2] = { 0.0927352503108910, 0.7217942490673265, 0.0927352503108910, 0.0927352503108915 };
    p[3] = { 0.0927352503108910, 0.0927352503108910, 0.7217942490673265, 0.0927352503108915 };
    p[4] = { 0.3108859192633005, 0.3108859192633005, 0.3108859192633005, 0.0673422422100984 };
    p[5] = { 0.0673422422100980, 0.3108859192633005, 0.3108859192633005, 0.3108859192633010 };
    p[6] = { 0.3108859192633005, 0.0673422422100980, 0.3108859192633005, 0.3108859192633010 };
    p[7] = { 0.3108859192633005, 0.3108859192633005, 0.0673422422100980, 0.3108859192633010 };
    p[8] = { 0.4544962958743505, 0.4544962958743505, 0.0455037041256495, 0.0455037041256495 };
    p[9] = { 0.4544962958743505, 0.0455037041256495, 0.4544962958743505, 0.0455037041256495 };
    p[10] = { 0.0455037041256495, 0.4544962958743505, 0.4544962958743505, 0.0455037041256495 };
    p[11] = { 0.4544962958743505, 0.0455037041256495, 0.0455037041256495, 0.4544962958743505 };
    p[12] = { 0.0455037041256495, 0.4544962958743505, 0.0455037041256495, 0.4544962958743505 };
    p[13] = { 0.0455037041256495, 0.0455037041256495, 0.4544962958743505, 0.4544962958743505 };

    break;
  case 6:
    // Assign weights
    w = { 0.0399227502581678,
	  0.0399227502581678,
	  0.0399227502581678,
	  0.0399227502581678,
	  0.0100772110553205,
	  0.0100772110553205,
	  0.0100772110553205,
	  0.0100772110553205,
	  0.0553571815436550,
	  0.0553571815436550,
	  0.0553571815436550,
	  0.0553571815436550,
	  0.0482142857142855,
	  0.0482142857142855,
	  0.0482142857142855,
	  0.0482142857142855,
	  0.0482142857142855,
	  0.0482142857142855,
	  0.0482142857142855,
	  0.0482142857142855,
	  0.0482142857142855,
	  0.0482142857142855,
	  0.0482142857142855,
	  0.0482142857142855 };

    // Assign points
    p.resize(24);
    p[0] = { 0.2146028712591520, 0.2146028712591520, 0.2146028712591520, 0.3561913862225440 };
    p[1] = { 0.3561913862225440, 0.2146028712591520, 0.2146028712591520, 0.2146028712591520 };
    p[2] = { 0.2146028712591520, 0.3561913862225440, 0.2146028712591520, 0.2146028712591520 };
    p[3] = { 0.2146028712591520, 0.2146028712591520, 0.3561913862225440, 0.2146028712591520 };
    p[4] = { 0.0406739585346115, 0.0406739585346115, 0.0406739585346115, 0.8779781243961655 };
    p[5] = { 0.8779781243961660, 0.0406739585346115, 0.0406739585346115, 0.0406739585346112 };
    p[6] = { 0.0406739585346115, 0.8779781243961660, 0.0406739585346115, 0.0406739585346112 };
    p[7] = { 0.0406739585346115, 0.0406739585346115, 0.8779781243961660, 0.0406739585346111 };
    p[8] = { 0.3223378901422755, 0.3223378901422755, 0.3223378901422755, 0.0329863295731734 };
    p[9] = { 0.0329863295731735, 0.3223378901422755, 0.3223378901422755, 0.3223378901422754 };
    p[10] = { 0.3223378901422755, 0.0329863295731735, 0.3223378901422755, 0.3223378901422754 };
    p[11] = { 0.3223378901422755, 0.3223378901422755, 0.0329863295731735, 0.3223378901422754 };
    p[12] = { 0.0636610018750175, 0.0636610018750175, 0.2696723314583160, 0.6030056647916490 };
    p[13] = { 0.0636610018750175, 0.2696723314583160, 0.0636610018750175, 0.6030056647916490 };
    p[14] = { 0.0636610018750175, 0.0636610018750175, 0.6030056647916490, 0.2696723314583160 };
    p[15] = { 0.0636610018750175, 0.6030056647916490, 0.0636610018750175, 0.2696723314583160 };
    p[16] = { 0.0636610018750175, 0.2696723314583160, 0.6030056647916490, 0.0636610018750174 };
    p[17] = { 0.0636610018750175, 0.6030056647916490, 0.2696723314583160, 0.0636610018750174 };
    p[18] = { 0.2696723314583160, 0.0636610018750175, 0.0636610018750175, 0.6030056647916490 };
    p[19] = { 0.2696723314583160, 0.0636610018750175, 0.6030056647916490, 0.0636610018750174 };
    p[20] = { 0.2696723314583160, 0.6030056647916490, 0.0636610018750175, 0.0636610018750174 };
    p[21] = { 0.6030056647916490, 0.0636610018750175, 0.2696723314583160, 0.0636610018750174 };
    p[22] = { 0.6030056647916490, 0.0636610018750175, 0.0636610018750175, 0.2696723314583160 };
    p[23] = { 0.6030056647916490, 0.2696723314583160, 0.0636610018750175, 0.0636610018750174 };

    break;
  default:
    dolfin_error("SimplexQuadrature.cpp",
                 "compute quadrature rule for triangle",
                 "Not implemented for order ", order);
  }

  // Find the determinant of the Jacobian (from ufc_geometry.h)
  double det = 0; // To keep compiler happy

  switch (gdim)
  {
  case 3:
    {
      const double J[] = {coordinates[1].x() - coordinates[0].x(),
                          coordinates[2].x() - coordinates[0].x(),
                          coordinates[3].x() - coordinates[0].x(),
                          coordinates[1].y() - coordinates[0].y(),
                          coordinates[2].y() - coordinates[0].y(),
                          coordinates[3].y() - coordinates[0].y(),
                          coordinates[1].z() - coordinates[0].z(),
                          coordinates[2].z() - coordinates[0].z(),
                          coordinates[3].z() - coordinates[0].z()};
      double d[9];
      d[0*3 + 0] = J[4]*J[8] - J[5]*J[7];
      // d[0*3 + 1] = J[5]*J[6] - J[3]*J[8];
      // d[0*3 + 2] = J[3]*J[7] - J[4]*J[6];
      d[1*3 + 0] = J[2]*J[7] - J[1]*J[8];
      // d[1*3 + 1] = J[0]*J[8] - J[2]*J[6];
      // d[1*3 + 2] = J[1]*J[6] - J[0]*J[7];
      d[2*3 + 0] = J[1]*J[5] - J[2]*J[4];
      // d[2*3 + 1] = J[2]*J[3] - J[0]*J[5];
      // d[2*3 + 2] = J[0]*J[4] - J[1]*J[3];

      det = J[0]*d[0*3 + 0] + J[3]*d[1*3 + 0] + J[6]*d[2*3 + 0];
      break;
    }
  default:
    dolfin_error("SimplexQuadrature.cpp",
                 "compute quadrature rule for tetrahedron",
                 "Not implemented for dimension ", gdim);
  }

  // Store points
  quadrature_rule.first.resize(gdim*p.size());
  for (std::size_t i = 0; i < p.size(); ++i)
    for (std::size_t d = 0; d < gdim; ++d)
      quadrature_rule.first[d + i*gdim]
        = p[i][0]*coordinates[0][d]
        + p[i][1]*coordinates[1][d]
        + p[i][2]*coordinates[2][d]
        + p[i][3]*coordinates[3][d];

  // Store weights
  quadrature_rule.second.assign(w.size(), std::abs(det) / 6.);
  for (std::size_t i = 0; i < w.size(); ++i)
    quadrature_rule.second[i] *= w[i];

  return quadrature_rule;
}
//-----------------------------------------------------------------------------
