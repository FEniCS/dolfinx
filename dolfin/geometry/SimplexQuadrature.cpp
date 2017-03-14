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
// Last changed: 2017-03-14

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

    // Assign points
    p = { -1./std::sqrt(3.),
	  1./std::sqrt(3.) };

    break;
  case 3:
    // Assign weights
    w = { 5./9.,
	  8./9.,
	  5./9. };

    // Assign points
    p = { -std::sqrt(3./5.),
	  0.,
	  std::sqrt(3./5.) };

    break;
  case 4:
    // Assign weights
    w.resize(4);
    w[0] = (18. - std::sqrt(30.)) / 36;
    w[1] = (18. + std::sqrt(30.)) / 36;
    w[2] = w[1];
    w[3] = w[0];

    // Assign points
    p.resize(4);
    p[0] = -std::sqrt(3./7. + 2./7.*std::sqrt(1.2));
    p[1] = -std::sqrt(3./7. - 2./7.*std::sqrt(1.2));
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
      const std::array<double, 2> J = { coordinates[1].x() - coordinates[0].x(),
					coordinates[1].y() - coordinates[0].y() };
      const double det2 = J[0]*J[0] + J[1]*J[1];
      det = std::sqrt(det2);
      break;
    }
  case 3:
    {
      const std::array<double, 3>  J = { coordinates[1].x() - coordinates[0].x(),
					 coordinates[1].y() - coordinates[0].y(),
					 coordinates[1].z() - coordinates[0].z() };
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
        = 0.5*(coordinates[0][d]*(1. - p[i]) + coordinates[1][d]*(1. + p[i]));
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
    p = { { 1./3., 1./3. } };

    break;
  case 2:
    // Assign weight 1/3
    w.assign(3, 1./3.);

    // Assign points corresponding to 2/3, 1/6, 1/6
    p.assign(3, std::vector<double>(2, 1./6.));
    p[0][0] = p[1][1] = 2./3.;

    break;
  case 3:
    // Assign weights
    w.assign(6, 1./6.);

    // Assign points
    p = { { 0.659027622374092, 0.231933368553031 },
	  { 0.659027622374092, 0.109039009072877 },
	  { 0.231933368553031, 0.659027622374092 },
	  { 0.231933368553031, 0.109039009072877 },
	  { 0.109039009072877, 0.659027622374092 },
	  { 0.109039009072877, 0.231933368553031 } };

    break;
  case 4:
    // Assign weights
    w = { 0.223381589678011,
	  0.223381589678011,
	  0.223381589678011,
	  0.109951743655322,
	  0.109951743655322,
	  0.109951743655322 };

    // Assign points
    p = { { 0.445948490915965, 0.445948490915965 },
	  { 0.445948490915965, 0.108103018168070 },
	  { 0.108103018168070, 0.445948490915965 },
	  { 0.091576213509771, 0.091576213509771 },
	  { 0.091576213509771, 0.816847572980459 },
	  { 0.816847572980459, 0.091576213509771 } };

    break;
  case 5:
    // Assign weights
    w = { 0.225,
          0.13239415278850616,
          0.13239415278850616,
          0.13239415278850616,
          0.12593918054482717,
          0.12593918054482717,
          0.12593918054482717 };

    // Assign points
    p = { { 0.33333333333333333, 0.33333333333333333 },
	  { 0.47014206410511505, 0.47014206410511505 },
	  { 0.47014206410511505, 0.05971587178976981 },
	  { 0.05971587178976981, 0.47014206410511505 },
	  { 0.10128650732345633, 0.10128650732345633 },
	  { 0.10128650732345633, 0.79742698535308720 },
	  { 0.79742698535308720, 0.10128650732345633 } };

    break;
  case 6:
    // Assign weights
    w = { 0.116786275726379,
	  0.116786275726379,
	  0.116786275726379,
	  0.050844906370207,
	  0.050844906370207,
	  0.050844906370207,
	  0.082851075618374,
	  0.082851075618374,
	  0.082851075618374,
	  0.082851075618374,
	  0.082851075618374,
	  0.082851075618374 };

    // Assign points
    p = { { 0.249286745170910, 0.249286745170910 },
	  { 0.249286745170910, 0.501426509658179 },
	  { 0.501426509658179, 0.249286745170910 },
	  { 0.063089014491502, 0.063089014491502 },
	  { 0.063089014491502, 0.873821971016996 },
	  { 0.873821971016996, 0.063089014491502 },
	  { 0.310352451033785, 0.636502499121399 },
	  { 0.636502499121399, 0.053145049844816 },
	  { 0.053145049844816, 0.310352451033785 },
	  { 0.310352451033785, 0.053145049844816 },
	  { 0.636502499121399, 0.310352451033785 },
	  { 0.053145049844816, 0.636502499121399 } };

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
      det = orient2d(coordinates[0], coordinates[1], coordinates[2]);

      break;
    }
  case 3:
    {
      const std::array<double, 6> J = { coordinates[1].x() - coordinates[0].x(),
					coordinates[2].x() - coordinates[0].x(),
					coordinates[1].y() - coordinates[0].y(),
					coordinates[2].y() - coordinates[0].y(),
					coordinates[1].z() - coordinates[0].z(),
					coordinates[2].z() - coordinates[0].z() };
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
        + (1. - p[i][0] - p[i][1])*coordinates[2][d];
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
    p.assign(1, std::vector<double>(3, 0.25));

    break;
  case 2:
    // Assign weights
    w.assign(4, 0.25);

    // Assign points
    p.assign(4, std::vector<double>(3, 0.138196601125011));
    p[0][0] = p[1][1] = p[2][2] = 0.585410196624969;

    break;
  case 3:
    // Assign weights
    w = { -4./5.,
	  9./20.,
	  9./20.,
	  9./20.,
	  9./20. };

    // Assign points
    p = { { 0.25,  0.25,  0.25  },
	  { 1./6., 1./6., 1./6. },
	  { 1./6., 1./6., 0.5,  },
	  { 1./6., 0.5,   1./6. },
	  { 0.5,   1./6., 1./6. } };

    break;
  case 4:
    // Assign weights
    // FIXME: Find new rule to avoid negative weight
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
    p = { { 0.2500000000000000, 0.2500000000000000, 0.2500000000000000 },
	  { 0.0714285714285715, 0.0714285714285715, 0.0714285714285715 },
	  { 0.0714285714285715, 0.0714285714285715, 0.7857142857142855 },
	  { 0.0714285714285715, 0.7857142857142855, 0.0714285714285715 },
	  { 0.7857142857142855, 0.0714285714285715, 0.0714285714285715 },
	  { 0.3994035761667990, 0.3994035761667990, 0.1005964238332010 },
	  { 0.3994035761667990, 0.1005964238332010, 0.3994035761667990 },
	  { 0.1005964238332010, 0.3994035761667990, 0.3994035761667990 },
	  { 0.3994035761667990, 0.1005964238332010, 0.1005964238332010 },
	  { 0.1005964238332010, 0.3994035761667990, 0.1005964238332010 },
	  { 0.1005964238332010, 0.1005964238332010, 0.3994035761667990 } };

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
    p = { { 0.0927352503108910, 0.0927352503108910, 0.0927352503108910 },
	  { 0.7217942490673265, 0.0927352503108910, 0.0927352503108910 },
	  { 0.0927352503108910, 0.7217942490673265, 0.0927352503108910 },
	  { 0.0927352503108910, 0.0927352503108910, 0.7217942490673265 },
	  { 0.3108859192633005, 0.3108859192633005, 0.3108859192633005 },
	  { 0.0673422422100980, 0.3108859192633005, 0.3108859192633005 },
	  { 0.3108859192633005, 0.0673422422100980, 0.3108859192633005 },
	  { 0.3108859192633005, 0.3108859192633005, 0.0673422422100980 },
	  { 0.4544962958743505, 0.4544962958743505, 0.0455037041256495 },
	  { 0.4544962958743505, 0.0455037041256495, 0.4544962958743505 },
	  { 0.0455037041256495, 0.4544962958743505, 0.4544962958743505 },
	  { 0.4544962958743505, 0.0455037041256495, 0.0455037041256495 },
	  { 0.0455037041256495, 0.4544962958743505, 0.0455037041256495 },
	  { 0.0455037041256495, 0.0455037041256495, 0.4544962958743505 } };

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
    p = { { 0.2146028712591520, 0.2146028712591520, 0.2146028712591520 },
	  { 0.3561913862225440, 0.2146028712591520, 0.2146028712591520 },
	  { 0.2146028712591520, 0.3561913862225440, 0.2146028712591520 },
	  { 0.2146028712591520, 0.2146028712591520, 0.3561913862225440 },
	  { 0.0406739585346115, 0.0406739585346115, 0.0406739585346115 },
	  { 0.8779781243961660, 0.0406739585346115, 0.0406739585346115 },
	  { 0.0406739585346115, 0.8779781243961660, 0.0406739585346115 },
	  { 0.0406739585346115, 0.0406739585346115, 0.8779781243961660 },
	  { 0.3223378901422755, 0.3223378901422755, 0.3223378901422755 },
	  { 0.0329863295731735, 0.3223378901422755, 0.3223378901422755 },
	  { 0.3223378901422755, 0.0329863295731735, 0.3223378901422755 },
	  { 0.3223378901422755, 0.3223378901422755, 0.0329863295731735 },
	  { 0.0636610018750175, 0.0636610018750175, 0.2696723314583160 },
	  { 0.0636610018750175, 0.2696723314583160, 0.0636610018750175 },
	  { 0.0636610018750175, 0.0636610018750175, 0.6030056647916490 },
	  { 0.0636610018750175, 0.6030056647916490, 0.0636610018750175 },
	  { 0.0636610018750175, 0.2696723314583160, 0.6030056647916490 },
	  { 0.0636610018750175, 0.6030056647916490, 0.2696723314583160 },
	  { 0.2696723314583160, 0.0636610018750175, 0.0636610018750175 },
	  { 0.2696723314583160, 0.0636610018750175, 0.6030056647916490 },
	  { 0.2696723314583160, 0.6030056647916490, 0.0636610018750175 },
	  { 0.6030056647916490, 0.0636610018750175, 0.2696723314583160 },
	  { 0.6030056647916490, 0.0636610018750175, 0.0636610018750175 },
	  { 0.6030056647916490, 0.2696723314583160, 0.0636610018750175 } };

    break;
  default:
    dolfin_error("SimplexQuadrature.cpp",
                 "compute quadrature rule for tetrahedron",
                 "Not implemented for order ", order);
  }

  // Find the determinant of the Jacobian (from ufc_geometry.h)
  double det = 0; // To keep compiler happy

  switch (gdim)
  {
  case 3:
    {
      const std::array<double, 9> J = { coordinates[1].x() - coordinates[0].x(),
					coordinates[2].x() - coordinates[0].x(),
					coordinates[3].x() - coordinates[0].x(),
					coordinates[1].y() - coordinates[0].y(),
					coordinates[2].y() - coordinates[0].y(),
					coordinates[3].y() - coordinates[0].y(),
					coordinates[1].z() - coordinates[0].z(),
					coordinates[2].z() - coordinates[0].z(),
					coordinates[3].z() - coordinates[0].z() };
      const std::array<double, 3> d = { J[4]*J[8] - J[5]*J[7],
					J[2]*J[7] - J[1]*J[8],
					J[1]*J[5] - J[2]*J[4] };
      det = J[0]*d[0] + J[3]*d[1] + J[6]*d[2];
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
        + (1. - p[i][0] - p[i][1] - p[i][2])*coordinates[3][d];

  // Store weights
  quadrature_rule.second.assign(w.size(), std::abs(det) / 6.);
  for (std::size_t i = 0; i < w.size(); ++i)
    quadrature_rule.second[i] *= w[i];

  return quadrature_rule;
}
//-----------------------------------------------------------------------------
