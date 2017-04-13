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
// Last changed: 2017-04-09

#include <dolfin/log/log.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshGeometry.h>
#include "SimplexQuadrature.h"
#include "predicates.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
SimplexQuadrature::SimplexQuadrature(std::size_t tdim,
				     std::size_t order)
{
  // Create and store quadrature rule for reference simplex
  switch (tdim)
  {
  case 1:
    setup_qr_reference_interval(order);
    break;
  case 2:
    setup_qr_reference_triangle(order);
    break;
  case 3:
    setup_qr_reference_tetrahedron(order);
    break;
  default:
    dolfin_error("SimplexQuadrature.cpp",
                 "setup quadrature rule for reference simplex",
                 "Only implemented for topological dimension 1, 2, 3");
  }

}
//-----------------------------------------------------------------------------
std::pair<std::vector<double>, std::vector<double>>
  SimplexQuadrature::compute_quadrature_rule(const Cell& cell,
                                             std::size_t order) const
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
                                             std::size_t order) const
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
						      std::size_t order) const
{
  log(PROGRESS, "Create quadrature rule using given interval coordinates");

  std::pair<std::vector<double>, std::vector<double>> quadrature_rule;

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
					coordinates[1].y() - coordinates[0].y()};
      const double det2 = J[0]*J[0] + J[1]*J[1];
      det = std::sqrt(det2);
      break;
    }
  case 3:
    {
      const std::array<double, 3>  J = { coordinates[1].x() - coordinates[0].x(),
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

  // Map (local) quadrature points (note that _p is a
  // std::vector<std::vector<double> >)
  quadrature_rule.first.resize(gdim*_p[0].size());
  for (std::size_t i = 0; i < _p[0].size(); ++i)
  {
    for (std::size_t d = 0; d < gdim; ++d)
    {
      quadrature_rule.first[d + i*gdim]
        = 0.5*(coordinates[0][d]*(1. - _p[0][i])
	       + coordinates[1][d]*(1. + _p[0][i]));
      dolfin_assert(std::isfinite(quadrature_rule.first[d + i*gdim]));
    }
  }

  dolfin_assert(det >= 0);

  // Store weights
  quadrature_rule.second.assign(_w.size(), 0.5*std::abs(det));
  for (std::size_t i = 0; i < _w.size(); ++i)
  {
    quadrature_rule.second[i] *= _w[i];
    dolfin_assert(std::isfinite(quadrature_rule.second[i]));
  }

  dolfin_assert(quadrature_rule.first.size() == gdim*quadrature_rule.second.size());

  return quadrature_rule;
}
//-----------------------------------------------------------------------------
std::pair<std::vector<double>, std::vector<double>>
  SimplexQuadrature::compute_quadrature_rule_triangle(const std::vector<Point>& coordinates,
						      std::size_t gdim,
						      std::size_t order) const
{
  log(PROGRESS, "Create quadrature rule using given triangle coordinates");

  std::pair<std::vector<double>, std::vector<double>> quadrature_rule;

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
  quadrature_rule.first.resize(gdim*_p.size());
  for (std::size_t i = 0; i < _p.size(); ++i)
    for (std::size_t d = 0; d < gdim; ++d)
    {
      quadrature_rule.first[d + i*gdim]
        = _p[i][0]*coordinates[0][d]
        + _p[i][1]*coordinates[1][d]
        + (1. - _p[i][0] - _p[i][1])*coordinates[2][d];
      dolfin_assert(std::isfinite(quadrature_rule.first[d + i*gdim]));
    }

  // Store weights
  quadrature_rule.second.assign(_w.size(), 0.5*std::abs(det));
  for (std::size_t i = 0; i < _w.size(); ++i)
  {
    quadrature_rule.second[i] *= _w[i];
    dolfin_assert(std::isfinite(quadrature_rule.second[i]));
  }

  dolfin_assert(quadrature_rule.first.size() == gdim*quadrature_rule.second.size());

  return quadrature_rule;
}
//-----------------------------------------------------------------------------
std::pair<std::vector<double>, std::vector<double>>
  SimplexQuadrature::compute_quadrature_rule_tetrahedron(const std::vector<Point>& coordinates,
							 std::size_t gdim,
							 std::size_t order) const
{
  log(PROGRESS, "Create quadrature rule using given tetrahedron coordinates");

  std::pair<std::vector<double>, std::vector<double>> quadrature_rule;

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
					coordinates[3].z() - coordinates[0].z()};
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
  quadrature_rule.first.resize(gdim*_p.size());
  for (std::size_t i = 0; i < _p.size(); ++i)
    for (std::size_t d = 0; d < gdim; ++d)
      quadrature_rule.first[d + i*gdim]
        = _p[i][0]*coordinates[0][d]
        + _p[i][1]*coordinates[1][d]
        + _p[i][2]*coordinates[2][d]
        + (1. - _p[i][0] - _p[i][1] - _p[i][2])*coordinates[3][d];

  // Store weights
  quadrature_rule.second.assign(_w.size(), std::abs(det) / 6.);
  for (std::size_t i = 0; i < _w.size(); ++i)
    quadrature_rule.second[i] *= _w[i];

  dolfin_assert(quadrature_rule.first.size() == gdim*quadrature_rule.second.size());

  return quadrature_rule;
}
//-----------------------------------------------------------------------------
std::vector<std::size_t>
SimplexQuadrature::compress(std::pair<std::vector<double>, std::vector<double>>& qr,
			    std::size_t gdim,
			    std::size_t quadrature_order)
{
  // Polynomial degree N that can be integrated exactly using the qr_base
  const std::size_t N = quadrature_order;

  // By construction the compressed quadrature rule will not have more
  // than choose(N + gdim, gdim) points.
  const std::size_t N_compressed_min = choose(N + gdim, gdim);
  if (qr.second.size() <= N_compressed_min)
  {
    // We cannot improve this rule. Return empty vector
    return std::vector<std::size_t>();
  }

  // Copy the input qr since we'll overwrite the input
  const std::pair<std::vector<double>, std::vector<double>> qr_input = qr;

  // Create Vandermonde-type matrix using a basis of Chebyshev
  // polynomials of the first kind
  const Eigen::MatrixXd V = Chebyshev_Vandermonde_matrix(qr_input, gdim, N);

  // A QR decomposition selects the subset of N columns (geometrically
  // the N columns with same volume as spanned by all M columns).
  Eigen::ColPivHouseholderQR<Eigen::MatrixXd> QR(V);
  Eigen::MatrixXd Q = QR.householderQ();

  // We do not need the full Q matrix but only what's known as the
  // "economy size" decomposition
  Q *= Eigen::MatrixXd::Identity(V.rows(), std::min(V.rows(), V.cols()));

  // We'll use Q^T
  Q.transposeInPlace();

  // Compute weights in the new basis
  Eigen::Map<const Eigen::VectorXd> w_base(qr_input.second.data(),
					   qr_input.second.size());
  const Eigen::VectorXd nu = Q*w_base;

  // Compute new weights
  const Eigen::VectorXd w_new = Q.colPivHouseholderQr().solve(nu);

  // Construct new qr using the non-zero weights. First find the
  // indices for these weights.
  std::vector<std::size_t> indices;
  for (std::size_t i = 0; i < w_new.size(); ++i)
  {
    if (std::abs(w_new[i]) > 0.0)
      indices.push_back(i);
  }

  // Resize qr and overwrite the points and weights
  dolfin_assert(indices.size() <= N_compressed_min);
  qr.first.resize(gdim*indices.size());
  qr.second.resize(indices.size());

  for (std::size_t i = 0; i < indices.size(); ++i)
  {
    // Save points
    for (std::size_t d = 0; d < gdim; ++d)
      qr.first[gdim*i + d] = qr_input.first[gdim*indices[i] + d];

    // Save weights
    qr.second[i] = w_new[indices[i]];
  }

  // Return indices. These are useful for mapping additional data, for
  // example the normals.
  return indices;
}
//-----------------------------------------------------------------------------
void SimplexQuadrature::setup_qr_reference_interval(std::size_t order)
{
  // Create quadrature rule with points on reference element [-1, 1].
  _p.resize(1);
  legendre_compute_glr(order, _p[0], _w);

}
//-----------------------------------------------------------------------------
void SimplexQuadrature::setup_qr_reference_triangle(std::size_t order)
{
  // Create quadrature rule with points on reference triangle [0, 0],
  // [1, 0] and [0, 1]
  return dunavant_rule(order, _p, _w);
}
//-----------------------------------------------------------------------------
void SimplexQuadrature::setup_qr_reference_tetrahedron(std::size_t order)
{
  // FIXME: Replace these hard coded rules by a general function

  switch (order)
  {
  case 1:
    // Assign weight 1 and midpoint
    _w.assign(1, 1.);
    _p.assign(1, std::vector<double>(3, 0.25));

    break;
  case 2:
    // Assign weights
    _w.assign(4, 0.25);

    // Assign points
    _p.assign(4, std::vector<double>(3, 0.138196601125011));
    _p[0][0] = _p[1][1] = _p[2][2] = 0.585410196624969;

    break;
  case 3:
    // Assign weights
    _w = { -4./5.,
	  9./20.,
	  9./20.,
	  9./20.,
	  9./20. };

    // Assign points
    _p = { { 0.25,  0.25,  0.25  },
	   { 1./6., 1./6., 1./6. },
	   { 1./6., 1./6., 0.5,  },
	   { 1./6., 0.5,   1./6. },
	   { 0.5,   1./6., 1./6. } };

    break;
  case 4:
    // Assign weights
    // FIXME: Find new rule to avoid negative weight
    _w = { -0.0789333333333330,
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
    _p = { { 0.2500000000000000, 0.2500000000000000, 0.2500000000000000 },
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
    _w = { 0.0734930431163618,
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
    _p = { { 0.0927352503108910, 0.0927352503108910, 0.0927352503108910 },
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
    _w = { 0.0399227502581678,
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
    _p = { { 0.2146028712591520, 0.2146028712591520, 0.2146028712591520 },
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
}
//-----------------------------------------------------------------------------
Eigen::MatrixXd SimplexQuadrature::Chebyshev_Vandermonde_matrix
(const std::pair<std::vector<double>, std::vector<double>>& qr,
 std::size_t gdim,
 std::size_t N)
{
  // Create the Chebyshev basis matrix for each dimension separately
  std::vector<std::vector<Eigen::VectorXd>> T(gdim);
  Eigen::VectorXd x(qr.second.size());

  for (std::size_t d = 0; d < gdim; ++d)
  {
    // Extract coordinates in one dimension
    for (std::size_t i = 0; i < qr.second.size(); ++i)
      x(i) = qr.first[i*gdim + d];

    // Map points to [-1, 1]
    const double xmin = x.minCoeff();
    const double xmax = x.maxCoeff();
    const double hx = xmax - xmin;
    const Eigen::VectorXd xmap = (2./hx) * x - ((xmin+xmax)/hx) * Eigen::VectorXd::Ones(x.size());

    // Evaluate the basis
    T[d] = Chebyshev_polynomial(xmap, N);
  }

  // Find the order of the polynomials in graded lexicographic
  // ordering
  const std::vector<std::vector<std::size_t>> P = grlex(gdim, N);

  // Setup the Vandermonde type matrix
  const std::size_t n_cols = P.size();
  Eigen::MatrixXd V(Eigen::MatrixXd::Ones(qr.second.size(), n_cols));

  // The first column is always [1, 1, ..., 1], hence we can start from 1
  for (std::size_t i = 1; i < n_cols; ++i)
  {
    // Pick out the correct order of polynomial from the P
    // matrix. Start with dimension 0.
    const std::size_t d = 0;
    const std::size_t grlex_order = P[i][d];
    Eigen::VectorXd V_col = T[d][grlex_order];

    // Do a .* style multiplication for each other dimension
    for (std::size_t d = 1; d < gdim; ++d)
    {
      const std::size_t grlex_order = P[i][d];
      V_col = V_col.cwiseProduct(T[d][grlex_order]);
    }
    V.col(i) = V_col;
  }

  return V;
}

//-----------------------------------------------------------------------------
std::vector<Eigen::VectorXd>
SimplexQuadrature::Chebyshev_polynomial(const Eigen::VectorXd& x,
					std::size_t N)
{
  // Create Chebyshev polynomial of the first kind of order N on [-1, 1]
  //
  // T_0(x) = 1
  // T_1(x) = x
  // T_{k}(x) = 2 * x * T_{k-1}(x) - T_{k-2}(x), k = 2, ..., N

  // Store in a matrix T such that (T)_ij = T_i(x_j). We don't use
  // Eigen::MatrixXd, because we want to slice out the rows later.
  std::vector<Eigen::VectorXd> T(N + 1, Eigen::VectorXd(x.size()));

  // Reccurence construction
  T[0] = Eigen::VectorXd::Ones(x.size());

  if (N >= 1)
  {
    T[1] = x;
    for (std::size_t k = 2; k <= N; ++k)
      T[k] = 2*x.cwiseProduct(T[k-1]) - T[k-2];
  }

  return T;
}

//-----------------------------------------------------------------------------
std::vector<std::vector<std::size_t>>
SimplexQuadrature::grlex(std::size_t gdim,
			 std::size_t N)
{
  // Generate a matrix with numbers in graded lexicographic ordering,
  // i.e. if N = 3 and dim = 2, P should be
  // P = [ 0 0
  //       0 1
  //       1 0
  //       0 2
  //       1 1
  //       2 0
  //       0 3
  //       1 2
  //       2 1
  //       3 0 ]

  const std::size_t n_rows = choose(N + gdim, gdim);
  std::vector<std::vector<std::size_t>> P(n_rows, std::vector<std::size_t>(gdim));

  // FIXME: Make this a dimension independent loop
  switch (gdim)
  {
  case 2:
    for (std::size_t sum = 0, row = 0; sum <= N; ++sum)
      for (std::size_t xi = 0; xi <= N; ++xi)
	for (std::size_t yi = 0; yi <= N; ++yi)
	  if (xi + yi == sum)
	  {
	    P[row][0] = xi;
	    P[row][1] = yi;
	    row++;
	  }
    break;
  case 3:
    for (std::size_t sum = 0, row = 0; sum <= N; ++sum)
      for (std::size_t xi = 0; xi <= N; ++xi)
	for (std::size_t yi = 0; yi <= N; ++yi)
	  for (std::size_t zi = 0; zi <= N; ++zi)
	    if (xi + yi + zi == sum)
	    {
	      P[row][0] = xi;
	      P[row][1] = yi;
	      P[row][2] = zi;
	      row++;
	    }
    break;
  default:
    dolfin_assert(false);
  }

  return P;
}
//-----------------------------------------------------------------------------
std::size_t
SimplexQuadrature::choose(std::size_t n,
			  std::size_t k)
{
  // Compute the number of combinations n over k
  if (k == 0)
    return 1;
  return (n * choose(n - 1, k - 1)) / k;
}
//-----------------------------------------------------------------------------
void SimplexQuadrature::dunavant_rule(std::size_t rule,
				      std::vector<std::vector<double> >& p,
				      std::vector<double>& w)
{
  //****************************************************************************80
  //
  //  Purpose:
  //
  //    DUNAVANT_RULE returns the points and weights of a Dunavant rule.
  //
  //  Licensing:
  //
  //    This code is distributed under the GNU LGPL license.
  //
  //  Modified:
  //
  //    11 December 2006
  //
  //  Author:
  //
  //    John Burkardt
  //
  //  Reference:
  //
  //    David Dunavant,
  //    High Degree Efficient Symmetrical Gaussian Quadrature Rules
  //    for the Triangle,
  //    International Journal for Numerical Methods in Engineering,
  //    Volume 21, 1985, pages 1129-1148.
  //
  //    James Lyness, Dennis Jespersen,
  //    Moderate Degree Symmetric Quadrature Rules for the Triangle,
  //    Journal of the Institute of Mathematics and its Applications,
  //    Volume 15, Number 1, February 1975, pages 19-32.
  //
  //  Parameters:
  //
  //    Input, int RULE, the index of the rule.
  //
  //    Input, int ORDER_NUM, the order (number of points) of the rule.
  //
  //    Output, double XY[2*ORDER_NUM], the points of the rule.
  //
  //    Output, double W[ORDER_NUM], the weights of the rule.
  //

  // Get the suborder information
  const std::size_t suborder_num = dunavant_suborder_num(rule);
  std::vector<double> suborder_xyz(3*suborder_num);
  std::vector<double> suborder_w(suborder_num);;
  const std::vector<std::size_t> suborder = dunavant_suborder(rule, suborder_num);
  dunavant_subrule(rule, suborder_num, suborder_xyz, suborder_w);

  // Resize p and w
  const std::size_t order_num = dunavant_order_num(rule);
  p.resize(order_num, std::vector<double>(2));
  w.resize(order_num);

  // Expand the suborder information to a full order rule
  std::size_t o = 0;

  for (std::size_t s = 0; s < suborder_num; s++)
  {
    if (suborder[s] == 1)
    {
      p[o][0] = suborder_xyz[0+s*3];
      p[o][1] = suborder_xyz[1+s*3];
      w[o] = suborder_w[s];
      o = o + 1;
    }
    else if (suborder[s] == 3)
    {
      for (std::size_t k = 0; k < 3; k++)
      {
        p[o][0] = suborder_xyz[i4_wrap(k,  0,2) + s*3];
        p[o][1] = suborder_xyz[i4_wrap(k+1,0,2) + s*3];
        w[o] = suborder_w[s];
        o = o + 1;
      }
    }
    else if (suborder[s] == 6)
    {
      for (std::size_t k = 0; k < 3; k++)
      {
        p[o][0] = suborder_xyz[i4_wrap(k,  0,2) + s*3];
        p[o][1] = suborder_xyz[i4_wrap(k+1,0,2) + s*3];
        w[o] = suborder_w[s];
        o = o + 1;
      }

      for (std::size_t k = 0; k < 3; k++)
      {
        p[o][0] = suborder_xyz[i4_wrap(k+1,0,2) + s*3];
        p[o][1] = suborder_xyz[i4_wrap(k,  0,2) + s*3];
        w[o] = suborder_w[s];
        o = o + 1;
      }
    }
    else
    {
      dolfin_error("SimplexQuadrature.cpp",
		   "compute quadrature rule for triangle",
		   "Dunavant rule not implemented for suborder ", suborder[s]);
    }
  }

}
//****************************************************************************80

std::size_t SimplexQuadrature::dunavant_order_num(std::size_t rule)

//****************************************************************************80
//
//  Purpose:
//
//    DUNAVANT_ORDER_NUM returns the order of a Dunavant rule for the triangle.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    11 December 2006
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    David Dunavant,
//    High Degree Efficient Symmetrical Gaussian Quadrature Rules
//    for the Triangle,
//    International Journal for Numerical Methods in Engineering,
//    Volume 21, 1985, pages 1129-1148.
//
//    James Lyness, Dennis Jespersen,
//    Moderate Degree Symmetric Quadrature Rules for the Triangle,
//    Journal of the Institute of Mathematics and its Applications,
//    Volume 15, Number 1, February 1975, pages 19-32.
//
//  Parameters:
//
//    Input, int RULE, the index of the rule.
//
//    Output, int DUNAVANT_ORDER_NUM, the order (number of points) of the rule.
//
{
  std::size_t order;
  std::size_t order_num;
  std::size_t suborder_num;

  suborder_num = dunavant_suborder_num(rule);

  std::vector<std::size_t> suborder = dunavant_suborder(rule, suborder_num);

  order_num = 0;
  for (order = 0; order < suborder_num; order++)
  {
    order_num = order_num + suborder[order];
  }

  return order_num;
}
//****************************************************************************80

std::vector<std::size_t> SimplexQuadrature::dunavant_suborder(int rule, int suborder_num)

//****************************************************************************80
//
//  Purpose:
//
//    DUNAVANT_SUBORDER returns the suborders for a Dunavant rule.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    11 December 2006
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    David Dunavant,
//    High Degree Efficient Symmetrical Gaussian Quadrature Rules
//    for the Triangle,
//    International Journal for Numerical Methods in Engineering,
//    Volume 21, 1985, pages 1129-1148.
//
//    James Lyness, Dennis Jespersen,
//    Moderate Degree Symmetric Quadrature Rules for the Triangle,
//    Journal of the Institute of Mathematics and its Applications,
//    Volume 15, Number 1, February 1975, pages 19-32.
//
//  Parameters:
//
//    Input, int RULE, the index of the rule.
//
//    Input, int SUBORDER_NUM, the number of suborders of the rule.
//
//    Output, int DUNAVANT_SUBORDER[SUBORDER_NUM], the suborders of the rule.
//
{
  std::vector<std::size_t> suborder(suborder_num);

  if (rule == 1)
  {
    suborder[0] = 1;
  }
  else if (rule == 2)
  {
    suborder[0] = 3;
  }
  else if (rule == 3)
  {
    suborder[0] = 1;
    suborder[1] = 3;
  }
  else if (rule == 4)
  {
    suborder[0] = 3;
    suborder[1] = 3;
  }
  else if (rule == 5)
  {
    suborder[0] = 1;
    suborder[1] = 3;
    suborder[2] = 3;
  }
  else if (rule == 6)
  {
    suborder[0] = 3;
    suborder[1] = 3;
    suborder[2] = 6;
  }
  else if (rule == 7)
  {
    suborder[0] = 1;
    suborder[1] = 3;
    suborder[2] = 3;
    suborder[3] = 6;
  }
  else if (rule == 8)
  {
    suborder[0] = 1;
    suborder[1] = 3;
    suborder[2] = 3;
    suborder[3] = 3;
    suborder[4] = 6;
  }
  else if (rule == 9)
  {
    suborder[0] = 1;
    suborder[1] = 3;
    suborder[2] = 3;
    suborder[3] = 3;
    suborder[4] = 3;
    suborder[5] = 6;
  }
  else if (rule == 10)
  {
    suborder[0] = 1;
    suborder[1] = 3;
    suborder[2] = 3;
    suborder[3] = 6;
    suborder[4] = 6;
    suborder[5] = 6;
  }
  else if (rule == 11)
  {
    suborder[0] = 3;
    suborder[1] = 3;
    suborder[2] = 3;
    suborder[3] = 3;
    suborder[4] = 3;
    suborder[5] = 6;
    suborder[6] = 6;
  }
  else if (rule == 12)
  {
    suborder[0] = 3;
    suborder[1] = 3;
    suborder[2] = 3;
    suborder[3] = 3;
    suborder[4] = 3;
    suborder[5] = 6;
    suborder[6] = 6;
    suborder[7] = 6;
  }
  else if (rule == 13)
  {
    suborder[0] = 1;
    suborder[1] = 3;
    suborder[2] = 3;
    suborder[3] = 3;
    suborder[4] = 3;
    suborder[5] = 3;
    suborder[6] = 3;
    suborder[7] = 6;
    suborder[8] = 6;
    suborder[9] = 6;
  }
  else if (rule == 14)
  {
    suborder[0] = 3;
    suborder[1] = 3;
    suborder[2] = 3;
    suborder[3] = 3;
    suborder[4] = 3;
    suborder[5] = 3;
    suborder[6] = 6;
    suborder[7] = 6;
    suborder[8] = 6;
    suborder[9] = 6;
  }
  else if (rule == 15)
  {
    suborder[0] = 3;
    suborder[1] = 3;
    suborder[2] = 3;
    suborder[3] = 3;
    suborder[4] = 3;
    suborder[5] = 3;
    suborder[6] = 6;
    suborder[7] = 6;
    suborder[8] = 6;
    suborder[9] = 6;
    suborder[10] = 6;
  }
  else if (rule == 16)
  {
    suborder[0] = 1;
    suborder[1] = 3;
    suborder[2] = 3;
    suborder[3] = 3;
    suborder[4] = 3;
    suborder[5] = 3;
    suborder[6] = 3;
    suborder[7] = 3;
    suborder[8] = 6;
    suborder[9] = 6;
    suborder[10] = 6;
    suborder[11] = 6;
    suborder[12] = 6;
  }
  else if (rule == 17)
  {
    suborder[0] = 1;
    suborder[1] = 3;
    suborder[2] = 3;
    suborder[3] = 3;
    suborder[4] = 3;
    suborder[5] = 3;
    suborder[6] = 3;
    suborder[7] = 3;
    suborder[8] = 3;
    suborder[9] = 6;
    suborder[10] = 6;
    suborder[11] = 6;
    suborder[12] = 6;
    suborder[13] = 6;
    suborder[14] = 6;
  }
  else if (rule == 18)
  {
    suborder[0] = 1;
    suborder[1] = 3;
    suborder[2] = 3;
    suborder[3] = 3;
    suborder[4] = 3;
    suborder[5] = 3;
    suborder[6] = 3;
    suborder[7] = 3;
    suborder[8] = 3;
    suborder[9] = 3;
    suborder[10] = 6;
    suborder[11] = 6;
    suborder[12] = 6;
    suborder[13] = 6;
    suborder[14] = 6;
    suborder[15] = 6;
    suborder[16] = 6;
  }
  else if (rule == 19)
  {
    suborder[0] = 1;
    suborder[1] = 3;
    suborder[2] = 3;
    suborder[3] = 3;
    suborder[4] = 3;
    suborder[5] = 3;
    suborder[6] = 3;
    suborder[7] = 3;
    suborder[8] = 3;
    suborder[9] = 6;
    suborder[10] = 6;
    suborder[11] = 6;
    suborder[12] = 6;
    suborder[13] = 6;
    suborder[14] = 6;
    suborder[15] = 6;
    suborder[16] = 6;
  }
  else if (rule == 20)
  {
    suborder[0] = 1;
    suborder[1] = 3;
    suborder[2] = 3;
    suborder[3] = 3;
    suborder[4] = 3;
    suborder[5] = 3;
    suborder[6] = 3;
    suborder[7] = 3;
    suborder[8] = 3;
    suborder[9] = 3;
    suborder[10] = 3;
    suborder[11] = 6;
    suborder[12] = 6;
    suborder[13] = 6;
    suborder[14] = 6;
    suborder[15] = 6;
    suborder[16] = 6;
    suborder[17] = 6;
    suborder[18] = 6;
  }
  else
  {
    dolfin_error("SimplexQuadrature.cpp",
                 "compute quadrature rule for triangle",
                 "dunavant_suborder not implemented for rule ", rule);
  }

  return suborder;
}
//****************************************************************************80

std::size_t SimplexQuadrature::dunavant_suborder_num(int rule)

//****************************************************************************80
//
//  Purpose:
//
//    DUNAVANT_SUBORDER_NUM returns the number of suborders for a Dunavant rule.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    11 December 2006
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    David Dunavant,
//    High Degree Efficient Symmetrical Gaussian Quadrature Rules
//    for the Triangle,
//    International Journal for Numerical Methods in Engineering,
//    Volume 21, 1985, pages 1129-1148.
//
//    James Lyness, Dennis Jespersen,
//    Moderate Degree Symmetric Quadrature Rules for the Triangle,
//    Journal of the Institute of Mathematics and its Applications,
//    Volume 15, Number 1, February 1975, pages 19-32.
//
//  Parameters:
//
//    Input, int RULE, the index of the rule.
//
//    Output, int DUNAVANT_SUBORDER_NUM, the number of suborders of the rule.
//
{
  std::size_t suborder_num;

  if (rule == 1)
  {
    suborder_num = 1;
  }
  else if (rule == 2)
  {
    suborder_num = 1;
  }
  else if (rule == 3)
  {
    suborder_num = 2;
  }
  else if (rule == 4)
  {
    suborder_num = 2;
  }
  else if (rule == 5)
  {
    suborder_num = 3;
  }
  else if (rule == 6)
  {
    suborder_num = 3;
  }
  else if (rule == 7)
  {
    suborder_num = 4;
  }
  else if (rule == 8)
  {
    suborder_num = 5;
  }
  else if (rule == 9)
  {
    suborder_num = 6;
  }
  else if (rule == 10)
  {
    suborder_num = 6;
  }
  else if (rule == 11)
  {
    suborder_num = 7;
  }
  else if (rule == 12)
  {
    suborder_num = 8;
  }
  else if (rule == 13)
  {
    suborder_num = 10;
  }
  else if (rule == 14)
  {
    suborder_num = 10;
  }
  else if (rule == 15)
  {
    suborder_num = 11;
  }
  else if (rule == 16)
  {
    suborder_num = 13;
  }
  else if (rule == 17)
  {
    suborder_num = 15;
  }
  else if (rule == 18)
  {
    suborder_num = 17;
  }
  else if (rule == 19)
  {
    suborder_num = 17;
  }
  else if (rule == 20)
  {
    suborder_num = 19;
  }
  else
  {
    dolfin_error("SimplexQuadrature.cpp",
                 "compute quadrature rule for triangle",
                 "dunavant_suborder_num not implemented for rule ", rule);
  }

  return suborder_num;
}
//****************************************************************************80

void SimplexQuadrature::dunavant_subrule(std::size_t rule,
					 std::size_t suborder_num,
					 std::vector<double>& suborder_xyz,
					 std::vector<double>& suborder_w)

//****************************************************************************80
//
//  Purpose:
//
//    DUNAVANT_SUBRULE returns a compressed Dunavant rule.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    11 December 2006
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    David Dunavant,
//    High Degree Efficient Symmetrical Gaussian Quadrature Rules
//    for the Triangle,
//    International Journal for Numerical Methods in Engineering,
//    Volume 21, 1985, pages 1129-1148.
//
//    James Lyness, Dennis Jespersen,
//    Moderate Degree Symmetric Quadrature Rules for the Triangle,
//    Journal of the Institute of Mathematics and its Applications,
//    Volume 15, Number 1, February 1975, pages 19-32.
//
//  Parameters:
//
//    Input, int RULE, the index of the rule.
//
//    Input, int SUBORDER_NUM, the number of suborders of the rule.
//
//    Output, double SUBORDER_XYZ[3*SUBORDER_NUM],
//    the barycentric coordinates of the abscissas.
//
//    Output, double SUBORDER_W[SUBORDER_NUM], the suborder weights.
//
{
  if (rule == 1)
  {
    dunavant_subrule_01(suborder_num, suborder_xyz, suborder_w);
  }
  else if (rule == 2)
  {
    dunavant_subrule_02(suborder_num, suborder_xyz, suborder_w);
  }
  else if (rule == 3)
  {
    dunavant_subrule_03(suborder_num, suborder_xyz, suborder_w);
  }
  else if (rule == 4)
  {
    dunavant_subrule_04(suborder_num, suborder_xyz, suborder_w);
  }
  else if (rule == 5)
  {
    dunavant_subrule_05(suborder_num, suborder_xyz, suborder_w);
  }
  else if (rule == 6)
  {
    dunavant_subrule_06(suborder_num, suborder_xyz, suborder_w);
  }
  else if (rule == 7)
  {
    dunavant_subrule_07(suborder_num, suborder_xyz, suborder_w);
  }
  else if (rule == 8)
  {
    dunavant_subrule_08(suborder_num, suborder_xyz, suborder_w);
  }
  else if (rule == 9)
  {
    dunavant_subrule_09(suborder_num, suborder_xyz, suborder_w);
  }
  else if (rule == 10)
  {
    dunavant_subrule_10(suborder_num, suborder_xyz, suborder_w);
  }
  else if (rule == 11)
  {
    dunavant_subrule_11(suborder_num, suborder_xyz, suborder_w);
  }
  else if (rule == 12)
  {
    dunavant_subrule_12(suborder_num, suborder_xyz, suborder_w);
  }
  else if (rule == 13)
  {
    dunavant_subrule_13(suborder_num, suborder_xyz, suborder_w);
  }
  else if (rule == 14)
  {
    dunavant_subrule_14(suborder_num, suborder_xyz, suborder_w);
  }
  else if (rule == 15)
  {
    dunavant_subrule_15(suborder_num, suborder_xyz, suborder_w);
  }
  else if (rule == 16)
  {
    dunavant_subrule_16(suborder_num, suborder_xyz, suborder_w);
  }
  else if (rule == 17)
  {
    dunavant_subrule_17(suborder_num, suborder_xyz, suborder_w);
  }
  else if (rule == 18)
  {
    dunavant_subrule_18(suborder_num, suborder_xyz, suborder_w);
  }
  else if (rule == 19)
  {
    dunavant_subrule_19(suborder_num, suborder_xyz, suborder_w);
  }
  else if (rule == 20)
  {
    dunavant_subrule_20(suborder_num, suborder_xyz, suborder_w);
  }
  else
  {
    dolfin_error("SimplexQuadrature.cpp",
                 "compute quadrature rule for triangle",
                 "dunavant_subrule not implemented for rule ", rule);
  }

  return;
}

//-----------------------------------------------------------------------------
void SimplexQuadrature::dunavant_subrule_01(int suborder_num, std::vector<double>& suborder_xyz,
					    std::vector<double>& suborder_w)

//****************************************************************************80
//
//  Purpose:
//
//    DUNAVANT_SUBRULE_01 returns a compressed Dunavant rule 1.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    11 December 2006
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    David Dunavant,
//    High Degree Efficient Symmetrical Gaussian Quadrature Rules
//    for the Triangle,
//    International Journal for Numerical Methods in Engineering,
//    Volume 21, 1985, pages 1129-1148.
//
//    James Lyness, Dennis Jespersen,
//    Moderate Degree Symmetric Quadrature Rules for the Triangle,
//    Journal of the Institute of Mathematics and its Applications,
//    Volume 15, Number 1, February 1975, pages 19-32.
//
//  Parameters:
//
//    Input, int SUBORDER_NUM, the number of suborders of the rule.
//
//    Output, double SUBORDER_XYZ[3*SUBORDER_NUM],
//    the barycentric coordinates of the abscissas.
//
//    Output, double SUBORDER_W[SUBORDER_NUM], the suborder weights.
//
{
  int s;
  double suborder_xy_rule_01[3*1] = {
    0.333333333333333,  0.333333333333333, 0.333333333333333
  };
  double suborder_w_rule_01[1] = {
    1.000000000000000
  };

  for (s = 0; s < suborder_num; s++)
  {
    suborder_xyz[0+s*3] = suborder_xy_rule_01[0+s*3];
    suborder_xyz[1+s*3] = suborder_xy_rule_01[1+s*3];
    suborder_xyz[2+s*3] = suborder_xy_rule_01[2+s*3];
  }

  for (s = 0; s < suborder_num; s++)
  {
    suborder_w[s] = suborder_w_rule_01[s];
  }

  return;
}
//****************************************************************************80

void SimplexQuadrature::dunavant_subrule_02(int suborder_num, std::vector<double>& suborder_xyz,
					    std::vector<double>& suborder_w)

//****************************************************************************80
//
//  Purpose:
//
//    DUNAVANT_SUBRULE_02 returns a compressed Dunavant rule 2.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    11 December 2006
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    David Dunavant,
//    High Degree Efficient Symmetrical Gaussian Quadrature Rules
//    for the Triangle,
//    International Journal for Numerical Methods in Engineering,
//    Volume 21, 1985, pages 1129-1148.
//
//    James Lyness, Dennis Jespersen,
//    Moderate Degree Symmetric Quadrature Rules for the Triangle,
//    Journal of the Institute of Mathematics and its Applications,
//    Volume 15, Number 1, February 1975, pages 19-32.
//
//  Parameters:
//
//    Input, int SUBORDER_NUM, the number of suborders of the rule.
//
//    Output, double SUBORDER_XYZ[3*SUBORDER_NUM],
//    the barycentric coordinates of the abscissas.
//
//    Output, double SUBORDER_W[SUBORDER_NUM], the suborder weights.
//
{
  int s;
  double suborder_xy_rule_02[3*1] = {
    0.666666666666667, 0.166666666666667, 0.166666666666667
  };
  double suborder_w_rule_02[1] = {
    0.333333333333333
  };

  for (s = 0; s < suborder_num; s++)
  {
    suborder_xyz[0+s*3] = suborder_xy_rule_02[0+s*3];
    suborder_xyz[1+s*3] = suborder_xy_rule_02[1+s*3];
    suborder_xyz[2+s*3] = suborder_xy_rule_02[2+s*3];
  }

  for (s = 0; s < suborder_num; s++)
  {
    suborder_w[s] = suborder_w_rule_02[s];
  }

  return;
}
//****************************************************************************80

void SimplexQuadrature::dunavant_subrule_03(int suborder_num, std::vector<double>& suborder_xyz,
					    std::vector<double>& suborder_w)

//****************************************************************************80
//
//  Purpose:
//
//    DUNAVANT_SUBRULE_03 returns a compressed Dunavant rule 3.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    11 December 2006
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    David Dunavant,
//    High Degree Efficient Symmetrical Gaussian Quadrature Rules
//    for the Triangle,
//    International Journal for Numerical Methods in Engineering,
//    Volume 21, 1985, pages 1129-1148.
//
//    James Lyness, Dennis Jespersen,
//    Moderate Degree Symmetric Quadrature Rules for the Triangle,
//    Journal of the Institute of Mathematics and its Applications,
//    Volume 15, Number 1, February 1975, pages 19-32.
//
//  Parameters:
//
//    Input, int SUBORDER_NUM, the number of suborders of the rule.
//
//    Output, double SUBORDER_XYZ[3*SUBORDER_NUM],
//    the barycentric coordinates of the abscissas.
//
//    Output, double SUBORDER_W[SUBORDER_NUM], the suborder weights.
//
{
  int s;
  double suborder_xy_rule_03[3*2] = {
    0.333333333333333, 0.333333333333333, 0.333333333333333,
    0.600000000000000, 0.200000000000000, 0.200000000000000
  };
  double suborder_w_rule_03[2] = {
    -0.562500000000000,
    0.520833333333333
  };

  for (s = 0; s < suborder_num; s++)
  {
    suborder_xyz[0+s*3] = suborder_xy_rule_03[0+s*3];
    suborder_xyz[1+s*3] = suborder_xy_rule_03[1+s*3];
    suborder_xyz[2+s*3] = suborder_xy_rule_03[2+s*3];
  }

  for (s = 0; s < suborder_num; s++)
  {
    suborder_w[s] = suborder_w_rule_03[s];
  }

  return;
}
//****************************************************************************80

void SimplexQuadrature::dunavant_subrule_04(int suborder_num, std::vector<double>& suborder_xyz,
					    std::vector<double>& suborder_w)

//****************************************************************************80
//
//  Purpose:
//
//    DUNAVANT_SUBRULE_04 returns a compressed Dunavant rule 4.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    11 December 2006
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    David Dunavant,
//    High Degree Efficient Symmetrical Gaussian Quadrature Rules
//    for the Triangle,
//    International Journal for Numerical Methods in Engineering,
//    Volume 21, 1985, pages 1129-1148.
//
//    James Lyness, Dennis Jespersen,
//    Moderate Degree Symmetric Quadrature Rules for the Triangle,
//    Journal of the Institute of Mathematics and its Applications,
//    Volume 15, Number 1, February 1975, pages 19-32.
//
//  Parameters:
//
//    Input, int SUBORDER_NUM, the number of suborders of the rule.
//
//    Output, double SUBORDER_XYZ[3*SUBORDER_NUM],
//    the barycentric coordinates of the abscissas.
//
//    Output, double SUBORDER_W[SUBORDER_NUM], the suborder weights.
//
{
  int s;
  double suborder_xy_rule_04[3*2] = {
    0.108103018168070, 0.445948490915965, 0.445948490915965,
    0.816847572980459, 0.091576213509771, 0.091576213509771
  };
  double suborder_w_rule_04[2] = {
    0.223381589678011,
    0.109951743655322
  };

  for (s = 0; s < suborder_num; s++)
  {
    suborder_xyz[0+s*3] = suborder_xy_rule_04[0+s*3];
    suborder_xyz[1+s*3] = suborder_xy_rule_04[1+s*3];
    suborder_xyz[2+s*3] = suborder_xy_rule_04[2+s*3];
  }

  for (s = 0; s < suborder_num; s++)
  {
    suborder_w[s] = suborder_w_rule_04[s];
  }

  return;
}
//****************************************************************************80

void SimplexQuadrature::dunavant_subrule_05(int suborder_num, std::vector<double>& suborder_xyz,
					    std::vector<double>& suborder_w)

//****************************************************************************80
//
//  Purpose:
//
//    DUNAVANT_SUBRULE_05 returns a compressed Dunavant rule 5.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    11 December 2006
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    David Dunavant,
//    High Degree Efficient Symmetrical Gaussian Quadrature Rules
//    for the Triangle,
//    International Journal for Numerical Methods in Engineering,
//    Volume 21, 1985, pages 1129-1148.
//
//    James Lyness, Dennis Jespersen,
//    Moderate Degree Symmetric Quadrature Rules for the Triangle,
//    Journal of the Institute of Mathematics and its Applications,
//    Volume 15, Number 1, February 1975, pages 19-32.
//
//  Parameters:
//
//    Input, int SUBORDER_NUM, the number of suborders of the rule.
//
//    Output, double SUBORDER_XYZ[3*SUBORDER_NUM],
//    the barycentric coordinates of the abscissas.
//
//    Output, double SUBORDER_W[SUBORDER_NUM], the suborder weights.
//
{
  int s;
  double suborder_xy_rule_05[3*3] = {
    0.333333333333333, 0.333333333333333, 0.333333333333333,
    0.059715871789770, 0.470142064105115, 0.470142064105115,
    0.797426985353087, 0.101286507323456, 0.101286507323456
  };
  double suborder_w_rule_05[3] = {
    0.225000000000000,
    0.132394152788506,
    0.125939180544827
  };

  for (s = 0; s < suborder_num; s++)
  {
    suborder_xyz[0+s*3] = suborder_xy_rule_05[0+s*3];
    suborder_xyz[1+s*3] = suborder_xy_rule_05[1+s*3];
    suborder_xyz[2+s*3] = suborder_xy_rule_05[2+s*3];
  }

  for (s = 0; s < suborder_num; s++)
  {
    suborder_w[s] = suborder_w_rule_05[s];
  }

  return;
}
//****************************************************************************80

void SimplexQuadrature::dunavant_subrule_06(int suborder_num, std::vector<double>& suborder_xyz,
					    std::vector<double>& suborder_w)

//****************************************************************************80
//
//  Purpose:
//
//    DUNAVANT_SUBRULE_06 returns a compressed Dunavant rule 6.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    11 December 2006
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    David Dunavant,
//    High Degree Efficient Symmetrical Gaussian Quadrature Rules
//    for the Triangle,
//    International Journal for Numerical Methods in Engineering,
//    Volume 21, 1985, pages 1129-1148.
//
//    James Lyness, Dennis Jespersen,
//    Moderate Degree Symmetric Quadrature Rules for the Triangle,
//    Journal of the Institute of Mathematics and its Applications,
//    Volume 15, Number 1, February 1975, pages 19-32.
//
//  Parameters:
//
//    Input, int SUBORDER_NUM, the number of suborders of the rule.
//
//    Output, double SUBORDER_XYZ[3*SUBORDER_NUM],
//    the barycentric coordinates of the abscissas.
//
//    Output, double SUBORDER_W[SUBORDER_NUM], the suborder weights.
//
{
  int s;
  double suborder_xy_rule_06[3*3] = {
    0.501426509658179, 0.249286745170910, 0.249286745170910,
    0.873821971016996, 0.063089014491502, 0.063089014491502,
    0.053145049844817, 0.310352451033784, 0.636502499121399
  };
  double suborder_w_rule_06[3] = {
    0.116786275726379,
    0.050844906370207,
    0.082851075618374
  };

  for (s = 0; s < suborder_num; s++)
  {
    suborder_xyz[0+s*3] = suborder_xy_rule_06[0+s*3];
    suborder_xyz[1+s*3] = suborder_xy_rule_06[1+s*3];
    suborder_xyz[2+s*3] = suborder_xy_rule_06[2+s*3];
  }

  for (s = 0; s < suborder_num; s++)
  {
    suborder_w[s] = suborder_w_rule_06[s];
  }

  return;
}
//****************************************************************************80

void SimplexQuadrature::dunavant_subrule_07(int suborder_num, std::vector<double>& suborder_xyz,
					    std::vector<double>& suborder_w)

//****************************************************************************80
//
//  Purpose:
//
//    DUNAVANT_SUBRULE_07 returns a compressed Dunavant rule 7.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    11 December 2006
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    David Dunavant,
//    High Degree Efficient Symmetrical Gaussian Quadrature Rules
//    for the Triangle,
//    International Journal for Numerical Methods in Engineering,
//    Volume 21, 1985, pages 1129-1148.
//
//    James Lyness, Dennis Jespersen,
//    Moderate Degree Symmetric Quadrature Rules for the Triangle,
//    Journal of the Institute of Mathematics and its Applications,
//    Volume 15, Number 1, February 1975, pages 19-32.
//
//  Parameters:
//
//    Input, int SUBORDER_NUM, the number of suborders of the rule.
//
//    Output, double SUBORDER_XYZ[3*SUBORDER_NUM],
//    the barycentric coordinates of the abscissas.
//
//    Output, double SUBORDER_W[SUBORDER_NUM], the suborder weights.
//
{
  int s;
  double suborder_xy_rule_07[3*4] = {
    0.333333333333333, 0.333333333333333, 0.333333333333333,
    0.479308067841920, 0.260345966079040, 0.260345966079040,
    0.869739794195568, 0.065130102902216, 0.065130102902216,
    0.048690315425316, 0.312865496004874, 0.638444188569810
  };
  double suborder_w_rule_07[4] = {
    -0.149570044467682,
    0.175615257433208,
    0.053347235608838,
    0.077113760890257
  };

  for (s = 0; s < suborder_num; s++)
  {
    suborder_xyz[0+s*3] = suborder_xy_rule_07[0+s*3];
    suborder_xyz[1+s*3] = suborder_xy_rule_07[1+s*3];
    suborder_xyz[2+s*3] = suborder_xy_rule_07[2+s*3];
  }

  for (s = 0; s < suborder_num; s++)
  {
    suborder_w[s] = suborder_w_rule_07[s];
  }

  return;
}
//****************************************************************************80

void SimplexQuadrature::dunavant_subrule_08(int suborder_num, std::vector<double>& suborder_xyz,
					    std::vector<double>& suborder_w)

//****************************************************************************80
//
//  Purpose:
//
//    DUNAVANT_SUBRULE_08 returns a compressed Dunavant rule 8.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    11 December 2006
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    David Dunavant,
//    High Degree Efficient Symmetrical Gaussian Quadrature Rules
//    for the Triangle,
//    International Journal for Numerical Methods in Engineering,
//    Volume 21, 1985, pages 1129-1148.
//
//    James Lyness, Dennis Jespersen,
//    Moderate Degree Symmetric Quadrature Rules for the Triangle,
//    Journal of the Institute of Mathematics and its Applications,
//    Volume 15, Number 1, February 1975, pages 19-32.
//
//  Parameters:
//
//    Input, int SUBORDER_NUM, the number of suborders of the rule.
//
//    Output, double SUBORDER_XYZ[3*SUBORDER_NUM],
//    the barycentric coordinates of the abscissas.
//
//    Output, double SUBORDER_W[SUBORDER_NUM], the suborder weights.
//
{
  int s;
  double suborder_xy_rule_08[3*5] = {
    0.333333333333333, 0.333333333333333, 0.333333333333333,
    0.081414823414554, 0.459292588292723, 0.459292588292723,
    0.658861384496480, 0.170569307751760, 0.170569307751760,
    0.898905543365938, 0.050547228317031, 0.050547228317031,
    0.008394777409958, 0.263112829634638, 0.728492392955404
  };
  double suborder_w_rule_08[5] = {
    0.144315607677787,
    0.095091634267285,
    0.103217370534718,
    0.032458497623198,
    0.027230314174435
  };

  for (s = 0; s < suborder_num; s++)
  {
    suborder_xyz[0+s*3] = suborder_xy_rule_08[0+s*3];
    suborder_xyz[1+s*3] = suborder_xy_rule_08[1+s*3];
    suborder_xyz[2+s*3] = suborder_xy_rule_08[2+s*3];
  }

  for (s = 0; s < suborder_num; s++)
  {
    suborder_w[s] = suborder_w_rule_08[s];
  }

  return;
}
//****************************************************************************80

void SimplexQuadrature::dunavant_subrule_09(int suborder_num, std::vector<double>& suborder_xyz,
					    std::vector<double>& suborder_w)

//****************************************************************************80
//
//  Purpose:
//
//    DUNAVANT_SUBRULE_09 returns a compressed Dunavant rule 9.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    11 December 2006
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    David Dunavant,
//    High Degree Efficient Symmetrical Gaussian Quadrature Rules
//    for the Triangle,
//    International Journal for Numerical Methods in Engineering,
//    Volume 21, 1985, pages 1129-1148.
//
//    James Lyness, Dennis Jespersen,
//    Moderate Degree Symmetric Quadrature Rules for the Triangle,
//    Journal of the Institute of Mathematics and its Applications,
//    Volume 15, Number 1, February 1975, pages 19-32.
//
//  Parameters:
//
//    Input, int SUBORDER_NUM, the number of suborders of the rule.
//
//    Output, double SUBORDER_XYZ[3*SUBORDER_NUM],
//    the barycentric coordinates of the abscissas.
//
//    Output, double SUBORDER_W[SUBORDER_NUM], the suborder weights.
//
{
  int s;
  double suborder_xy_rule_09[3*6] = {
    0.333333333333333, 0.333333333333333, 0.333333333333333,
    0.020634961602525, 0.489682519198738, 0.489682519198738,
    0.125820817014127, 0.437089591492937, 0.437089591492937,
    0.623592928761935, 0.188203535619033, 0.188203535619033,
    0.910540973211095, 0.044729513394453, 0.044729513394453,
    0.036838412054736, 0.221962989160766, 0.741198598784498
  };
  double suborder_w_rule_09[6] = {
    0.097135796282799,
    0.031334700227139,
    0.077827541004774,
    0.079647738927210,
    0.025577675658698,
    0.043283539377289
  };

  for (s = 0; s < suborder_num; s++)
  {
    suborder_xyz[0+s*3] = suborder_xy_rule_09[0+s*3];
    suborder_xyz[1+s*3] = suborder_xy_rule_09[1+s*3];
    suborder_xyz[2+s*3] = suborder_xy_rule_09[2+s*3];
  }

  for (s = 0; s < suborder_num; s++)
  {
    suborder_w[s] = suborder_w_rule_09[s];
  }

  return;
}
//****************************************************************************80

void SimplexQuadrature::dunavant_subrule_10(int suborder_num, std::vector<double>& suborder_xyz,
					    std::vector<double>& suborder_w)

//****************************************************************************80
//
//  Purpose:
//
//    DUNAVANT_SUBRULE_10 returns a compressed Dunavant rule 10.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    11 December 2006
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    David Dunavant,
//    High Degree Efficient Symmetrical Gaussian Quadrature Rules
//    for the Triangle,
//    International Journal for Numerical Methods in Engineering,
//    Volume 21, 1985, pages 1129-1148.
//
//    James Lyness, Dennis Jespersen,
//    Moderate Degree Symmetric Quadrature Rules for the Triangle,
//    Journal of the Institute of Mathematics and its Applications,
//    Volume 15, Number 1, February 1975, pages 19-32.
//
//  Parameters:
//
//    Input, int SUBORDER_NUM, the number of suborders of the rule.
//
//    Output, double SUBORDER_XYZ[3*SUBORDER_NUM],
//    the barycentric coordinates of the abscissas.
//
//    Output, double SUBORDER_W[SUBORDER_NUM], the suborder weights.
//
{
  int s;
  double suborder_xy_rule_10[3*6] = {
    0.333333333333333, 0.333333333333333, 0.333333333333333,
    0.028844733232685, 0.485577633383657, 0.485577633383657,
    0.781036849029926, 0.109481575485037, 0.109481575485037,
    0.141707219414880, 0.307939838764121, 0.550352941820999,
    0.025003534762686, 0.246672560639903, 0.728323904597411,
    0.009540815400299, 0.066803251012200, 0.923655933587500
  };
  double suborder_w_rule_10[6] = {
    0.090817990382754,
    0.036725957756467,
    0.045321059435528,
    0.072757916845420,
    0.028327242531057,
    0.009421666963733
  };

  for (s = 0; s < suborder_num; s++)
  {
    suborder_xyz[0+s*3] = suborder_xy_rule_10[0+s*3];
    suborder_xyz[1+s*3] = suborder_xy_rule_10[1+s*3];
    suborder_xyz[2+s*3] = suborder_xy_rule_10[2+s*3];
  }

  for (s = 0; s < suborder_num; s++)
  {
    suborder_w[s] = suborder_w_rule_10[s];
  }

  return;
}
//****************************************************************************80

void SimplexQuadrature::dunavant_subrule_11(int suborder_num, std::vector<double>& suborder_xyz,
					    std::vector<double>& suborder_w)

//****************************************************************************80
//
//  Purpose:
//
//    DUNAVANT_SUBRULE_11 returns a compressed Dunavant rule 11.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    11 December 2006
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    David Dunavant,
//    High Degree Efficient Symmetrical Gaussian Quadrature Rules
//    for the Triangle,
//    International Journal for Numerical Methods in Engineering,
//    Volume 21, 1985, pages 1129-1148.
//
//    James Lyness, Dennis Jespersen,
//    Moderate Degree Symmetric Quadrature Rules for the Triangle,
//    Journal of the Institute of Mathematics and its Applications,
//    Volume 15, Number 1, February 1975, pages 19-32.
//
//  Parameters:
//
//    Input, int SUBORDER_NUM, the number of suborders of the rule.
//
//    Output, double SUBORDER_XYZ[3*SUBORDER_NUM],
//    the barycentric coordinates of the abscissas.
//
//    Output, double SUBORDER_W[SUBORDER_NUM], the suborder weights.
//
{
  int s;
  double suborder_xy_rule_11[3*7] = {
    -0.069222096541517, 0.534611048270758, 0.534611048270758,
    0.202061394068290, 0.398969302965855, 0.398969302965855,
    0.593380199137435, 0.203309900431282, 0.203309900431282,
    0.761298175434837, 0.119350912282581, 0.119350912282581,
    0.935270103777448, 0.032364948111276, 0.032364948111276,
    0.050178138310495, 0.356620648261293, 0.593201213428213,
    0.021022016536166, 0.171488980304042, 0.807489003159792
  };
  double suborder_w_rule_11[7] = {
    0.000927006328961,
    0.077149534914813,
    0.059322977380774,
    0.036184540503418,
    0.013659731002678,
    0.052337111962204,
    0.020707659639141
  };

  for (s = 0; s < suborder_num; s++)
  {
    suborder_xyz[0+s*3] = suborder_xy_rule_11[0+s*3];
    suborder_xyz[1+s*3] = suborder_xy_rule_11[1+s*3];
    suborder_xyz[2+s*3] = suborder_xy_rule_11[2+s*3];
  }

  for (s = 0; s < suborder_num; s++)
  {
    suborder_w[s] = suborder_w_rule_11[s];
  }

  return;
}
//****************************************************************************80

void SimplexQuadrature::dunavant_subrule_12(int suborder_num, std::vector<double>& suborder_xyz,
					    std::vector<double>& suborder_w)

//****************************************************************************80
//
//  Purpose:
//
//    DUNAVANT_SUBRULE_12 returns a compressed Dunavant rule 12.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    11 December 2006
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    David Dunavant,
//    High Degree Efficient Symmetrical Gaussian Quadrature Rules
//    for the Triangle,
//    International Journal for Numerical Methods in Engineering,
//    Volume 21, 1985, pages 1129-1148.
//
//    James Lyness, Dennis Jespersen,
//    Moderate Degree Symmetric Quadrature Rules for the Triangle,
//    Journal of the Institute of Mathematics and its Applications,
//    Volume 15, Number 1, February 1975, pages 19-32.
//
//  Parameters:
//
//    Input, int SUBORDER_NUM, the number of suborders of the rule.
//
//    Output, double SUBORDER_XYZ[3*SUBORDER_NUM],
//    the barycentric coordinates of the abscissas.
//
//    Output, double SUBORDER_W[SUBORDER_NUM], the suborder weights.
//
{
  int s;
  double suborder_xy_rule_12[3*8] = {
    0.023565220452390, 0.488217389773805, 0.488217389773805,
    0.120551215411079, 0.439724392294460, 0.439724392294460,
    0.457579229975768, 0.271210385012116, 0.271210385012116,
    0.744847708916828, 0.127576145541586, 0.127576145541586,
    0.957365299093579, 0.021317350453210, 0.021317350453210,
    0.115343494534698, 0.275713269685514, 0.608943235779788,
    0.022838332222257, 0.281325580989940, 0.695836086787803,
    0.025734050548330, 0.116251915907597, 0.858014033544073
  };
  double suborder_w_rule_12[8] = {
    0.025731066440455,
    0.043692544538038,
    0.062858224217885,
    0.034796112930709,
    0.006166261051559,
    0.040371557766381,
    0.022356773202303,
    0.017316231108659
  };

  for (s = 0; s < suborder_num; s++)
  {
    suborder_xyz[0+s*3] = suborder_xy_rule_12[0+s*3];
    suborder_xyz[1+s*3] = suborder_xy_rule_12[1+s*3];
    suborder_xyz[2+s*3] = suborder_xy_rule_12[2+s*3];
  }

  for (s = 0; s < suborder_num; s++)
  {
    suborder_w[s] = suborder_w_rule_12[s];
  }

  return;
}
//****************************************************************************80

void SimplexQuadrature::dunavant_subrule_13(int suborder_num, std::vector<double>& suborder_xyz,
					    std::vector<double>& suborder_w)

//****************************************************************************80
//
//  Purpose:
//
//    DUNAVANT_SUBRULE_13 returns a compressed Dunavant rule 13.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    11 December 2006
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    David Dunavant,
//    High Degree Efficient Symmetrical Gaussian Quadrature Rules
//    for the Triangle,
//    International Journal for Numerical Methods in Engineering,
//    Volume 21, 1985, pages 1129-1148.
//
//    James Lyness, Dennis Jespersen,
//    Moderate Degree Symmetric Quadrature Rules for the Triangle,
//    Journal of the Institute of Mathematics and its Applications,
//    Volume 15, Number 1, February 1975, pages 19-32.
//
//  Parameters:
//
//    Input, int SUBORDER_NUM, the number of suborders of the rule.
//
//    Output, double SUBORDER_XYZ[3*SUBORDER_NUM],
//    the barycentric coordinates of the abscissas.
//
//    Output, double SUBORDER_W[SUBORDER_NUM], the suborder weights.
//
{
  int s;
  double suborder_xy_rule_13[3*10] = {
    0.333333333333333, 0.333333333333333, 0.333333333333333,
    0.009903630120591, 0.495048184939705, 0.495048184939705,
    0.062566729780852, 0.468716635109574, 0.468716635109574,
    0.170957326397447, 0.414521336801277, 0.414521336801277,
    0.541200855914337, 0.229399572042831, 0.229399572042831,
    0.771151009607340, 0.114424495196330, 0.114424495196330,
    0.950377217273082, 0.024811391363459, 0.024811391363459,
    0.094853828379579, 0.268794997058761, 0.636351174561660,
    0.018100773278807, 0.291730066734288, 0.690169159986905,
    0.022233076674090, 0.126357385491669, 0.851409537834241
  };
  double suborder_w_rule_13[10] = {
    0.052520923400802,
    0.011280145209330,
    0.031423518362454,
    0.047072502504194,
    0.047363586536355,
    0.031167529045794,
    0.007975771465074,
    0.036848402728732,
    0.017401463303822,
    0.015521786839045
  };

  for (s = 0; s < suborder_num; s++)
  {
    suborder_xyz[0+s*3] = suborder_xy_rule_13[0+s*3];
    suborder_xyz[1+s*3] = suborder_xy_rule_13[1+s*3];
    suborder_xyz[2+s*3] = suborder_xy_rule_13[2+s*3];
  }

  for (s = 0; s < suborder_num; s++)
  {
    suborder_w[s] = suborder_w_rule_13[s];
  }

  return;
}
//****************************************************************************80

void SimplexQuadrature::dunavant_subrule_14(int suborder_num, std::vector<double>& suborder_xyz,
					    std::vector<double>& suborder_w)

//****************************************************************************80
//
//  Purpose:
//
//    DUNAVANT_SUBRULE_14 returns a compressed Dunavant rule 14.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    11 December 2006
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    David Dunavant,
//    High Degree Efficient Symmetrical Gaussian Quadrature Rules
//    for the Triangle,
//    International Journal for Numerical Methods in Engineering,
//    Volume 21, 1985, pages 1129-1148.
//
//    James Lyness, Dennis Jespersen,
//    Moderate Degree Symmetric Quadrature Rules for the Triangle,
//    Journal of the Institute of Mathematics and its Applications,
//    Volume 15, Number 1, February 1975, pages 19-32.
//
//  Parameters:
//
//    Input, int SUBORDER_NUM, the number of suborders of the rule.
//
//    Output, double SUBORDER_XYZ[3*SUBORDER_NUM],
//    the barycentric coordinates of the abscissas.
//
//    Output, double SUBORDER_W[SUBORDER_NUM], the suborder weights.
//
{
  int s;
  double suborder_xy_rule_14[3*10] = {
    0.022072179275643, 0.488963910362179, 0.488963910362179,
    0.164710561319092, 0.417644719340454, 0.417644719340454,
    0.453044943382323, 0.273477528308839, 0.273477528308839,
    0.645588935174913, 0.177205532412543, 0.177205532412543,
    0.876400233818255, 0.061799883090873, 0.061799883090873,
    0.961218077502598, 0.019390961248701, 0.019390961248701,
    0.057124757403648, 0.172266687821356, 0.770608554774996,
    0.092916249356972, 0.336861459796345, 0.570222290846683,
    0.014646950055654, 0.298372882136258, 0.686980167808088,
    0.001268330932872, 0.118974497696957, 0.879757171370171
  };
  double suborder_w_rule_14[10] = {
    0.021883581369429,
    0.032788353544125,
    0.051774104507292,
    0.042162588736993,
    0.014433699669777,
    0.004923403602400,
    0.024665753212564,
    0.038571510787061,
    0.014436308113534,
    0.005010228838501
  };

  for (s = 0; s < suborder_num; s++)
  {
    suborder_xyz[0+s*3] = suborder_xy_rule_14[0+s*3];
    suborder_xyz[1+s*3] = suborder_xy_rule_14[1+s*3];
    suborder_xyz[2+s*3] = suborder_xy_rule_14[2+s*3];
  }

  for (s = 0; s < suborder_num; s++)
  {
    suborder_w[s] = suborder_w_rule_14[s];
  }

  return;
}
//****************************************************************************80

void SimplexQuadrature::dunavant_subrule_15(int suborder_num, std::vector<double>& suborder_xyz,
					    std::vector<double>& suborder_w)

//****************************************************************************80
//
//  Purpose:
//
//    DUNAVANT_SUBRULE_15 returns a compressed Dunavant rule 15.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    11 December 2006
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    David Dunavant,
//    High Degree Efficient Symmetrical Gaussian Quadrature Rules
//    for the Triangle,
//    International Journal for Numerical Methods in Engineering,
//    Volume 21, 1985, pages 1129-1148.
//
//    James Lyness, Dennis Jespersen,
//    Moderate Degree Symmetric Quadrature Rules for the Triangle,
//    Journal of the Institute of Mathematics and its Applications,
//    Volume 15, Number 1, February 1975, pages 19-32.
//
//  Parameters:
//
//    Input, int SUBORDER_NUM, the number of suborders of the rule.
//
//    Output, double SUBORDER_XYZ[3*SUBORDER_NUM],
//    the barycentric coordinates of the abscissas.
//
//    Output, double SUBORDER_W[SUBORDER_NUM], the suborder weights.
//
{
  int s;
  double suborder_xy_rule_15[3*11] = {
    -0.013945833716486, 0.506972916858243, 0.506972916858243,
    0.137187291433955, 0.431406354283023, 0.431406354283023,
    0.444612710305711, 0.277693644847144, 0.277693644847144,
    0.747070217917492, 0.126464891041254, 0.126464891041254,
    0.858383228050628, 0.070808385974686, 0.070808385974686,
    0.962069659517853, 0.018965170241073, 0.018965170241073,
    0.133734161966621, 0.261311371140087, 0.604954466893291,
    0.036366677396917, 0.388046767090269, 0.575586555512814,
    -0.010174883126571, 0.285712220049916, 0.724462663076655,
    0.036843869875878, 0.215599664072284, 0.747556466051838,
    0.012459809331199, 0.103575616576386, 0.883964574092416
  };
  double suborder_w_rule_15[11] = {
    0.001916875642849,
    0.044249027271145,
    0.051186548718852,
    0.023687735870688,
    0.013289775690021,
    0.004748916608192,
    0.038550072599593,
    0.027215814320624,
    0.002182077366797,
    0.021505319847731,
    0.007673942631049
  };

  for (s = 0; s < suborder_num; s++)
  {
    suborder_xyz[0+s*3] = suborder_xy_rule_15[0+s*3];
    suborder_xyz[1+s*3] = suborder_xy_rule_15[1+s*3];
    suborder_xyz[2+s*3] = suborder_xy_rule_15[2+s*3];
  }

  for (s = 0; s < suborder_num; s++)
  {
    suborder_w[s] = suborder_w_rule_15[s];
  }

  return;
}
//****************************************************************************80

void SimplexQuadrature::dunavant_subrule_16(int suborder_num, std::vector<double>& suborder_xyz,
					    std::vector<double>& suborder_w)

//****************************************************************************80
//
//  Purpose:
//
//    DUNAVANT_SUBRULE_16 returns a compressed Dunavant rule 16.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    11 December 2006
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    David Dunavant,
//    High Degree Efficient Symmetrical Gaussian Quadrature Rules
//    for the Triangle,
//    International Journal for Numerical Methods in Engineering,
//    Volume 21, 1985, pages 1129-1148.
//
//    James Lyness, Dennis Jespersen,
//    Moderate Degree Symmetric Quadrature Rules for the Triangle,
//    Journal of the Institute of Mathematics and its Applications,
//    Volume 15, Number 1, February 1975, pages 19-32.
//
//  Parameters:
//
//    Input, int SUBORDER_NUM, the number of suborders of the rule.
//
//    Output, double SUBORDER_XYZ[3*SUBORDER_NUM],
//    the barycentric coordinates of the abscissas.
//
//    Output, double SUBORDER_W[SUBORDER_NUM], the suborder weights.
//
{
  int s;
  double suborder_xy_rule_16[3*13] = {
    0.333333333333333, 0.333333333333333, 0.333333333333333,
    0.005238916103123, 0.497380541948438, 0.497380541948438,
    0.173061122901295, 0.413469438549352, 0.413469438549352,
    0.059082801866017, 0.470458599066991, 0.470458599066991,
    0.518892500060958, 0.240553749969521, 0.240553749969521,
    0.704068411554854, 0.147965794222573, 0.147965794222573,
    0.849069624685052, 0.075465187657474, 0.075465187657474,
    0.966807194753950, 0.016596402623025, 0.016596402623025,
    0.103575692245252, 0.296555596579887, 0.599868711174861,
    0.020083411655416, 0.337723063403079, 0.642193524941505,
    -0.004341002614139, 0.204748281642812, 0.799592720971327,
    0.041941786468010, 0.189358492130623, 0.768699721401368,
    0.014317320230681, 0.085283615682657, 0.900399064086661
  };
  double suborder_w_rule_16[13] = {
    0.046875697427642,
    0.006405878578585,
    0.041710296739387,
    0.026891484250064,
    0.042132522761650,
    0.030000266842773,
    0.014200098925024,
    0.003582462351273,
    0.032773147460627,
    0.015298306248441,
    0.002386244192839,
    0.019084792755899,
    0.006850054546542
  };

  for (s = 0; s < suborder_num; s++)
  {
    suborder_xyz[0+s*3] = suborder_xy_rule_16[0+s*3];
    suborder_xyz[1+s*3] = suborder_xy_rule_16[1+s*3];
    suborder_xyz[2+s*3] = suborder_xy_rule_16[2+s*3];
  }

  for (s = 0; s < suborder_num; s++)
  {
    suborder_w[s] = suborder_w_rule_16[s];
  }

  return;
}
//****************************************************************************80

void SimplexQuadrature::dunavant_subrule_17(int suborder_num, std::vector<double>& suborder_xyz,
					    std::vector<double>& suborder_w)

//****************************************************************************80
//
//  Purpose:
//
//    DUNAVANT_SUBRULE_17 returns a compressed Dunavant rule 17.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    11 December 2006
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    David Dunavant,
//    High Degree Efficient Symmetrical Gaussian Quadrature Rules
//    for the Triangle,
//    International Journal for Numerical Methods in Engineering,
//    Volume 21, 1985, pages 1129-1148.
//
//    James Lyness, Dennis Jespersen,
//    Moderate Degree Symmetric Quadrature Rules for the Triangle,
//    Journal of the Institute of Mathematics and its Applications,
//    Volume 15, Number 1, February 1975, pages 19-32.
//
//  Parameters:
//
//    Input, int SUBORDER_NUM, the number of suborders of the rule.
//
//    Output, double SUBORDER_XYZ[3*SUBORDER_NUM],
//    the barycentric coordinates of the abscissas.
//
//    Output, double SUBORDER_W[SUBORDER_NUM], the suborder weights.
//
{
  int s;
  double suborder_xy_rule_17[3*15] = {
    0.333333333333333, 0.333333333333333, 0.333333333333333,
    0.005658918886452, 0.497170540556774, 0.497170540556774,
    0.035647354750751, 0.482176322624625, 0.482176322624625,
    0.099520061958437, 0.450239969020782, 0.450239969020782,
    0.199467521245206, 0.400266239377397, 0.400266239377397,
    0.495717464058095, 0.252141267970953, 0.252141267970953,
    0.675905990683077, 0.162047004658461, 0.162047004658461,
    0.848248235478508, 0.075875882260746, 0.075875882260746,
    0.968690546064356, 0.015654726967822, 0.015654726967822,
    0.010186928826919, 0.334319867363658, 0.655493203809423,
    0.135440871671036, 0.292221537796944, 0.572337590532020,
    0.054423924290583, 0.319574885423190, 0.626001190286228,
    0.012868560833637, 0.190704224192292, 0.796427214974071,
    0.067165782413524, 0.180483211648746, 0.752351005937729,
    0.014663182224828, 0.080711313679564, 0.904625504095608
  };
  double suborder_w_rule_17[15] = {
    0.033437199290803,
    0.005093415440507,
    0.014670864527638,
    0.024350878353672,
    0.031107550868969,
    0.031257111218620,
    0.024815654339665,
    0.014056073070557,
    0.003194676173779,
    0.008119655318993,
    0.026805742283163,
    0.018459993210822,
    0.008476868534328,
    0.018292796770025,
    0.006665632004165
  };

  for (s = 0; s < suborder_num; s++)
  {
    suborder_xyz[0+s*3] = suborder_xy_rule_17[0+s*3];
    suborder_xyz[1+s*3] = suborder_xy_rule_17[1+s*3];
    suborder_xyz[2+s*3] = suborder_xy_rule_17[2+s*3];
  }

  for (s = 0; s < suborder_num; s++)
  {
    suborder_w[s] = suborder_w_rule_17[s];
  }

  return;
}
//****************************************************************************80

void SimplexQuadrature::dunavant_subrule_18(int suborder_num, std::vector<double>& suborder_xyz,
					    std::vector<double>& suborder_w)

//****************************************************************************80
//
//  Purpose:
//
//    DUNAVANT_SUBRULE_18 returns a compressed Dunavant rule 18.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    11 December 2006
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    David Dunavant,
//    High Degree Efficient Symmetrical Gaussian Quadrature Rules
//    for the Triangle,
//    International Journal for Numerical Methods in Engineering,
//    Volume 21, 1985, pages 1129-1148.
//
//    James Lyness, Dennis Jespersen,
//    Moderate Degree Symmetric Quadrature Rules for the Triangle,
//    Journal of the Institute of Mathematics and its Applications,
//    Volume 15, Number 1, February 1975, pages 19-32.
//
//  Parameters:
//
//    Input, int SUBORDER_NUM, the number of suborders of the rule.
//
//    Output, double SUBORDER_XYZ[3*SUBORDER_NUM],
//    the barycentric coordinates of the abscissas.
//
//    Output, double SUBORDER_W[SUBORDER_NUM], the suborder weights.
//
{
  int s;
  double suborder_xy_rule_18[3*17] = {
    0.333333333333333, 0.333333333333333, 0.333333333333333,
    0.013310382738157, 0.493344808630921, 0.493344808630921,
    0.061578811516086, 0.469210594241957, 0.469210594241957,
    0.127437208225989, 0.436281395887006, 0.436281395887006,
    0.210307658653168, 0.394846170673416, 0.394846170673416,
    0.500410862393686, 0.249794568803157, 0.249794568803157,
    0.677135612512315, 0.161432193743843, 0.161432193743843,
    0.846803545029257, 0.076598227485371, 0.076598227485371,
    0.951495121293100, 0.024252439353450, 0.024252439353450,
    0.913707265566071, 0.043146367216965, 0.043146367216965,
    0.008430536202420, 0.358911494940944, 0.632657968856636,
    0.131186551737188, 0.294402476751957, 0.574410971510855,
    0.050203151565675, 0.325017801641814, 0.624779046792512,
    0.066329263810916, 0.184737559666046, 0.748933176523037,
    0.011996194566236, 0.218796800013321, 0.769207005420443,
    0.014858100590125, 0.101179597136408, 0.883962302273467,
    -0.035222015287949, 0.020874755282586, 1.014347260005363
  };
  double suborder_w_rule_18[17] = {
    0.030809939937647,
    0.009072436679404,
    0.018761316939594,
    0.019441097985477,
    0.027753948610810,
    0.032256225351457,
    0.025074032616922,
    0.015271927971832,
    0.006793922022963,
    -0.002223098729920,
    0.006331914076406,
    0.027257538049138,
    0.017676785649465,
    0.018379484638070,
    0.008104732808192,
    0.007634129070725,
    0.000046187660794
  };

  for (s = 0; s < suborder_num; s++)
  {
    suborder_xyz[0+s*3] = suborder_xy_rule_18[0+s*3];
    suborder_xyz[1+s*3] = suborder_xy_rule_18[1+s*3];
    suborder_xyz[2+s*3] = suborder_xy_rule_18[2+s*3];
  }

  for (s = 0; s < suborder_num; s++)
  {
    suborder_w[s] = suborder_w_rule_18[s];
  }

  return;
}
//****************************************************************************80

void SimplexQuadrature::dunavant_subrule_19(int suborder_num, std::vector<double>& suborder_xyz,
					    std::vector<double>& suborder_w)

//****************************************************************************80
//
//  Purpose:
//
//    DUNAVANT_SUBRULE_19 returns a compressed Dunavant rule 19.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    11 December 2006
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    David Dunavant,
//    High Degree Efficient Symmetrical Gaussian Quadrature Rules
//    for the Triangle,
//    International Journal for Numerical Methods in Engineering,
//    Volume 21, 1985, pages 1129-1148.
//
//    James Lyness, Dennis Jespersen,
//    Moderate Degree Symmetric Quadrature Rules for the Triangle,
//    Journal of the Institute of Mathematics and its Applications,
//    Volume 15, Number 1, February 1975, pages 19-32.
//
//  Parameters:
//
//    Input, int SUBORDER_NUM, the number of suborders of the rule.
//
//    Output, double SUBORDER_XYZ[3*SUBORDER_NUM],
//    the barycentric coordinates of the abscissas.
//
//    Output, double SUBORDER_W[SUBORDER_NUM], the suborder weights.
//
{
  int s;
  double suborder_xy_rule_19[3*17] = {
    0.333333333333333, 0.333333333333333, 0.333333333333333,
    0.020780025853987, 0.489609987073006, 0.489609987073006,
    0.090926214604215, 0.454536892697893, 0.454536892697893,
    0.197166638701138, 0.401416680649431, 0.401416680649431,
    0.488896691193805, 0.255551654403098, 0.255551654403098,
    0.645844115695741, 0.177077942152130, 0.177077942152130,
    0.779877893544096, 0.110061053227952, 0.110061053227952,
    0.888942751496321, 0.055528624251840, 0.055528624251840,
    0.974756272445543, 0.012621863777229, 0.012621863777229,
    0.003611417848412, 0.395754787356943, 0.600633794794645,
    0.134466754530780, 0.307929983880436, 0.557603261588784,
    0.014446025776115, 0.264566948406520, 0.720987025817365,
    0.046933578838178, 0.358539352205951, 0.594527068955871,
    0.002861120350567, 0.157807405968595, 0.839331473680839,
    0.223861424097916, 0.075050596975911, 0.701087978926173,
    0.034647074816760, 0.142421601113383, 0.822931324069857,
    0.010161119296278, 0.065494628082938, 0.924344252620784
  };
  double suborder_w_rule_19[17] = {
    0.032906331388919,
    0.010330731891272,
    0.022387247263016,
    0.030266125869468,
    0.030490967802198,
    0.024159212741641,
    0.016050803586801,
    0.008084580261784,
    0.002079362027485,
    0.003884876904981,
    0.025574160612022,
    0.008880903573338,
    0.016124546761731,
    0.002491941817491,
    0.018242840118951,
    0.010258563736199,
    0.003799928855302
  };

  for (s = 0; s < suborder_num; s++)
  {
    suborder_xyz[0+s*3] = suborder_xy_rule_19[0+s*3];
    suborder_xyz[1+s*3] = suborder_xy_rule_19[1+s*3];
    suborder_xyz[2+s*3] = suborder_xy_rule_19[2+s*3];
  }

  for (s = 0; s < suborder_num; s++)
  {
    suborder_w[s] = suborder_w_rule_19[s];
  }

  return;
}
//****************************************************************************80

void SimplexQuadrature::dunavant_subrule_20(int suborder_num, std::vector<double>& suborder_xyz,
					    std::vector<double>& suborder_w)

//****************************************************************************80
//
//  Purpose:
//
//    DUNAVANT_SUBRULE_20 returns a compressed Dunavant rule 20.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    11 December 2006
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    David Dunavant,
//    High Degree Efficient Symmetrical Gaussian Quadrature Rules
//    for the Triangle,
//    International Journal for Numerical Methods in Engineering,
//    Volume 21, 1985, pages 1129-1148.
//
//    James Lyness, Dennis Jespersen,
//    Moderate Degree Symmetric Quadrature Rules for the Triangle,
//    Journal of the Institute of Mathematics and its Applications,
//    Volume 15, Number 1, February 1975, pages 19-32.
//
//  Parameters:
//
//    Input, int SUBORDER_NUM, the number of suborders of the rule.
//
//    Output, double SUBORDER_XYZ[3*SUBORDER_NUM],
//    the barycentric coordinates of the abscissas.
//
//    Output, double SUBORDER_W[SUBORDER_NUM], the suborder weights.
//
{
  int s;
  double suborder_xy_rule_20[3*19] = {
    0.333333333333333, 0.333333333333333, 0.333333333333333,
    -0.001900928704400, 0.500950464352200, 0.500950464352200,
    0.023574084130543, 0.488212957934729, 0.488212957934729,
    0.089726636099435, 0.455136681950283, 0.455136681950283,
    0.196007481363421, 0.401996259318289, 0.401996259318289,
    0.488214180481157, 0.255892909759421, 0.255892909759421,
    0.647023488009788, 0.176488255995106, 0.176488255995106,
    0.791658289326483, 0.104170855336758, 0.104170855336758,
    0.893862072318140, 0.053068963840930, 0.053068963840930,
    0.916762569607942, 0.041618715196029, 0.041618715196029,
    0.976836157186356, 0.011581921406822, 0.011581921406822,
    0.048741583664839, 0.344855770229001, 0.606402646106160,
    0.006314115948605, 0.377843269594854, 0.615842614456541,
    0.134316520547348, 0.306635479062357, 0.559048000390295,
    0.013973893962392, 0.249419362774742, 0.736606743262866,
    0.075549132909764, 0.212775724802802, 0.711675142287434,
    -0.008368153208227, 0.146965436053239, 0.861402717154987,
    0.026686063258714, 0.137726978828923, 0.835586957912363,
    0.010547719294141, 0.059696109149007, 0.929756171556853
  };
  double suborder_w_rule_20[19] = {
    0.033057055541624,
    0.000867019185663,
    0.011660052716448,
    0.022876936356421,
    0.030448982673938,
    0.030624891725355,
    0.024368057676800,
    0.015997432032024,
    0.007698301815602,
    -0.000632060497488,
    0.001751134301193,
    0.016465839189576,
    0.004839033540485,
    0.025804906534650,
    0.008471091054441,
    0.018354914106280,
    0.000704404677908,
    0.010112684927462,
    0.003573909385950
  };

  for (s = 0; s < suborder_num; s++)
  {
    suborder_xyz[0+s*3] = suborder_xy_rule_20[0+s*3];
    suborder_xyz[1+s*3] = suborder_xy_rule_20[1+s*3];
    suborder_xyz[2+s*3] = suborder_xy_rule_20[2+s*3];
  }

  for (s = 0; s < suborder_num; s++)
  {
    suborder_w[s] = suborder_w_rule_20[s];
  }

  return;
}
//****************************************************************************80

int SimplexQuadrature::i4_modp(int i, int j)

//****************************************************************************80
//
//  Purpose:
//
//    I4_MODP returns the nonnegative remainder of I4 division.
//
//  Formula:
//
//    If
//      NREM = I4_MODP(I, J)
//      NMULT = (I - NREM) / J
//    then
//      I = J * NMULT + NREM
//    where NREM is always nonnegative.
//
//  Discussion:
//
//    The MOD function computes a result with the same sign as the
//    quantity being divided.  Thus, suppose you had an angle A,
//    and you wanted to ensure that it was between 0 and 360.
//    Then mod(A,360) would do, if A was positive, but if A
//    was negative, your result would be between -360 and 0.
//
//    On the other hand, I4_MODP(A,360) is between 0 and 360, always.
//
//  Example:
//
//        I         J     MOD  I4_MODP   I4_MODP Factorization
//
//      107        50       7       7    107 =  2 *  50 + 7
//      107       -50       7       7    107 = -2 * -50 + 7
//     -107        50      -7      43   -107 = -3 *  50 + 43
//     -107       -50      -7      43   -107 =  3 * -50 + 43
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    26 May 1999
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, int I, the number to be divided.
//
//    Input, int J, the number that divides I.
//
//    Output, int I4_MODP, the nonnegative remainder when I is
//    divided by J.
//
{
  int value;

  if (j == 0)
  {
    dolfin_error("SimplexQuadrature.cpp",
                 "compute quadrature rule for triangle",
                 "i4_modp must have non-zero j, which is here ", j);
  }

  value = i % j;

  if (value < 0)
  {
    value = value + abs(j);
  }

  return value;
}
//****************************************************************************80*

int SimplexQuadrature::i4_wrap(int ival, int ilo, int ihi)

//****************************************************************************80*
//
//  Purpose:
//
//    I4_WRAP forces an integer to lie between given limits by wrapping.
//
//  Example:
//
//    ILO = 4, IHI = 8
//
//    I   Value
//
//    -2     8
//    -1     4
//     0     5
//     1     6
//     2     7
//     3     8
//     4     4
//     5     5
//     6     6
//     7     7
//     8     8
//     9     4
//    10     5
//    11     6
//    12     7
//    13     8
//    14     4
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    19 August 2003
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, int IVAL, an integer value.
//
//    Input, int ILO, IHI, the desired bounds for the integer value.
//
//    Output, int I4_WRAP, a "wrapped" version of IVAL.
//
{
  int jhi;
  int jlo;
  int value;
  int wide;

  jlo = std::min(ilo, ihi);
  jhi = std::max(ilo, ihi);

  wide = jhi + 1 - jlo;

  if (wide == 1)
  {
    value = jlo;
  }
  else
  {
    value = jlo + i4_modp(ival - jlo, wide);
  }

  return value;
}
//-----------------------------------------------------------------------------
void SimplexQuadrature::legendre_compute_glr(std::size_t n,
					     std::vector<double>& x,
					     std::vector<double>& w)

//****************************************************************************80
//
//  Purpose:
//
//    LEGENDRE_COMPUTE_GLR: Legendre quadrature by the Glaser-Liu-Rokhlin method.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    20 October 2009
//
//  Author:
//
//    Original C++ version by Nick Hale.
//    This C++ version by John Burkardt.
//
//  Reference:
//
//    Andreas Glaser, Xiangtao Liu, Vladimir Rokhlin,
//    A fast algorithm for the calculation of the roots of special functions,
//    SIAM Journal on Scientific Computing,
//    Volume 29, Number 4, pages 1420-1438, 2007.
//
//  Parameters:
//
//    Input, int N, the order.
//
//    Output, double X[N], the abscissas.
//
//    Output, double W[N], the weights.
//
{
  x.resize(n);
  w.resize(n);

  int i;
  double p;
  double pp;
  double w_sum;
  //
  //  Get the value and derivative of the N-th Legendre polynomial at 0.
  //
  legendre_compute_glr0(n, p, pp);
  //
  //  If N is odd, then zero is a root.
  //
  if (n % 2 == 1)
  {
    x[(n-1)/2] = p;
    w[(n-1)/2] = pp;
  }
  //
  //  If N is even, we have to call a function to find the first root.
  //
  else
  {
    legendre_compute_glr2(p, n, x[n/2], w[n/2]);
  }
  //
  //  Get the complete set of roots and derivatives.
  //
  legendre_compute_glr1(n, x, w);
  //
  //  Compute the W.
  //
  for (i = 0; i < n; i++)
  {
    w[i] = 2.0 /(1.0 - x[i]) /(1.0 + x[i]) / w[i] / w[i];
  }
  w_sum = 0.0;
  for (i = 0; i < n; i++)
  {
    w_sum = w_sum + w[i];
  }
  for (i = 0; i < n; i++)
  {
    w[i] = 2.0 * w[i] / w_sum;
  }
  return;
}
//****************************************************************************80

void SimplexQuadrature::legendre_compute_glr0(std::size_t n, double& p, double& pp)

//****************************************************************************80
//
//  Purpose:
//
//    LEGENDRE_COMPUTE_GLR0 gets a starting value for the fast algorithm.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    19 October 2009
//
//  Author:
//
//    Original C++ version by Nick Hale.
//    This C++ version by John Burkardt.
//
//  Reference:
//
//    Andreas Glaser, Xiangtao Liu, Vladimir Rokhlin,
//    A fast algorithm for the calculation of the roots of special functions,
//    SIAM Journal on Scientific Computing,
//    Volume 29, Number 4, pages 1420-1438, 2007.
//
//  Parameters:
//
//    Input, int N, the order of the Legendre polynomial.
//
//    Output, double *P, *PP, the value of the N-th Legendre polynomial
//    and its derivative at 0.
//
{
  double dk;
  std::size_t k;
  double pm1;
  double pm2;
  double ppm1;
  double ppm2;

  pm2 = 0.0;
  pm1 = 1.0;
  ppm2 = 0.0;
  ppm1 = 0.0;

  for (k = 0; k < n; k++)
  {
    dk = static_cast<double>(k);
    p = - dk * pm2 /(dk + 1.0);
    pp = (( 2.0 * dk + 1.0) * pm1 - dk * ppm2) /(dk + 1.0);
    pm2 = pm1;
    pm1 = p;
    ppm2 = ppm1;
    ppm1 = pp;
  }
  return;
}
//****************************************************************************80

void SimplexQuadrature::legendre_compute_glr1(std::size_t n,
					      std::vector<double>& x,
					      std::vector<double>& w)

//****************************************************************************80
//
//  Purpose:
//
//    LEGENDRE_COMPUTE_GLR1 gets the complete set of Legendre points and weights.
//
//  Discussion:
//
//    This routine requires that a starting estimate be provided for one
//    root and its derivative.  This information will be stored in entry
//    (N+1)/2 if N is odd, or N/2 if N is even, of X and W.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    19 October 2009
//
//  Author:
//
//    Original C++ version by Nick Hale.
//    This C++ version by John Burkardt.
//
//  Reference:
//
//    Andreas Glaser, Xiangtao Liu, Vladimir Rokhlin,
//    A fast algorithm for the calculation of the roots of special functions,
//    SIAM Journal on Scientific Computing,
//    Volume 29, Number 4, pages 1420-1438, 2007.
//
//  Parameters:
//
//    Input, int N, the order of the Legendre polynomial.
//
//    Input/output, double X[N].  On input, a starting value
//    has been set in one entry.  On output, the roots of the Legendre
//    polynomial.
//
//    Input/output, double W[N].  On input, a starting value
//    has been set in one entry.  On output, the derivatives of the Legendre
//    polynomial at the zeros.
//
//  Local Parameters:
//
//    Local, int M, the number of terms in the Taylor expansion.
//
{
  double dk;
  double dn;
  double h;
  int j;
  int k;
  int l;
  int m = 30;
  int n2;
  int s;
  // double *u;
  // double *up;
  double xp;

  if (n % 2 == 1)
  {
    n2 = (n - 1) / 2 - 1;
    s = 1;
  }
  else
  {
    n2 = n / 2 - 1;
    s = 0;
  }

  // u = new double[m+2];
  // up = new double[m+1];
  std::vector<double> u(m+2);
  std::vector<double> up(m+1);

  dn = static_cast<double>(n);

  for (j = n2 + 1; j < n - 1; j++)
  {
    xp = x[j];

    h = rk2_leg(DOLFIN_PI/2.0, -DOLFIN_PI/2.0, xp, n) - xp;

    u[0] = 0.0;
    u[1] = 0.0;
    u[2] = w[j];

    up[0] = 0.0;
    up[1] = u[2];

    for (k = 0; k <= m - 2; k++)
    {
      dk = static_cast<double>(k);

      u[k+3] =
	(
	 2.0 * xp *(dk + 1.0) * u[k+2]
	 +(dk *(dk + 1.0) - dn *(dn + 1.0)) * u[k+1] /(dk + 1.0)
	 ) /(1.0 - xp) /(1.0 + xp) /(dk + 2.0);

      up[k+2] = (dk + 2.0) * u[k+3];
    }

    for (l = 0; l < 5; l++)
    {
      h = h - ts_mult(u, h, m) / ts_mult(up, h, m-1);
    }

    x[j+1] = xp + h;
    w[j+1] = ts_mult(up, h, m - 1);
  }

  for (k = 0; k <= n2 + s; k++)
  {
    x[k] = - x[n-1-k];
    w[k] = w[n-1-k];
  }
  return;
}
//****************************************************************************80

void SimplexQuadrature::legendre_compute_glr2(double pn0, int n, double& x1, double& d1)

//****************************************************************************80
//
//  Purpose:
//
//    LEGENDRE_COMPUTE_GLR2 finds the first real root.
//
//  Discussion:
//
//    This function is only called if N is even.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    19 October 2009
//
//  Author:
//
//    Original C++ version by Nick Hale.
//    This C++ version by John Burkardt.
//
//  Reference:
//
//    Andreas Glaser, Xiangtao Liu, Vladimir Rokhlin,
//    A fast algorithm for the calculation of the roots of special functions,
//    SIAM Journal on Scientific Computing,
//    Volume 29, Number 4, pages 1420-1438, 2007.
//
//  Parameters:
//
//    Input, double PN0, the value of the N-th Legendre polynomial
//    at 0.
//
//    Input, int N, the order of the Legendre polynomial.
//
//    Output, double *X1, the first real root.
//
//    Output, double *D1, the derivative at X1.
//
//  Local Parameters:
//
//    Local, int M, the number of terms in the Taylor expansion.
//
{
  double dk;
  double dn;
  int k;
  int l;
  int m = 30;
  double t;
  // double *u;
  // double *up;

  t = 0.0;
  x1 = rk2_leg(t, -DOLFIN_PI/2.0, 0.0, n);

  // u = new double[m+2];
  // up = new double[m+1];
  std::vector<double> u(m+2);
  std::vector<double> up(m+1);

  dn = static_cast<double>(n);
  //
  //  U[0] and UP[0] are never used.
  //  U[M+1] is set, but not used, and UP[M] is set and not used.
  //  What gives?
  //
  u[0] = 0.0;
  u[1] = pn0;

  up[0] = 0.0;

  for (k = 0; k <= m - 2; k = k + 2)
  {
    dk = static_cast<double>(k);

    u[k+2] = 0.0;
    u[k+3] = (dk *(dk + 1.0) - dn *(dn + 1.0)) * u[k+1]
      / (dk + 1.0) / (dk + 2.0);

    up[k+1] = 0.0;
    up[k+2] = (dk + 2.0) * u[k+3];
  }

  for (l = 0; l < 5; l++)
  {
    x1 = x1 - ts_mult(u, x1, m) / ts_mult(up, x1, m-1);
  }
  d1 = ts_mult(up, x1, m-1);

  return;
}
//****************************************************************************80

double SimplexQuadrature::rk2_leg(double t1, double t2, double x, int n)

//****************************************************************************80
//
//  Purpose:
//
//    RK2_LEG advances the value of X(T) using a Runge-Kutta method.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    22 October 2009
//
//  Author:
//
//    Original C++ version by Nick Hale.
//    This C++ version by John Burkardt.
//
//  Parameters:
//
//    Input, double T1, T2, the range of the integration interval.
//
//    Input, double X, the value of X at T1.
//
//    Input, int N, the number of steps to take.
//
//    Output, double RK2_LEG, the value of X at T2.
//
{
  double f;
  double h;
  int j;
  double k1;
  double k2;
  int m = 10;
  double snn1;
  double t;

  h = (t2 - t1) / static_cast<double>(m);
  snn1 = sqrt(static_cast<double>(n *(n + 1)));
  t = t1;

  for (j = 0; j < m; j++)
  {
    f = (1.0 - x) *(1.0 + x);
    k1 = - h * f /(snn1 * sqrt(f) - 0.5 * x * sin(2.0 * t));
    x = x + k1;

    t = t + h;

    f = (1.0 - x) *(1.0 + x);
    k2 = - h * f /(snn1 * sqrt(f) - 0.5 * x * sin(2.0 * t));
    x = x + 0.5 *(k2 - k1);
  }
  return x;
}
//****************************************************************************80

double SimplexQuadrature::ts_mult(std::vector<double>& u, double h, int n)

//****************************************************************************80
//
//  Purpose:
//
//    TS_MULT evaluates a polynomial.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    17 May 2013
//
//  Author:
//
//    Original C++ version by Nick Hale.
//    This C++ version by John Burkardt.
//
//  Parameters:
//
//    Input, double U[N+1], the polynomial coefficients.
//    U[0] is ignored.
//
//    Input, double H, the polynomial argument.
//
//    Input, int N, the number of terms to compute.
//
//    Output, double TS_MULT, the value of the polynomial.
//
{
  double hk;
  int k;
  double ts;

  ts = 0.0;
  hk = 1.0;
  for (k = 1; k<= n; k++)
  {
    ts = ts + u[k] * hk;
    hk = hk * h;
  }
  return ts;
}
//-----------------------------------------------------------------------------
