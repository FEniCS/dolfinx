// Copyright (C) 2020 Chris Richardson and Mattia Montinari
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "GJK.h"
#include <Eigen/Geometry>
#include <array>
#include <iomanip>
#include <iostream>
#include <stdexcept>

using namespace dolfinx;

namespace
{
// Find the resulting sub-simplex of the input simplex which is nearest to the
// origin. Also, return the shortest vector from the origin to the resulting
// simplex.
std::pair<Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>,
          Eigen::Vector3d>
nearest_simplex(
    const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor, 4, 3>& s)
{
  if (s.rows() == 2)
  {
    const Eigen::Vector3d ab = s.row(1) - s.row(0);
    const double lm = -s.row(0).dot(ab) / ab.squaredNorm();
    if (lm >= 0.0 and lm <= 1.0)
    {
      // The origin is between A and B
      Eigen::Vector3d v = s.row(0).transpose() + lm * ab;
      return {s.topRows(2), v};
    }
    if (lm < 0.0)
      return {s.row(0), s.row(0)};
    else
      return {s.row(1), s.row(1)};
  }
  else if (s.rows() == 4)
  {
    Eigen::Vector4d B;
    const Eigen::Vector3d W1 = s.row(0).cross(s.row(1));
    const Eigen::Vector3d W2 = s.row(2).cross(s.row(3));
    B[0] = s.row(2) * W1;
    B[1] = -s.row(3) * W1;
    B[2] = s.row(0) * W2;
    B[3] = -s.row(1) * W2;

    bool signDetM = std::signbit(B.sum());
    Eigen::Array<bool, 4, 1> f_inside = B.unaryExpr(
        [&signDetM](double b) { return (std::signbit(b) == signDetM); });

    if (f_inside[1] and f_inside[2] and f_inside[3])
    {
      if (f_inside[0]) // The origin is inside the tetrahedron
        return {s, Eigen::Vector3d::Zero()};

      // The origin projection P faces BCD
      return nearest_simplex(s.topRows(3));
    }

    // Test ACD, ABD and/or ABC.
    Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> smin;
    Eigen::Vector3d vmin = {0, 0, 0};
    static const int facets[3][3] = {{0, 1, 3}, {0, 2, 3}, {1, 2, 3}};
    double qmin = std::numeric_limits<double>::max();
    for (int i = 0; i < 3; ++i)
    {
      if (f_inside[i + 1] == false)
      {
        Eigen::Matrix<double, 3, 3, Eigen::RowMajor> M;
        M << s.row(facets[i][0]), s.row(facets[i][1]), s.row(facets[i][2]);
        auto [snew, v] = nearest_simplex(M);
        const double q = v.squaredNorm();
        if (q < qmin)
        {
          qmin = q;
          vmin = v;
          smin = snew;
        }
      }
    }
    return {smin, vmin};
  }

  assert(s.rows() == 3);
  const auto a = s.row(0);
  const auto b = s.row(1);
  const auto c = s.row(2);
  const double ab2 = (a - b).squaredNorm();
  const double ac2 = (a - c).squaredNorm();
  const double bc2 = (b - c).squaredNorm();
  const Eigen::Vector3d lm
      = {a.dot(a - b) / ab2, a.dot(a - c) / ac2, b.dot(b - c) / bc2};

  // Calculate triangle ABC
  const double caba = (c - a).dot(b - a);
  const double c2 = 1 - caba * caba / (ab2 * ac2);
  const double lbb = (lm[0] - lm[1] * caba / ab2) / c2;
  const double lcc = (lm[1] - lm[0] * caba / ac2) / c2;

  // Intersects triangle
  if (lbb >= 0.0 and lcc >= 0.0 and (lbb + lcc) <= 1.0)
  {
    // Calculate intersection more accurately
    Eigen::Vector3d v = (c - a).cross(b - a);

    // Barycentre of triangle
    Eigen::Vector3d p = (a + b + c) / 3.0;
    // Renormalise n in plane of ABC
    v *= v.dot(p) / v.squaredNorm();
    return {s, v};
  }

  // Get closest point
  int i;
  s.rowwise().squaredNorm().minCoeff(&i);
  Eigen::Vector3d vmin = s.row(i);
  double qmin = vmin.squaredNorm();
  Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> smin
      = vmin.transpose();

  // Check if edges are closer
  static const int f[3][2] = {{0, 1}, {0, 2}, {1, 2}};
  for (int i = 0; i < s.rows(); ++i)
  {
    if (lm[i] > 0 and lm[i] < 1)
    {
      const Eigen::Vector3d v
          = s.row(f[i][0]) + lm[i] * (s.row(f[i][1]) - s.row(f[i][0]));
      const double qnorm = v.squaredNorm();
      if (qnorm < qmin)
      {
        vmin = v;
        qmin = qnorm;
        smin.resize(2, 3);
        smin.row(0) = s.row(f[i][0]);
        smin.row(1) = s.row(f[i][1]);
      }
    }
  }

  return {smin, vmin};
}
//-------------------------------------------------------------------------------
// Support function, finds point p in bd which maximises p.v
Eigen::Vector3d
support(const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& bd,
        const Eigen::Vector3d& v)
{
  int i = 0;
  double qmax = bd.row(0) * v;
  for (int m = 1; m < bd.rows(); ++m)
  {
    const double q = bd.row(m) * v;
    if (q > qmax)
    {
      qmax = q;
      i = m;
    }
  }
  return bd.row(i);
}
} // namespace
//-----------------------------------------------------
Eigen::Vector3d geometry::compute_distance_gjk(
    const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& p,
    const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& q)
{
  const int maxk = 10; // Maximum number of iterations of the GJK algorithm

  // Tolerance
  const double eps = 1e-12;

  // Initialise vector and simplex
  Eigen::Vector3d v = p.row(0) - q.row(0);
  Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor, 4, 3> s
      = v.transpose();

  // Begin GJK iteration
  int k;
  for (k = 0; k < maxk; ++k)
  {
    // Support function
    const Eigen::Vector3d w = support(p, -v) - support(q, v);

    // Break if any existing points are the same as w
    int m;
    for (m = 0; m < s.rows(); ++m)
    {
      if (s(m, 0) == w[0] and s(m, 1) == w[1] and s(m, 2) == w[2])
        break;
    }
    if (m != s.rows())
      break;

    // 1st exit condition (v-w).v = 0
    const double vnorm2 = v.squaredNorm();
    const double vw = vnorm2 - v.dot(w);
    if (vw < (eps * vnorm2) or vw < eps)
      break;

    // Add new vertex to simplex
    s.conservativeResize(s.rows() + 1, 3);
    s.bottomRows(1) = w.transpose();

    // Find nearest subset of simplex
    std::tie(s, v) = nearest_simplex(s);

    // 2nd exit condition - intersecting or touching
    if (v.squaredNorm() < eps * eps)
      break;
  }
  if (k == maxk)
    throw std::runtime_error("GJK error: max iteration limit reached");

  // Compute and return distance
  return v;
}
