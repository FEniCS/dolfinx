// Copyright (C) 2020 Mattia Montanari, Chris Richardson
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    to be decided

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *
 *                                   #####        # #    #                *
 *       ####  #####  ###### #    # #     #       # #   #                 *
 *      #    # #    # #      ##   # #             # #  #                  *
 *      #    # #    # #####  # #  # #  ####       # ###                   *
 *      #    # #####  #      #  # # #     # #     # #  #                  *
 *      #    # #      #      #   ## #     # #     # #   #                 *
 *       ####  #      ###### #    #  #####   #####  #    #                *
 *                                                                        *
 *  This file is part of openGJK.                                         *
 *                                                                        *
 *       openGJK: open-source Gilbert-Johnson-Keerthi algorithm           *
 *            Copyright (C) Mattia Montanari 2018 - 2020                  *
 *              http://iel.eng.ox.ac.uk/?page_id=504                      *
 *                                                                        *
 * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *
 *                                                                        *
 * This file implements the GJK algorithm and the Signed Volumes method as*
 * presented in:                                                          *
 *   M. Montanari, N. Petrinic, E. Barbieri, "Improving the GJK Algorithm *
 *   for Faster and More Reliable Distance Queries Between Convex Objects"*
 *   ACM Transactions on Graphics, vol. 36, no. 3, Jun. 2017.             *
 *                                                                        *
 * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  */

#include "openGJK.h"

#include <Eigen/Geometry>
#include <iomanip>
#include <iostream>
#include <stdexcept>

using namespace dolfinx;

using Simplex = struct Simplex
{
  /// Vertex coordinates
  Eigen::Matrix<double, 4, 3, Eigen::RowMajor> vrtx;

  /// Number of active vertices (1=point, 2=line, 3=triangle, 4=tetrahedron)
  int nvrtx;
};

namespace
{
/// @brief Finds point of minimum norm of 1-simplex. Robust, but slower,
/// version of algorithm presented in paper.
Eigen::Vector3d S1D(Simplex& s)
{
  assert(s.nvrtx == 2);
  const Eigen::RowVector3d t = s.vrtx.row(0) - s.vrtx.row(1);

  const double tnorm2 = t.squaredNorm();
  if (tnorm2 == 0.0)
    throw std::runtime_error("t=0 in S1D");

  // Project origin onto the 1D simplex - line
  const double pt = s.vrtx.row(0).dot(t) / tnorm2;

  if (pt >= 0.0 and pt <= 1.0)
  {
    // The origin is between A and B
    return s.vrtx.row(0) - pt * t;
  }

  // The origin is beyond A, change point
  if (pt > 1.0)
    s.vrtx.row(0) = s.vrtx.row(1);

  s.nvrtx = 1;
  return s.vrtx.row(0);
}

/// @brief Finds point of minimum norm of 2-simplex. Robust, but slower,
/// version of algorithm presented in paper.
Eigen::Vector3d S2D(Simplex& s)
{
  assert(s.nvrtx == 3);
  const Eigen::Vector3d ac = s.vrtx.row(0) - s.vrtx.row(2);
  const Eigen::Vector3d bc = s.vrtx.row(0) - s.vrtx.row(1);
  const Eigen::Vector3d& a = s.vrtx.row(2);

  // Find best axis for projection
  Eigen::Vector3d n = ac.cross(bc);
  if (n.squaredNorm() == 0.0)
    throw std::runtime_error("Zero normal in S2D");
  const Eigen::Vector3d nu_fabs = n.cwiseAbs();
  int indexI;
  nu_fabs.maxCoeff(&indexI);
  const double nu_max = n[indexI];

  // Renormalise n in plane of ABC
  n *= n.dot(a) / n.squaredNorm();

  const int indexJ0 = (indexI + 1) % 3;
  const int indexJ1 = (indexI + 2) % 3;
  const Eigen::Vector3d W1 = s.vrtx.col(indexJ0).head(3).reverse();
  const Eigen::Vector3d W2 = s.vrtx.col(indexJ1).head(3).reverse();
  const Eigen::Vector3d W3 = (W1 * n[indexJ1] - W2 * n[indexJ0]);
  Eigen::Vector3d B = W1.cross(W2);
  B[0] += (W3[2] - W3[1]);
  B[1] += (W3[0] - W3[2]);
  B[2] += (W3[1] - W3[0]);

  // Test if sign of ABC is equal to the signs of the auxiliary simplices
  int FacetsTest[3];
  for (int i = 0; i < 3; ++i)
    FacetsTest[i] = (std::signbit(nu_max) == std::signbit(B[i]));

  if ((FacetsTest[0] + FacetsTest[1] + FacetsTest[2]) == 3)
  {
    // The origin projection lays onto the triangle
    s.nvrtx = 3;
    return n;
  }

  if (FacetsTest[1] == 0 and FacetsTest[2] == 0)
  {
    Eigen::Vector3d v, vTmp;
    Simplex sTmp;
    sTmp.nvrtx = 2;
    sTmp.vrtx.row(0) = s.vrtx.row(1);
    sTmp.vrtx.row(1) = s.vrtx.row(2);
    vTmp = S1D(sTmp);

    s.nvrtx = 2;
    s.vrtx.row(1) = s.vrtx.row(2);
    v = S1D(s);

    if (vTmp.squaredNorm() < v.squaredNorm())
    {
      s = sTmp;
      return vTmp;
    }
    return v;
  }

  if (FacetsTest[2] == 0)
  {
    // The origin projection P faces the segment AB
    s.nvrtx = 2;
    s.vrtx.row(0) = s.vrtx.row(1);
    s.vrtx.row(1) = s.vrtx.row(2);
    return S1D(s);
  }

  if (FacetsTest[1] == 0)
  {
    // The origin projection P faces the segment AC
    s.nvrtx = 2;
    s.vrtx.row(1) = s.vrtx.row(2);
    return S1D(s);
  }

  // FacetsTest[0] == 0
  // The origin projection P faces the segment BC
  s.nvrtx = 2;
  return S1D(s);
}

/// @brief Finds point of minimum norm of 3-simplex. Robust, but slower,
/// version of algorithm presented in paper.
Eigen::Vector3d S3D(Simplex& s)
{
  assert(s.nvrtx == 4);

  Eigen::Vector4d B;
  const Eigen::Vector3d W1 = s.vrtx.row(0).cross(s.vrtx.row(1));
  const Eigen::Vector3d W2 = s.vrtx.row(2).cross(s.vrtx.row(3));
  B[0] = s.vrtx.row(2) * W1;
  B[1] = -s.vrtx.row(3) * W1;
  B[2] = s.vrtx.row(0) * W2;
  B[3] = -s.vrtx.row(1) * W2;

  double detM = B.sum();

  // Test if sign of ABCD is equal to the signs of the auxiliary simplices
  const double eps = 1e-15;
  int FacetsTest[4] = {1, 1, 1, 1};
  if (std::abs(detM) < eps)
  {
    Eigen::Vector4d Babs = B.cwiseAbs();
    if (Babs[2] < eps and Babs[3] < eps)
      FacetsTest[1] = 0; // A = B. Test only ACD
    else if (Babs[1] < eps and Babs[3] < eps)
      FacetsTest[2] = 0; // A = C. Test only ABD
    else if (Babs[1] < eps and Babs[2] < eps)
      FacetsTest[3] = 0; // A = D. Test only ABC
    else if (Babs[0] < eps and Babs[3] < eps)
      FacetsTest[1] = 0; // B = C. Test only ACD
    else if (Babs[0] < eps and Babs[2] < eps)
      FacetsTest[1] = 0; // B = D. Test only ACD
    else if (Babs[0] < eps and Babs[1] < eps)
      FacetsTest[2] = 0; // C = D. Test only ABD
    else
    {
      for (int i = 0; i < 4; i++)
        FacetsTest[i] = 0; // Any other case. Test ABC, ABD, ACD
    }
  }
  else
  {
    for (int i = 0; i < 4; ++i)
      FacetsTest[i] = (std::signbit(detM) == std::signbit(B[i]));
  }

  // Compare signed volumes and compute barycentric coordinates
  if (FacetsTest[0] + FacetsTest[1] + FacetsTest[2] + FacetsTest[3] == 4)
  {
    // All signs are equal, therefore the origin is inside the simplex
    s.nvrtx = 4;
    return Eigen::Vector3d::Zero();
  }

  if (FacetsTest[1] + FacetsTest[2] + FacetsTest[3] == 3)
  {
    // The origin projection P faces the facet BCD
    s.nvrtx = 3;
    return S2D(s);
  }

  // Either 1, 2 or 3 of ACD, ABD or ABC are closest.
  Simplex sTmp[2];
  Eigen::Vector3d vBest;
  static const int facets[3][3] = {{0, 1, 3}, {0, 2, 3}, {1, 2, 3}};
  double vmin = std::numeric_limits<double>::max();
  for (int i = 0; i < 3; ++i)
  {
    if (FacetsTest[i + 1] == 0)
    {
      sTmp[0].nvrtx = 3;
      sTmp[0].vrtx.row(0) = s.vrtx.row(facets[i][0]);
      sTmp[0].vrtx.row(1) = s.vrtx.row(facets[i][1]);
      sTmp[0].vrtx.row(2) = s.vrtx.row(facets[i][2]);
      Eigen::Vector3d vTmp = S2D(sTmp[0]);
      const double vnorm = vTmp.squaredNorm();
      if (vnorm < vmin)
      {
        vmin = vnorm;
        vBest = vTmp;
        sTmp[1] = sTmp[0];
      }
    }
  }
  s = sTmp[1];
  return vBest;
}
//--------------------------------------------------------------------------
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
//--------------------------------------------------------------------------
Eigen::Vector3d geometry::gjk_vector(
    const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& bd1,
    const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& bd2)
{
  const int mk = 50; // Maximum number of iterations of the GJK algorithm

  // Tolerances
  const double eps_tot = 1e-12;
  double eps_rel = 1e-12;
  // Minimum relative tolerance, after 10 cycles
  const double eps_rel_min = 1e-6;

  // Initialise
  Simplex s;
  s.nvrtx = 1;
  Eigen::Vector3d v = bd1.row(0) - bd2.row(0);
  s.vrtx.row(0) = v;

  // Begin GJK iteration
  int k;
  for (k = 0; k < mk; ++k)
  {
    // Support function
    const Eigen::Vector3d w = support(bd1, -v) - support(bd2, v);

    // Break if any existing points are the same as w
    int m;
    for (m = 0; m < s.nvrtx; ++m)
      if (s.vrtx(m, 0) == w[0] and s.vrtx(m, 1) == w[1]
          and s.vrtx(m, 2) == w[2])
        break;
    if (m != s.nvrtx)
      break;

    // If proving hard to converge, relax tolerance
    // FIXME...
    if (k > 10 and eps_rel < eps_rel_min)
      eps_rel *= 10;

    // 1st exit condition (v-w).v = 0
    const double vnorm2 = v.squaredNorm();
    const double vw = vnorm2 - v.dot(w);
    if (vw < (eps_rel * vnorm2) or vw < eps_tot)
      break;

    // Add new vertex to simplex
    assert(s.nvrtx != 4);
    s.vrtx.row(s.nvrtx) = w;
    ++s.nvrtx;

    // Invoke distance sub-algorithm
    switch (s.nvrtx)
    {
    case 4:
      v = S3D(s);
      break;
    case 3:
      v = S2D(s);
      break;
    case 2:
      v = S1D(s);
      break;
    }

    // 2nd exit condition - intersecting or touching
    if (v.squaredNorm() < eps_rel * eps_rel)
      break;
  }
  if (k == mk)
    throw std::runtime_error("OpenGJK error: max iteration limit reached");

  // Compute and return distance
  return v;
}
