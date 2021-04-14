// Copyright (C) 2020 Chris Richardson and Mattia Montinari
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "gjk.h"
#include <array>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <tuple>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xfixed.hpp>
#include <xtensor/xnorm.hpp>
#include <xtensor/xsort.hpp>
#include <xtensor/xtensor.hpp>

using namespace dolfinx;

namespace
{
// Find the resulting sub-simplex of the input simplex which is nearest to the
// origin. Also, return the shortest vector from the origin to the resulting
// simplex.
std::pair<xt::xtensor<double, 2>, xt::xtensor_fixed<double, xt::xshape<3>>>
nearest_simplex(const xt::xtensor<double, 2>& s)
{
  if (s.shape(0) == 2)
  {
    auto s0 = xt::row(s, 0);
    auto s1 = xt::row(s, 1);
    auto ab = s1 - s0;
    const double lm = -xt::norm_sq(s0 * ab)() / xt::norm_sq(ab)();
    if (lm >= 0.0 and lm <= 1.0)
    {
      // The origin is between A and B
      auto v = s0 + lm * ab;
      return {xt::view(s, xt::range(0, 2), xt::all()), v};
    }
    if (lm < 0.0)
      return {s0, s0};
    else
      return {s1, s1};
  }
  else if (s.shape(0) == 4)
  {
    auto s0 = xt::row(s, 0);
    auto s1 = xt::row(s, 1);
    auto s2 = xt::row(s, 2);
    auto s3 = xt::row(s, 3);

    auto W1 = xt::linalg::cross(s0, s1);
    auto W2 = xt::linalg::cross(s2, s3);

    // xt::xtensor_fixed<double, xt::xshape<4>> B
    //     = {s2 * W1, -s3 * W1, s0 * W2, -s1 * W2};
    xt::xtensor_fixed<double, xt::xshape<4>> B;
    B[0] = xt::sum(s2 * W1)();
    B[1] = -xt::sum(s3 * W1)();
    B[2] = xt::sum(s0 * W2)();
    B[3] = -xt::sum(s1 * W2)();

    const bool signDetM = std::signbit(xt::sum(B)());
    std::array<bool, 4> f_inside;
    for (int i = 0; i < 4; ++i)
      f_inside[i] = (std::signbit(B[i]) == signDetM);

    if (f_inside[1] and f_inside[2] and f_inside[3])
    {
      if (f_inside[0]) // The origin is inside the tetrahedron
      {
        xt::xtensor_fixed<double, xt::xshape<3>> tmp = xt::zeros<double>({3});
        return {s, tmp};
      }

      // The origin projection P faces BCD
      auto s_top = xt::view(s, xt::range(0, 3), xt::all());
      return nearest_simplex(s_top);
    }

    // Test ACD, ABD and/or ABC.
    xt::xtensor<double, 2> smin;
    xt::xtensor_fixed<double, xt::xshape<3>> vmin = xt::zeros<double>({3});
    constexpr int facets[3][3] = {{0, 1, 3}, {0, 2, 3}, {1, 2, 3}};
    double qmin = std::numeric_limits<double>::max();
    for (int i = 0; i < 3; ++i)
    {
      if (f_inside[i + 1] == false)
      {
        xt::xtensor_fixed<double, xt::xshape<3, 3>> M;
        xt::row(M, 0) = xt::row(s, facets[i][0]);
        xt::row(M, 1) = xt::row(s, facets[i][1]);
        xt::row(M, 2) = xt::row(s, facets[i][2]);

        auto [snew, v] = nearest_simplex(M);
        const double q = xt::norm_sq(v)();
        if (q < qmin)
        {
          qmin = q;
          vmin = v;
          smin = snew;
        }
      }
    }
    return {std::move(smin), std::move(vmin)};
  }

  assert(s.shape(0) == 3);
  const auto a = xt::row(s, 0);
  const auto b = xt::row(s, 1);
  const auto c = xt::row(s, 2);
  const double ab2 = xt::norm_sq(a - b)();
  const double ac2 = xt::norm_sq(a - c)();
  const double bc2 = xt::norm_sq(b - c)();
  const xt::xtensor_fixed<double, xt::xshape<3>> lm
      = {xt::sum(a * (a - b))() / ab2, xt::sum(a * (a - c))() / ac2,
         xt::sum(b * (b - c))() / bc2};

  // Calculate triangle ABC
  const double caba = xt::sum((c - a) * (b - a))();
  const double c2 = 1 - caba * caba / (ab2 * ac2);
  const double lbb = (lm[0] - lm[1] * caba / ab2) / c2;
  const double lcc = (lm[1] - lm[0] * caba / ac2) / c2;

  // Intersects triangle
  if (lbb >= 0.0 and lcc >= 0.0 and (lbb + lcc) <= 1.0)
  {
    // Calculate intersection more accurately
    auto v = xt::linalg::cross(c - a, b - a);

    // Barycentre of triangle
    auto p = (a + b + c) / 3.0;

    // Renormalise n in plane of ABC
    v *= xt::sum(v * p) / xt::norm_sq(v);
    return {std::move(s), std::move(v)};
  }

  // Get closest point
  std::size_t i = xt::argmin(xt::norm_sq(s, {1}))();
  xt::xtensor_fixed<double, xt::xshape<3>> vmin = xt::row(s, i);
  double qmin = xt::norm_sq(vmin)();

  xt::xtensor<double, 2> smin({1, 3});
  smin = vmin;

  // Check if edges are closer
  constexpr const int f[3][2] = {{0, 1}, {0, 2}, {1, 2}};
  for (std::size_t i = 0; i < s.shape(0); ++i)
  {
    auto s0 = xt::row(s, f[i][0]);
    auto s1 = xt::row(s, f[i][1]);
    if (lm[i] > 0 and lm[i] < 1)
    {
      auto v = s0 + lm[i] * (s1 - s0);
      const double qnorm = xt::norm_sq(v)();
      if (qnorm < qmin)
      {
        vmin = v;
        qmin = qnorm;
        smin.resize({2, 3});
        xt::row(smin, 0) = s0;
        xt::row(smin, 1) = s1;
      }
    }
  }

  return {std::move(smin), std::move(vmin)};
}
//----------------------------------------------------------------------------
// Support function, finds point p in bd which maximises p.v
xt::xtensor_fixed<double, xt::xshape<3>>
support(const xt::xtensor<double, 2>& bd,
        const xt::xtensor_fixed<double, xt::xshape<3>>& v)
{
  int i = 0;
  double qmax = xt::sum(xt::row(bd, 0) * v)();
  for (std::size_t m = 1; m < bd.shape(0); ++m)
  {
    double q = xt::sum(xt::row(bd, m) * v)();
    if (q > qmax)
    {
      qmax = q;
      i = m;
    }
  }

  return xt::row(bd, i);
}
} // namespace
//----------------------------------------------------------------------------
std::array<double, 3>
geometry::compute_distance_gjk(const xt::xtensor<double, 2>& p,
                               const xt::xtensor<double, 2>& q)
{
  assert(p.shape(1) == 3);
  assert(q.shape(1) == 3);

  const int maxk = 10; // Maximum number of iterations of the GJK algorithm

  // Tolerance
  constexpr double eps = 1e-12;

  // Initialise vector and simplex
  xt::xtensor_fixed<double, xt::xshape<3>> v = xt::row(p, 0) - xt::row(q, 0);
  xt::xtensor<double, 2> s({1, 3});
  s = v;

  // Begin GJK iteration
  int k;
  for (k = 0; k < maxk; ++k)
  {
    // Support function
    auto w = support(p, -v) - support(q, v);

    // Break if any existing points are the same as w
    std::size_t m;
    for (m = 0; m < s.shape(0); ++m)
    {
      if (s(m, 0) == w[0] and s(m, 1) == w[1] and s(m, 2) == w[2])
        break;
    }
    if (m != s.shape(0))
      break;

    // 1st exit condition (v-w).v = 0
    const double vnorm2 = xt::norm_sq(v)();
    const double vw = vnorm2 - xt::sum(v * w)();
    if (vw < (eps * vnorm2) or vw < eps)
      break;

    // Add new vertex to simplex
    s = xt::vstack(xt::xtuple(s, w));

    // Find nearest subset of simplex
    std::tie(s, v) = nearest_simplex(s);

    // 2nd exit condition - intersecting or touching
    if (xt::norm_sq(v)() < eps * eps)
      break;
  }

  if (k == maxk)
    throw std::runtime_error("GJK error: max iteration limit reached");

  return {v[0], v[1], v[2]};
}
//----------------------------------------------------------------------------
