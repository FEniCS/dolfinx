// Copyright (C) 2020 Chris Richardson and Mattia Montinari
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "gjk.h"
#include <dolfinx/common/math.h>
#include <stdexcept>
#include <tuple>
#include <xtensor/xadapt.hpp>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xfixed.hpp>
#include <xtensor/xnorm.hpp>
#include <xtensor/xsort.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>

using namespace dolfinx;

namespace
{

// Find the resulting sub-simplex of the input simplex which is nearest to the
// origin. Also, return the shortest vector from the origin to the resulting
// simplex.
std::pair<xt::xtensor<double, 2>, xt::xtensor_fixed<double, xt::xshape<3>>>
nearest_simplex(const xt::xtensor<double, 2>& s)
{
  assert(s.shape(1) == 3);
  const std::size_t s_rows = s.shape(0);
  switch (s_rows)
  {
  case 2:
  {
    auto s0 = xt::row(s, 0);
    auto s1 = xt::row(s, 1);
    // const double lm = -xt::sum(s0 * (s1 - s0))() / xt::norm_sq(s1 - s0)();
    std::array ds = {s1[0] - s0[0], s1[1] - s0[1], (s1[2] - s0[2])};
    const double lm = -(s0[0] * ds[0] + s[1] * ds[1] + s[2] * ds[2])
                      / (ds[0] * ds[0] + ds[1] * ds[1] + ds[2] * ds[2]);
    if (lm >= 0.0 and lm <= 1.0)
    {
      // The origin is between A and B
      // auto v = s0 + lm * (s1 - s0);
      xt::xtensor_fixed<double, xt::xshape<3>> v
          = {s0[0] + lm * ds[0], s0[1] + lm * ds[1], s0[2] + lm * ds[2]};
      return {s, v};
    }

    if (lm < 0.0)
      return {xt::reshape_view(s0, {1, 3}), s0};
    else
      return {xt::reshape_view(s1, {1, 3}), s1};
  }
  case 3:
  {
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
      auto v = math::cross(c - a, b - a);

      // Barycentre of triangle
      auto p = (a + b + c) / 3.0;

      // Renormalise n in plane of ABC
      v *= xt::sum(v * p) / xt::norm_sq(v);
      return {std::move(s), v};
    }

    // Get closest point
    std::size_t i = xt::argmin(xt::norm_sq(s, {1}))();
    xt::xtensor_fixed<double, xt::xshape<3>> vmin = xt::row(s, i);
    double qmin = xt::norm_sq(vmin)();

    xt::xtensor<double, 2> smin({1, 3});
    xt::row(smin, 0) = vmin;

    // Check if edges are closer
    constexpr const int f[3][2] = {{0, 1}, {0, 2}, {1, 2}};
    for (std::size_t i = 0; i < s_rows; ++i)
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
    return {std::move(smin), vmin};
  }
  case 4:
  {
    auto s0 = xt::row(s, 0);
    auto s1 = xt::row(s, 1);
    auto s2 = xt::row(s, 2);
    auto s3 = xt::row(s, 3);
    auto W1 = math::cross(s0, s1);
    auto W2 = math::cross(s2, s3);

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
        return {s, xt::zeros<double>({3})};
      else // The origin projection P faces BCD
        return nearest_simplex(xt::view(s, xt::range(0, 3), xt::all()));
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

        const auto [snew, v] = nearest_simplex(M);
        const double q = xt::norm_sq(v)();
        if (q < qmin)
        {
          qmin = q;
          vmin = v;
          smin = snew;
        }
      }
    }
    return {std::move(smin), vmin};
  }
  default:
  {
    throw std::runtime_error("Number of rows defining simplex not supported.");
  }
  }
}
//----------------------------------------------------------------------------
// Helper function, finds point p in bd which maximises p.v
std::array<double, 3> support(const std::span<const double>& bd,
                              const std::array<double, 3>& v)
{
  int i = 0;
  // double qmax = xt::sum(xt::row(bd, 0) * v)();
  double qmax = bd[0] * v[0] + bd[1] * v[1] + bd[2] * v[2];
  for (std::size_t m = 1; m < bd.size() / 3; ++m)
  {
    // double q = xt::sum(xt::row(bd, m) * v)();
    double q = bd[3 * m] * v[0] + bd[3 * m + 1] * v[1] + bd[3 * m + 2] * v[2];
    if (q > qmax)
    {
      qmax = q;
      i = m;
    }
  }

  return {bd[3 * i], bd[3 * i + 1], bd[3 * i + 2]};
  // return xt::row(bd, i);
}
} // namespace
//----------------------------------------------------------------------------
std::array<double, 3>
geometry::compute_distance_gjk(const std::span<const double>& p,
                               const std::span<const double>& q)
{
  assert(p.size() % 3 == 0);
  assert(q.size() % 3 == 0);

  constexpr int maxk = 10; // Maximum number of iterations of the GJK algorithm

  // Tolerance
  constexpr double eps = 1e-12;

  // Initialise vector and simplex
  std::array<double, 3> v = {p[0] - q[0], p[1] - q[1], p[2] - q[2]};
  std::vector<double> s = {v[0], v[1], v[2]};

  // Begin GJK iteration
  int k;
  for (k = 0; k < maxk; ++k)
  {
    // Support function
    std::array w1 = support(p, {-v[0], -v[1], -v[2]});
    std::array w0 = support(q, {v[0], v[1], v[2]});
    const std::array w = {w1[0] - w0[0], w1[1] - w0[1], w1[2] - w0[2]};

    // Break if any existing points are the same as w
    assert(s.size() % 3 == 0);
    std::size_t m;
    for (m = 0; m < s.size() / 3; ++m)
    {
      auto it = std::next(s.begin(), 3 * m);
      if (std::equal(it, std::next(it, 3), w.begin(), w.end()))
        break;
    }

    if (m != s.size() / 3)
      break;

    // 1st exit condition (v - w).v = 0
    const double vnorm2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    const double vw = vnorm2 - (v[0] * w[0] + v[1] * w[1] + v[2] * w[2]);
    if (vw < (eps * vnorm2) or vw < eps)
      break;

    // Add new vertex to simplex
    s.insert(s.end(), w.begin(), w.end());

    // Find nearest subset of simplex
    auto swrap = xt::adapt(s, std::vector<std::size_t>{s.size() / 3, 3});
    auto [snew, vnew] = nearest_simplex(swrap);
    s.assign(snew.data(), snew.data() + snew.size());
    v = {vnew[0], vnew[1], vnew[2]};

    // 2nd exit condition - intersecting or touching
    if ((v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) < eps * eps)
      break;
  }

  if (k == maxk)
    throw std::runtime_error("GJK error: max iteration limit reached");

  return v;
}
//----------------------------------------------------------------------------
