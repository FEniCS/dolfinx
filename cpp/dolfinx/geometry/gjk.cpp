// Copyright (C) 2020 Chris Richardson and Mattia Montinari
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "gjk.h"
#include <dolfinx/common/math.h>
#include <numeric>
#include <stdexcept>
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
std::pair<std::vector<double>, std::array<double, 3>>
nearest_simplex(const std::span<const double>& ss)
{
  auto s = xt::adapt(ss.data(), ss.size(), xt::no_ownership(),
                     std::vector<std::size_t>{ss.size() / 3, 3});

  assert(ss.size() % 3 == 0);
  const std::size_t s_rows = s.shape(0);
  switch (s_rows)
  {
  case 2:
  {
    auto s0 = ss.subspan(0, 3);
    auto s1 = ss.subspan(3, 3);
    std::array ds = {s1[0] - s0[0], s1[1] - s0[1], s1[2] - s0[2]};
    const double lm = -(s0[0] * ds[0] + s[1] * ds[1] + s[2] * ds[2])
                      / (ds[0] * ds[0] + ds[1] * ds[1] + ds[2] * ds[2]);
    if (lm >= 0.0 and lm <= 1.0)
    {
      // The origin is between A and B
      // v = s0 + lm * (s1 - s0);
      std::array<double, 3> v
          = {s0[0] + lm * ds[0], s0[1] + lm * ds[1], s0[2] + lm * ds[2]};
      return {std::vector<double>(ss.begin(), ss.end()), v};
    }

    if (lm < 0.0)
      return {std::vector<double>(s0.begin(), s0.end()), {s0[0], s0[1], s0[2]}};
    else
      return {std::vector<double>(s1.begin(), s1.end()), {s1[0], s1[1], s1[2]}};
  }
  case 3:
  {
    auto a = ss.subspan(0, 3);
    auto b = ss.subspan(3, 3);
    auto c = ss.subspan(6, 3);
    auto length = [](auto& x, auto& y)
    {
      return std::transform_reduce(
          x.begin(), x.end(), y.begin(), 0.0, std::plus{},
          [](auto x, auto y) { return (x - y) * (x - y); });
    };
    const double ab2 = length(a, b);
    const double ac2 = length(a, c);
    const double bc2 = length(b, c);

    auto helper = [](auto& x, auto& y)
    {
      return std::transform_reduce(x.begin(), x.end(), y.begin(), 0.0,
                                   std::plus{},
                                   [](auto x, auto y) { return x * (x - y); });
    };
    const std::array<double, 3> lm
        = {helper(a, b) / ab2, helper(a, c) / ac2, helper(b, c) / bc2};

    double caba = 0;
    for (std::size_t i = 0; i < 3; ++i)
      caba += (c[i] - a[i]) * (b[i] - a[i]);

    // Calculate triangle ABC
    const double c2 = 1 - caba * caba / (ab2 * ac2);
    const double lbb = (lm[0] - lm[1] * caba / ab2) / c2;
    const double lcc = (lm[1] - lm[0] * caba / ac2) / c2;

    // Intersects triangle
    if (lbb >= 0.0 and lcc >= 0.0 and (lbb + lcc) <= 1.0)
    {
      // Calculate intersection more accurately
      // auto v = math::cross(c - a, b - a);

      std::array<double, 3> dx0 = {c[0] - a[0], c[1] - a[1], c[2] - a[2]};
      std::array<double, 3> dx1 = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
      std::array<double, 3> v = math::cross_new(dx0, dx1);

      // Barycentre of triangle
      std::array<double, 3> p
          = {(a[0] + b[0] + c[0]) / 3, (a[1] + b[1] + c[1]) / 3,
             (a[2] + b[2] + c[2]) / 3};

      double sum = v[0] * p[0] + v[1] * p[1] + v[2] * p[2];
      double vnorm2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
      for (std::size_t i = 0; i < 3; ++i)
        v[i] *= sum / vnorm2;

      return {std::vector<double>(ss.begin(), ss.end()), v};
    }

    // Get closest point
    std::size_t i = xt::argmin(xt::norm_sq(s, {1}))();
    std::array<double, 3> vmin = {s(i, 0), s(i, 1), s(i, 2)};
    double qmin = 0;
    for (std::size_t k = 0; k < 3; ++k)
      qmin += vmin[k] * vmin[k];

    std::vector<double> smin = {vmin[0], vmin[1], vmin[2]};

    // Check if edges are closer
    constexpr const int f[3][2] = {{0, 1}, {0, 2}, {1, 2}};
    for (std::size_t i = 0; i < s_rows; ++i)
    {
      auto s0 = ss.subspan(3 * f[i][0], 3);
      auto s1 = ss.subspan(3 * f[i][1], 3);
      if (lm[i] > 0 and lm[i] < 1)
      {
        std::array<double, 3> v;
        for (std::size_t k = 0; k < 3; ++k)
          v[k] = s0[k] + lm[i] * (s1[k] - s0[k]);
        double qnorm = 0;
        for (std::size_t k = 0; k < 3; ++k)
          qnorm += v[k] * v[k];
        if (qnorm < qmin)
        {
          std::copy(v.begin(), v.end(), vmin.begin());
          qmin = qnorm;
          smin.resize(2 * 3);
          std::span<double, 3> smin0(smin.data(), 3);
          std::copy(s0.begin(), s0.end(), smin0.begin());
          std::span<double, 3> smin1(smin.data() + 3, 3);
          std::copy(s1.begin(), s1.end(), smin1.begin());
        }
      }
    }
    return {std::move(smin), {vmin[0], vmin[1], vmin[2]}};
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
        return {std::vector<double>(ss.begin(), ss.end()), {0, 0, 0}};
      else // The origin projection P faces BCD
        return nearest_simplex(ss.subspan(0, 3 * 3));
    }

    // Test ACD, ABD and/or ABC
    std::vector<double> smin;
    std::array<double, 3> vmin = {0, 0, 0};
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
        double q = std::transform_reduce(v.begin(), v.end(), v.begin(), 0);
        if (q < qmin)
        {
          qmin = q;
          vmin = v;
          smin = snew;
        }
      }
    }
    return {std::move(smin), {vmin[0], vmin[1], vmin[2]}};
  }
  default:
    throw std::runtime_error("Number of rows defining simplex not supported.");
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
