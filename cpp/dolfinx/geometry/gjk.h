// Copyright (C) 2020 Chris Richardson
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <array>
#include <concepts>
#include <dolfinx/common/math.h>
#include <limits>
#include <numeric>
#include <span>
#include <utility>
#include <vector>

namespace dolfinx::geometry
{

namespace impl_gjk
{

/// @brief Find the resulting sub-simplex of the input simplex which is
/// nearest to the origin. Also, return the shortest vector from the
/// origin to the resulting simplex.
template <std::floating_point T>
std::pair<std::vector<T>, std::array<T, 3>>
nearest_simplex(std::span<const T> s)
{
  assert(s.size() % 3 == 0);
  const std::size_t s_rows = s.size() / 3;
  switch (s_rows)
  {
  case 2:
  {
    // Compute lm = dot(s0, ds / |ds|)
    auto s0 = s.template subspan<0, 3>();
    auto s1 = s.template subspan<3, 3>();
    std::array ds = {s1[0] - s0[0], s1[1] - s0[1], s1[2] - s0[2]};
    const T lm = -(s0[0] * ds[0] + s0[1] * ds[1] + s0[2] * ds[2])
                 / (ds[0] * ds[0] + ds[1] * ds[1] + ds[2] * ds[2]);
    if (lm >= 0.0 and lm <= 1.0)
    {
      // The origin is between A and B
      // v = s0 + lm * (s1 - s0);
      std::array v
          = {s0[0] + lm * ds[0], s0[1] + lm * ds[1], s0[2] + lm * ds[2]};
      return {std::vector<T>(s.begin(), s.end()), v};
    }

    if (lm < 0.0)
      return {std::vector<T>(s0.begin(), s0.end()), {s0[0], s0[1], s0[2]}};
    else
      return {std::vector<T>(s1.begin(), s1.end()), {s1[0], s1[1], s1[2]}};
  }
  case 3:
  {
    auto a = s.template subspan<0, 3>();
    auto b = s.template subspan<3, 3>();
    auto c = s.template subspan<6, 3>();
    auto length = [](auto& x, auto& y)
    {
      return std::transform_reduce(x.begin(), x.end(), y.begin(), 0.0,
                                   std::plus{}, [](auto x, auto y)
                                   { return (x - y) * (x - y); });
    };
    const T ab2 = length(a, b);
    const T ac2 = length(a, c);
    const T bc2 = length(b, c);

    // Helper to compute dot(x, x - y)
    auto helper = [](auto& x, auto& y)
    {
      return std::transform_reduce(x.begin(), x.end(), y.begin(), 0.0,
                                   std::plus{},
                                   [](auto x, auto y) { return x * (x - y); });
    };
    const std::array lm
        = {helper(a, b) / ab2, helper(a, c) / ac2, helper(b, c) / bc2};

    T caba = 0;
    for (std::size_t i = 0; i < 3; ++i)
      caba += (c[i] - a[i]) * (b[i] - a[i]);

    // Calculate triangle ABC
    const T c2 = 1 - caba * caba / (ab2 * ac2);
    const T lbb = (lm[0] - lm[1] * caba / ab2) / c2;
    const T lcc = (lm[1] - lm[0] * caba / ac2) / c2;

    // Intersects triangle
    if (lbb >= 0.0 and lcc >= 0.0 and (lbb + lcc) <= 1.0)
    {
      // Calculate intersection more accurately
      // v = (c - a)  x (b - a)
      std::array dx0 = {c[0] - a[0], c[1] - a[1], c[2] - a[2]};
      std::array dx1 = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
      std::array v = math::cross(dx0, dx1);

      // Barycentre of triangle
      std::array p = {(a[0] + b[0] + c[0]) / 3, (a[1] + b[1] + c[1]) / 3,
                      (a[2] + b[2] + c[2]) / 3};

      T sum = v[0] * p[0] + v[1] * p[1] + v[2] * p[2];
      T vnorm2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
      for (std::size_t i = 0; i < 3; ++i)
        v[i] *= sum / vnorm2;

      return {std::vector(s.begin(), s.end()), v};
    }

    // Get closest point
    std::size_t pos = 0;
    {
      T norm0 = std::numeric_limits<T>::max();
      for (std::size_t i = 0; i < s.size(); i += 3)
      {
        std::span<const T, 3> p(s.data() + i, 3);
        T norm = p[0] * p[0] + p[1] * p[1] + p[2] * p[2];
        if (norm < norm0)
        {
          pos = i / 3;
          norm0 = norm;
        }
      }
    }

    std::array vmin = {s[3 * pos], s[3 * pos + 1], s[3 * pos + 2]};
    T qmin = 0;
    for (std::size_t k = 0; k < 3; ++k)
      qmin += vmin[k] * vmin[k];

    std::vector smin = {vmin[0], vmin[1], vmin[2]};

    // Check if edges are closer
    constexpr int f[3][2] = {{0, 1}, {0, 2}, {1, 2}};
    for (std::size_t i = 0; i < s_rows; ++i)
    {
      auto s0 = s.subspan(3 * f[i][0], 3);
      auto s1 = s.subspan(3 * f[i][1], 3);
      if (lm[i] > 0 and lm[i] < 1)
      {
        std::array<T, 3> v;
        for (std::size_t k = 0; k < 3; ++k)
          v[k] = s0[k] + lm[i] * (s1[k] - s0[k]);
        T qnorm = 0;
        for (std::size_t k = 0; k < 3; ++k)
          qnorm += v[k] * v[k];
        if (qnorm < qmin)
        {
          std::copy(v.begin(), v.end(), vmin.begin());
          qmin = qnorm;
          smin.resize(2 * 3);
          std::span<T, 3> smin0(smin.data(), 3);
          std::copy(s0.begin(), s0.end(), smin0.begin());
          std::span<T, 3> smin1(smin.data() + 3, 3);
          std::copy(s1.begin(), s1.end(), smin1.begin());
        }
      }
    }
    return {std::move(smin), vmin};
  }
  case 4:
  {
    auto s0 = s.template subspan<0, 3>();
    auto s1 = s.template subspan<3, 3>();
    auto s2 = s.template subspan<6, 3>();
    auto s3 = s.template subspan<9, 3>();
    auto W1 = math::cross(s0, s1);
    auto W2 = math::cross(s2, s3);

    std::array<T, 4> B;
    B[0] = std::transform_reduce(s2.begin(), s2.end(), W1.begin(), 0.0);
    B[1] = -std::transform_reduce(s3.begin(), s3.end(), W1.begin(), 0.0);
    B[2] = std::transform_reduce(s0.begin(), s0.end(), W2.begin(), 0.0);
    B[3] = -std::transform_reduce(s1.begin(), s1.end(), W2.begin(), 0.0);

    const bool signDetM = std::signbit(std::reduce(B.begin(), B.end(), 0.0));
    std::array<bool, 4> f_inside;
    for (int i = 0; i < 4; ++i)
      f_inside[i] = (std::signbit(B[i]) == signDetM);

    if (f_inside[1] and f_inside[2] and f_inside[3])
    {
      if (f_inside[0]) // The origin is inside the tetrahedron
        return {std::vector<T>(s.begin(), s.end()), {0, 0, 0}};
      else // The origin projection P faces BCD
        return nearest_simplex<T>(s.template subspan<0, 3 * 3>());
    }

    // Test ACD, ABD and/or ABC
    std::vector<T> smin;
    std::array<T, 3> vmin = {0, 0, 0};
    constexpr int facets[3][3] = {{0, 1, 3}, {0, 2, 3}, {1, 2, 3}};
    T qmin = std::numeric_limits<T>::max();
    std::vector<T> M(9);
    for (int i = 0; i < 3; ++i)
    {
      if (f_inside[i + 1] == false)
      {
        std::copy_n(std::next(s.begin(), 3 * facets[i][0]), 3, M.begin());
        std::copy_n(std::next(s.begin(), 3 * facets[i][1]), 3,
                    std::next(M.begin(), 3));
        std::copy_n(std::next(s.begin(), 3 * facets[i][2]), 3,
                    std::next(M.begin(), 6));

        const auto [snew, v] = nearest_simplex<T>(M);
        T q = std::transform_reduce(v.begin(), v.end(), v.begin(), 0);
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
  default:
    throw std::runtime_error("Number of rows defining simplex not supported.");
  }
}

/// @brief 'support' function, finds point p in bd which maximises p.v
template <std::floating_point T>
std::array<T, 3> support(std::span<const T> bd, std::array<T, 3> v)
{
  int i = 0;
  T qmax = bd[0] * v[0] + bd[1] * v[1] + bd[2] * v[2];
  for (std::size_t m = 1; m < bd.size() / 3; ++m)
  {
    T q = bd[3 * m] * v[0] + bd[3 * m + 1] * v[1] + bd[3 * m + 2] * v[2];
    if (q > qmax)
    {
      qmax = q;
      i = m;
    }
  }

  return {bd[3 * i], bd[3 * i + 1], bd[3 * i + 2]};
}
} // namespace impl_gjk

/// @brief Compute the distance between two convex bodies p and q, each
/// defined by a set of points.
///
/// Uses the Gilbert–Johnson–Keerthi (GJK) distance algorithm.
///
/// @param[in] p Body 1 list of points, shape (num_points, 3). Row-major
/// storage.
/// @param[in] q Body 2 list of points, shape (num_points, 3). Row-major
/// storage.
/// @return shortest vector between bodies
template <std::floating_point T>
std::array<T, 3> compute_distance_gjk(std::span<const T> p,
                                      std::span<const T> q)
{
  assert(p.size() % 3 == 0);
  assert(q.size() % 3 == 0);

  constexpr int maxk = 15; // Maximum number of iterations of the GJK algorithm

  // Tolerance
  constexpr T eps = 1.0e4 * std::numeric_limits<T>::epsilon();

  // Initialise vector and simplex
  std::array<T, 3> v = {p[0] - q[0], p[1] - q[1], p[2] - q[2]};
  std::vector<T> s = {v[0], v[1], v[2]};

  // Begin GJK iteration
  int k;
  for (k = 0; k < maxk; ++k)
  {
    // Support function
    std::array w1 = impl_gjk::support(p, {-v[0], -v[1], -v[2]});
    std::array w0 = impl_gjk::support(q, {v[0], v[1], v[2]});
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
    const T vnorm2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    const T vw = vnorm2 - (v[0] * w[0] + v[1] * w[1] + v[2] * w[2]);
    if (vw < (eps * vnorm2) or vw < eps)
      break;

    // Add new vertex to simplex
    s.insert(s.end(), w.begin(), w.end());

    // Find nearest subset of simplex
    auto [snew, vnew] = impl_gjk::nearest_simplex<T>(s);
    s.assign(snew.data(), snew.data() + snew.size());
    v = {vnew[0], vnew[1], vnew[2]};

    // 2nd exit condition - intersecting or touching
    if ((v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) < eps * eps)
      break;
  }

  if (k == maxk)
    throw std::runtime_error("GJK error - max iteration limit reached");

  return v;
}

} // namespace dolfinx::geometry
