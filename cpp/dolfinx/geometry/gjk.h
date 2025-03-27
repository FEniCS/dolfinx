// Copyright (C) 2020 Chris Richardson
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <concepts>
#include <dolfinx/common/math.h>
#include <gmpxx.h>
#include <limits>
#include <numeric>
#include <span>
#include <utility>
#include <vector>

namespace dolfinx::geometry
{

namespace impl_gjk
{

template <typename T>
T det3(std::span<const T, 9> A)
{
  T w0 = A[3 + 1] * A[2 * 3 + 2] - A[3 + 2] * A[3 * 2 + 1];
  T w1 = A[3] * A[3 * 2 + 2] - A[3 + 2] * A[3 * 2];
  T w2 = A[3] * A[3 * 2 + 1] - A[3 + 1] * A[3 * 2];
  T w3 = A[0] * w0 - A[1] * w1;
  T w4 = A[2] * w2 + w3;
  return w4;
}

template <typename T>
T dot3(std::span<const T, 3> a, std::span<const T, 3> b)
{
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

/// @brief Find the resulting sub-simplex of the input simplex which is
/// nearest to the origin. Also, return the shortest vector from the
/// origin to the resulting simplex.
template <typename T>
std::vector<T> nearest_simplex(std::span<const T> s)
{

  T smax2 = 0.0;
  for (auto sv : s)
    if (sv * sv > smax2)
      smax2 = sv * sv;

  assert(s.size() % 3 == 0);
  const std::size_t s_rows = s.size() / 3;

  spdlog::info("GJK: nearest_simplex({})", s_rows);

  switch (s_rows)
  {
  case 2:
  {
    // Compute lm = dot(s0, ds / |ds|)
    std::span<const T, 3> s0 = s.template subspan<0, 3>();
    std::span<const T, 3> s1 = s.template subspan<3, 3>();
    std::array ds = {s1[0] - s0[0], s1[1] - s0[1], s1[2] - s0[2]};

    T lm = dot3(s0, s0) - dot3(s0, s1);
    if (lm < 0.0)
    {
      spdlog::info("GJK: line point A");
      return {1.0, 0.0};
    }
    T mu = dot3(s1, s1) - dot3(s1, s0);
    if (mu < 0.0)
    {
      spdlog::info("GJK: line point B");
      return {0.0, 1.0};
    }

    spdlog::info("GJK line: AB");
    T lmsum = lm + mu;
    lm /= lmsum;
    mu /= lmsum;
    return {mu, lm};
  }
  case 3:
  {
    auto a = s.template subspan<0, 3>();
    auto b = s.template subspan<3, 3>();
    auto c = s.template subspan<6, 3>();

    T d1 = (dot3(a, a) - dot3(a, b)) / smax2;
    T d2 = (dot3(a, a) - dot3(a, c)) / smax2;
    if (d1 < 0 and d2 < 0)
    {
      spdlog::info("GJK: Point A");
      return {1, 0, 0};
    }

    T d3 = (dot3(b, b) - dot3(a, b)) / smax2;
    T d4 = (dot3(b, b) - dot3(b, c)) / smax2;
    if (d3 < 0 and d4 < 0)
    {
      spdlog::info("GJK: Point B");
      return {0, 1, 0};
    }

    T d5 = (dot3(c, c) - dot3(a, c)) / smax2;
    T d6 = (dot3(c, c) - dot3(b, c)) / smax2;
    if (d5 < 0 and d6 < 0)
    {
      spdlog::info("GJK: Point C");
      return {0, 0, 1};
    }

    T vc = d4 * d1 - d1 * d3 + d3 * d2;
    if (vc < 0 and d1 > 0 and d3 > 0)
    {
      spdlog::info("GJK: edge AB");
      T f1 = 1.0 / (d1 + d3);
      T lm = f1 * d1;
      T mu = f1 * d3;
      return {mu, lm, 0};
    }
    T vb = d1 * d5 - d5 * d2 + d2 * d6;
    if (vb < 0 and d2 > 0 and d5 > 0)
    {
      spdlog::info("GJK: edge AC");
      T f1 = 1 / (d2 + d5);
      T lm = d2 * f1;
      T mu = d5 * f1;
      return {mu, 0, lm};
    }
    T va = d3 * d6 - d6 * d4 + d4 * d5;
    if (va < 0 and d4 > 0 and d6 > 0)
    {
      spdlog::info("GJK: edge BC");
      T f1 = 1 / (d4 + d6);
      T lm = d4 * f1;
      T mu = d6 * f1;
      return {0, mu, lm};
    }

    spdlog::info("GJK: triangle ABC");
    T f1 = 1.0 / (va + vb + vc);
    va *= f1;
    vb *= f1;
    vc *= f1;
    return {va, vb, vc};
  }
  case 4:
  {
    auto s0 = s.template subspan<0, 3>();
    auto s1 = s.template subspan<3, 3>();
    auto s2 = s.template subspan<6, 3>();
    auto s3 = s.template subspan<9, 3>();

    std::vector<T> rv = {0, 0, 0, 0};

    spdlog::info("d[4][4]");
    T d[4][4];
    for (int i = 0; i < 4; ++i)
    // Compute dot products at each vertex
    {
      std::span<const T, 3> si(s.begin() + i * 3, 3);
      T sii = dot3(si, si);
      bool out = true;
      for (int j = 0; j < 4; ++j)
      {
        std::span<const T, 3> sj(s.begin() + j * 3, 3);
        if (i != j)
          d[i][j] = (sii - dot3(si, sj)) / smax2;
        if (d[i][j] > 0)
          out = false;
      }
      if (out)
      {
        // Return if a corner is closest
        rv[i] = 1;
        return rv;
      }
    }

    // Check if an edge is closest
    // T vf = [&d](int i, int j, int k)
    // { return d[j][k] * d[i][j] - d[i][j] * d[j][i] + d[j][i] * d[i][k]; };

    T v[6][2] = {0};
    int edges[6][2] = {{2, 3}, {1, 3}, {1, 2}, {0, 3}, {0, 2}, {0, 1}};
    for (int i = 0; i < 6; ++i)
    {
      int j0 = edges[i][0];
      int j1 = edges[i][1];
      int j2 = edges[5 - i][0];
      int j3 = edges[5 - i][1];
      v[i][0] = d[j1][j2] * d[j0][j1] - d[j0][j1] * d[j1][j0]
                + d[j1][j0] * d[j0][j2];
      v[i][1] = d[j1][j3] * d[j0][j1] - d[j0][j1] * d[j1][j0]
                + d[j1][j0] * d[j0][j3];
      if (v[i][0] <= 0 and v[i][1] <= 0 and d[j0][j1] >= 0 and d[j1][j0] >= 0)
      {
        // On an edge
        T f1 = 1 / (d[j0][j1] + d[j1][j0]);
        rv[j0] = f1 * d[j1][j0];
        rv[j1] = f1 * d[j0][j1];
        return rv;
      }
    }

    std::array<T, 4> w;
    std::array<T, 9> M;
    std::span<const T, 9> Mspan(M.begin(), M.size());
    std::copy(s.begin(), s.begin() + 9, M.begin());
    w[0] = -det3(Mspan);
    std::copy(s.begin() + 9, s.begin() + 12, M.begin() + 6);
    w[1] = det3(Mspan);
    std::copy(s.begin() + 6, s.begin() + 9, M.begin() + 3);
    w[2] = -det3(Mspan);
    std::copy(s.begin() + 3, s.begin() + 6, M.begin() + 0);
    w[3] = det3(Mspan);
    T wsum = w[0] + w[1] + w[2] + w[3];
    if (wsum < 0.0)
    {
      w[0] = -w[0];
      w[1] = -w[1];
      w[2] = -w[2];
      w[3] = -w[3];
      wsum = -wsum;
    }

    if (w[0] < 0 and v[2][0] > 0 and v[4][0] > 0 and v[5][0] > 0)
    {
      T f1 = 1 / (v[2][0] + v[4][0] + v[5][0]);
      return {v[2][0] * f1, v[4][0] * f1, v[5][0] * f1, 0.0};
    }

    if (w[1] < 0 and v[1][0] > 0 and v[3][0] > 0 and v[5][1] > 0)
    {
      T f1 = 1 / (v[1][0] + v[3][0] + v[5][1]);
      return {v[1][0] * f1, v[3][0] * f1, 0.0, v[5][1] * f1};
    }

    if (w[2] < 0 and v[0][0] > 0 and v[3][1] > 0 and v[4][1] > 0)
    {
      T f1 = 1 / (v[0][0] + v[3][1] + v[4][1]);
      return {v[0][0] * f1, 0.0, v[3][1] * f1, v[4][1] * f1};
    }

    if (w[3] < 0 and v[0][1] > 0 and v[1][1] > 0 and v[2][1] > 0)
    {
      T f1 = 1 / (v[0][1] + v[1][1] + v[2][1]);
      return {0.0, v[0][1] * f1, v[1][1] * f1, v[2][1] * f1};
    }

    return {w[3] / wsum, w[2] / wsum, w[1] / wsum, w[0] / wsum};
  }
  default:
    throw std::runtime_error("Number of rows defining simplex not supported.");
  }
}

/// @brief 'support' function, finds point p in bd which maximises p.v
template <typename T>
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

#ifdef __GNU_MP__
using U = mpf_class;
#else
using U = double;
#endif

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
std::array<T, 3> compute_distance_gjk(std::span<const T> p0,
                                      std::span<const T> q0)
{
  assert(p0.size() % 3 == 0);
  assert(q0.size() % 3 == 0);

  // Copy from T to type U
  std::vector<U> p(p0.begin(), p0.end());
  std::vector<U> q(q0.begin(), q0.end());

  constexpr int maxk = 15; // Maximum number of iterations of the GJK algorithm

  // Tolerance
  const U eps = 1.0e4 * std::numeric_limits<T>::epsilon();

  // Initialise vector and simplex
  std::array<U, 3> v = {p[0] - q[0], p[1] - q[1], p[2] - q[2]};
  std::vector<U> s = {v[0], v[1], v[2]};

  // Begin GJK iteration
  int k;
  for (k = 0; k < maxk; ++k)
  {
    // Support function
    std::array w1
        = impl_gjk::support(std::span<const U>(p), {-v[0], -v[1], -v[2]});
    std::array w0
        = impl_gjk::support(std::span<const U>(q), {v[0], v[1], v[2]});
    const std::array<U, 3> w = {w1[0] - w0[0], w1[1] - w0[1], w1[2] - w0[2]};

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
    const U vnorm2
        = impl_gjk::dot3(std::span<const U, 3>(v), std::span<const U, 3>(v));
    const U vw
        = vnorm2
          - impl_gjk::dot3(std::span<const U, 3>(v), std::span<const U, 3>(w));
    if (vw < (eps * vnorm2) or vw < eps)
      break;

    if constexpr (std::is_same_v<U, double>)
      spdlog::info("GJK: vw={}/{}", vw, eps);
    else
      spdlog::info("GJK: vw={}/{}", vw.get_d(), eps.get_d());

    // Add new vertex to simplex
    s.insert(s.end(), w.begin(), w.end());

    std::stringstream qw;
    for (auto sv : s)
    {
      if constexpr (std::is_same_v<U, double>)
        qw << sv << ", ";
      else
        qw << sv.get_d() << ", ";
    }
    spdlog::info("s(in) = [{}]", qw.str());

    // Find nearest subset of simplex
    std::vector<U> lmn = impl_gjk::nearest_simplex<U>(s);
    std::stringstream lmns;
    for (auto q : lmn)
    {
      if constexpr (std::is_same_v<U, double>)
        lmns << q << " ";
      else
        lmns << q.get_d() << " ";
    }
    spdlog::info("lmn = {}", lmns.str());
    v = {0.0, 0.0, 0.0};
    std::vector<U> snew;
    for (std::size_t i = 0; i < lmn.size(); ++i)
    {
      std::span<U> sc(s.begin() + 3 * i, 3);
      if (lmn[i] > 0)
      {
        v[0] += lmn[i] * sc[0];
        v[1] += lmn[i] * sc[1];
        v[2] += lmn[i] * sc[2];
        snew.insert(snew.end(), sc.begin(), sc.end());
      }
    }
    spdlog::info("snew.size={}", snew.size());
    s.assign(snew.data(), snew.data() + snew.size());

    std::stringstream st;
    for (auto q : s)
    {
      if constexpr (std::is_same_v<U, double>)
        st << q << " ";
      else
        st << q.get_d() << " ";
    }
    spdlog::info("New s = {}", st.str());

    if constexpr (std::is_same_v<U, double>)
      spdlog::info("New v = [{}, {}, {}]", v[0], v[1], v[2]);

    U vn = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    //    spdlog::info("GJK: vnorm={}", vn.get_d());

    // 2nd exit condition - intersecting or touching
    if (vn < eps * eps)
      break;
  }

  if (k == maxk)
    // spdlog::info("GJK max iteration reached");
    throw std::runtime_error("GJK error - max iteration limit reached");

  std::array<T, 3> result;
  if constexpr (std::is_same_v<U, double>)
    result = {v[0], v[1], v[2]};
  else
    result = {v[0].get_d(), v[1].get_d(), v[2].get_d()};
  return result;
}

} // namespace dolfinx::geometry
