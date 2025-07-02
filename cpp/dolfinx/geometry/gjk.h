// Copyright (C) 2020 Chris Richardson
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <boost/multiprecision/cpp_bin_float.hpp>
#include <concepts>
#include <limits>
#include <numeric>
#include <span>
#include <utility>
#include <vector>

namespace dolfinx::geometry
{

namespace impl_gjk
{

/// @brief Determinant of 3x3 matrix A
/// @param A 3x3 matrix
/// @returns det(A)
template <typename T>
inline T det3(std::span<const T, 9> A)
{
  T w0 = A[3 + 1] * A[2 * 3 + 2] - A[3 + 2] * A[3 * 2 + 1];
  T w1 = A[3] * A[3 * 2 + 2] - A[3 + 2] * A[3 * 2];
  T w2 = A[3] * A[3 * 2 + 1] - A[3 + 1] * A[3 * 2];
  return A[0] * w0 - A[1] * w1 + A[2] * w2;
}

/// @brief Dot product of vectors a and b
/// @param a
/// @param b
/// @returns a.b
template <typename Vec>
inline Vec::value_type dot3(const Vec& a, const Vec& b)
{
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

/// @brief Find the barycentric coordinates in the simplex `s`, of the point in
/// `s` which is closest to the origin.
/// @param s Simplex described by a set of points in 3D, row-major, flattened.
/// @return Barycentric coordinates of the point in s closest to the origin.
/// @note `s` may be an interval, a triangle or a tetrahedron.
template <typename T>
std::vector<T> nearest_simplex(std::span<const T> s)
{
  assert(s.size() % 3 == 0);
  const std::size_t s_rows = s.size() / 3;

  SPDLOG_DEBUG("GJK: nearest_simplex({})", s_rows);

  switch (s_rows)
  {
  case 2:
  {
    // Simplex is an interval. Point may lie on the interval, or on either end.
    // Compute lm = dot(s0, ds / |ds|)
    std::span<const T, 3> s0 = s.template subspan<0, 3>();
    std::span<const T, 3> s1 = s.template subspan<3, 3>();

    T lm = dot3(s0, s0) - dot3(s0, s1);
    if (lm < 0.0)
    {
      SPDLOG_DEBUG("GJK: line point A");
      return {1.0, 0.0};
    }
    T mu = dot3(s1, s1) - dot3(s1, s0);
    if (mu < 0.0)
    {
      SPDLOG_DEBUG("GJK: line point B");
      return {0.0, 1.0};
    }

    SPDLOG_DEBUG("GJK line: AB");
    T f1 = 1.0 / (lm + mu);
    return {mu * f1, lm * f1};
  }
  case 3:
  {
    // Simplex is a triangle. Point may lie in one of 7 regions (outside near a
    // vertex, outside near an edge, or on the interior)
    auto a = s.template subspan<0, 3>();
    auto b = s.template subspan<3, 3>();
    auto c = s.template subspan<6, 3>();

    T aa = dot3(a, a);
    T ab = dot3(a, b);
    T ac = dot3(a, c);
    T d1 = aa - ab;
    T d2 = aa - ac;
    if (d1 < 0.0 and d2 < 0.0)
    {
      SPDLOG_DEBUG("GJK: Point A");
      return {1.0, 0.0, 0.0};
    }

    T bb = dot3(b, b);
    T bc = dot3(b, c);
    T d3 = bb - ab;
    T d4 = bb - bc;
    if (d3 < 0.0 and d4 < 0.0)
    {
      SPDLOG_DEBUG("GJK: Point B");
      return {0.0, 1.0, 0.0};
    }

    T cc = dot3(c, c);
    T d5 = cc - ac;
    T d6 = cc - bc;
    if (d5 < 0.0 and d6 < 0.0)
    {
      SPDLOG_DEBUG("GJK: Point C");
      return {0.0, 0.0, 1.0};
    }

    T vc = d4 * d1 - d1 * d3 + d3 * d2;
    if (vc < 0.0 and d1 > 0.0 and d3 > 0.0)
    {
      SPDLOG_DEBUG("GJK: edge AB");
      T f1 = 1.0 / (d1 + d3);
      T lm = d1 * f1;
      T mu = d3 * f1;
      return {mu, lm, 0.0};
    }
    T vb = d1 * d5 - d5 * d2 + d2 * d6;
    if (vb < 0.0 and d2 > 0.0 and d5 > 0.0)
    {
      SPDLOG_DEBUG("GJK: edge AC");
      T f1 = 1.0 / (d2 + d5);
      T lm = d2 * f1;
      T mu = d5 * f1;
      return {mu, 0.0, lm};
    }
    T va = d3 * d6 - d6 * d4 + d4 * d5;
    if (va < 0.0 and d4 > 0.0 and d6 > 0.0)
    {
      SPDLOG_DEBUG("GJK: edge BC");
      T f1 = 1.0 / (d4 + d6);
      T lm = d4 * f1;
      T mu = d6 * f1;
      return {0.0, mu, lm};
    }

    SPDLOG_DEBUG("GJK: triangle ABC");
    T f1 = 1.0 / (va + vb + vc);
    return {va * f1, vb * f1, vc * f1};
  }
  case 4:
  {
    // Most complex case, where simplex is a tetrahedron, with 15 possible
    // outcomes (4 vertices, 6 edges, 4 facets and the interior).
    std::vector<T> rv = {0.0, 0.0, 0.0, 0.0};

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
          d[i][j] = (sii - dot3(si, sj));
        SPDLOG_DEBUG("d[{}][{}] = {}", i, j, static_cast<double>(d[i][j]));
        if (d[i][j] > 0.0)
          out = false;
      }
      if (out)
      {
        // Return if a vertex is closest
        rv[i] = 1.0;
        return rv;
      }
    }

    SPDLOG_DEBUG("Check for edges");

    // Check if an edge is closest
    T v[6][2] = {{0.0}};
    int edges[6][2] = {{2, 3}, {1, 3}, {1, 2}, {0, 3}, {0, 2}, {0, 1}};
    for (int i = 0; i < 6; ++i)
    {
      // Four vertices of the tetrahedron, j0 and j1 at the ends of the current
      // edge and j2  and j3 on the opposing edge.
      int j0 = edges[i][0];
      int j1 = edges[i][1];
      int j2 = edges[5 - i][0];
      int j3 = edges[5 - i][1];
      v[i][0] = d[j1][j2] * d[j0][j1] - d[j0][j1] * d[j1][j0]
                + d[j1][j0] * d[j0][j2];
      v[i][1] = d[j1][j3] * d[j0][j1] - d[j0][j1] * d[j1][j0]
                + d[j1][j0] * d[j0][j3];

      SPDLOG_DEBUG("v[{}] = {},{}", i, (double)v[i][0], (double)v[i][1]);
      if (v[i][0] <= 0.0 and v[i][1] <= 0.0 and d[j0][j1] >= 0.0
          and d[j1][j0] >= 0.0)
      {
        // On an edge
        T f1 = 1.0 / (d[j0][j1] + d[j1][j0]);
        rv[j0] = f1 * d[j1][j0];
        rv[j1] = f1 * d[j0][j1];
        return rv;
      }
    }

    // Now check the facets of a tetrahedron
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

    if (w[0] < 0.0 and v[2][0] > 0.0 and v[4][0] > 0.0 and v[5][0] > 0.0)
    {
      T f1 = 1.0 / (v[2][0] + v[4][0] + v[5][0]);
      return {v[2][0] * f1, v[4][0] * f1, v[5][0] * f1, 0.0};
    }

    if (w[1] < 0.0 and v[1][0] > 0.0 and v[3][0] > 0.0 and v[5][1] > 0.0)
    {
      T f1 = 1.0 / (v[1][0] + v[3][0] + v[5][1]);
      return {v[1][0] * f1, v[3][0] * f1, 0.0, v[5][1] * f1};
    }

    if (w[2] < 0.0 and v[0][0] > 0.0 and v[3][1] > 0 and v[4][1] > 0.0)
    {
      T f1 = 1.0 / (v[0][0] + v[3][1] + v[4][1]);
      return {v[0][0] * f1, 0.0, v[3][1] * f1, v[4][1] * f1};
    }

    if (w[3] < 0.0 and v[0][1] > 0.0 and v[1][1] > 0.0 and v[2][1] > 0.0)
    {
      T f1 = 1.0 / (v[0][1] + v[1][1] + v[2][1]);
      return {0.0, v[0][1] * f1, v[1][1] * f1, v[2][1] * f1};
    }

    // Point lies in interior of tetrahedron with these barycentric coordinates
    return {w[3] / wsum, w[2] / wsum, w[1] / wsum, w[0] / wsum};
  }
  default:
    throw std::runtime_error("Number of rows defining simplex not supported.");
  }
}

/// @brief 'support' function, finds point p in bd which maximises p.v
/// @param bd Body described by set of 3D points, flattened
/// @param v A point in 3D
/// @returns Point p in `bd` which maximises p.v
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

/// @brief Compute the distance between two convex bodies `p0` and `q0`, each
/// defined by a set of points.
///
/// Uses the Gilbert–Johnson–Keerthi (GJK) distance algorithm.
///
/// @param[in] p0 Body 1 list of points, `shape=(num_points, 3)`. Row-major
/// storage.
/// @param[in] q0 Body 2 list of points, `shape=(num_points, 3)`. Row-major
/// storage.
/// @tparam T Floating point type
/// @tparam U Floating point type used for geometry computations internally,
/// which should be higher precision than T, to maintain accuracy.
/// @return shortest vector between bodies
template <std::floating_point T,
          typename U = boost::multiprecision::cpp_bin_float_double_extended>
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
    const U vnorm2 = impl_gjk::dot3(v, v);
    const U vw = vnorm2 - impl_gjk::dot3(v, w);
    if (vw < (eps * vnorm2) or vw < eps)
      break;

    SPDLOG_DEBUG("GJK: vw={}/{}", static_cast<double>(vw),
                 static_cast<double>(eps));

    // Add new vertex to simplex
    s.insert(s.end(), w.begin(), w.end());

    // Find nearest subset of simplex
    std::vector<U> lmn = impl_gjk::nearest_simplex<U>(s);
    v = {0.0, 0.0, 0.0};
    std::vector<U> snew;
    snew.reserve(12); // maximum size
    for (std::size_t i = 0; i < lmn.size(); ++i)
    {
      std::span<U> sc(s.begin() + 3 * i, 3);
      if (lmn[i] > 0.0)
      {
        v[0] += lmn[i] * sc[0];
        v[1] += lmn[i] * sc[1];
        v[2] += lmn[i] * sc[2];
        snew.insert(snew.end(), sc.begin(), sc.end());
      }
    }
    SPDLOG_DEBUG("snew.size={}", snew.size());
    s.assign(snew.data(), snew.data() + snew.size());

    U vn = impl_gjk::dot3(v, v);

    // 2nd exit condition - intersecting or touching
    if (vn < eps * eps)
      break;
  }

  if (k == maxk)
    throw std::runtime_error("GJK error - max iteration limit reached");

  return {static_cast<T>(v[0]), static_cast<T>(v[1]), static_cast<T>(v[2])};
}

} // namespace dolfinx::geometry
