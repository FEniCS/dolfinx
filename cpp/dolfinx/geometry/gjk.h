// Copyright (C) 2020-2026 Chris Richardson and Jørgen S. Dokken
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

/// @brief Dot product of vectors a and b, both size 3.
/// @param a Vector of size 3
/// @param b Vector of size 3
/// @returns a.b
template <typename Vec>
inline Vec::value_type dot3(const Vec& a, const Vec& b)
{
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

/// @brief Find the barycentric coordinates in the simplex `s`, of the point in
/// `s` which is closest to the origin.
/// @param s Simplex described by a set of points in 3D, row-major, flattened.
/// @param[in,out] coordinates Barycentric coordinates of the point in s
/// closest to the origin.
/// @note `s` may be an interval, a triangle or a tetrahedron.
template <typename T>
void nearest_simplex(std::span<const T> s, std::array<T, 4>& coordinates)
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

      coordinates[0] = 1.0;
      coordinates[1] = 0.0;
      return;
    }
    T mu = dot3(s1, s1) - dot3(s1, s0);
    if (mu < 0.0)
    {
      SPDLOG_DEBUG("GJK: line point B");
      coordinates[0] = 0.0;
      coordinates[1] = 1.0;
      return;
    }

    SPDLOG_DEBUG("GJK line: AB");
    T f1 = 1.0 / (lm + mu);
    coordinates[0] = mu * f1;
    coordinates[1] = lm * f1;
    return;
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
      coordinates[0] = 1.0;
      coordinates[1] = 0.0;
      coordinates[2] = 0.0;
      return;
    }

    T bb = dot3(b, b);
    T bc = dot3(b, c);
    T d3 = bb - ab;
    T d4 = bb - bc;
    if (d3 < 0.0 and d4 < 0.0)
    {
      SPDLOG_DEBUG("GJK: Point B");
      coordinates[0] = 0.0;
      coordinates[1] = 1.0;
      coordinates[2] = 0.0;
      return;
    }

    T cc = dot3(c, c);
    T d5 = cc - ac;
    T d6 = cc - bc;
    if (d5 < 0.0 and d6 < 0.0)
    {
      SPDLOG_DEBUG("GJK: Point C");
      coordinates[0] = 0.0;
      coordinates[1] = 0.0;
      coordinates[2] = 1.0;
      return;
    }

    T vc = d4 * d1 - d1 * d3 + d3 * d2;
    if (vc < 0.0 and d1 > 0.0 and d3 > 0.0)
    {
      SPDLOG_DEBUG("GJK: edge AB");
      T f1 = 1.0 / (d1 + d3);
      T lm = d1 * f1;
      T mu = d3 * f1;
      coordinates[0] = mu;
      coordinates[1] = lm;
      coordinates[2] = 0.0;
      return;
    }
    T vb = d1 * d5 - d5 * d2 + d2 * d6;
    if (vb < 0.0 and d2 > 0.0 and d5 > 0.0)
    {
      SPDLOG_DEBUG("GJK: edge AC");
      T f1 = 1.0 / (d2 + d5);
      T lm = d2 * f1;
      T mu = d5 * f1;
      coordinates[0] = mu;
      coordinates[1] = 0.0;
      coordinates[2] = lm;
      return;
    }
    T va = d3 * d6 - d6 * d4 + d4 * d5;
    if (va < 0.0 and d4 > 0.0 and d6 > 0.0)
    {
      SPDLOG_DEBUG("GJK: edge BC");
      T f1 = 1.0 / (d4 + d6);
      T lm = d4 * f1;
      T mu = d6 * f1;
      coordinates[0] = 0.0;
      coordinates[1] = mu;
      coordinates[2] = lm;
      return;
    }

    SPDLOG_DEBUG("GJK: triangle ABC");
    T f1 = 1.0 / (va + vb + vc);
    coordinates[0] = va * f1;
    coordinates[1] = vb * f1;
    coordinates[2] = vc * f1;
    return;
  }
  case 4:
  {
    // Most complex case, where simplex is a tetrahedron, with 15 possible
    // outcomes (4 vertices, 6 edges, 4 facets and the interior).
    std::ranges::fill(coordinates, 0.0);

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
        coordinates[i] = 1.0;
        return;
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
        coordinates[j0] = f1 * d[j1][j0];
        coordinates[j1] = f1 * d[j0][j1];
        return;
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
      coordinates[0] = v[2][0] * f1;
      coordinates[1] = v[4][0] * f1;
      coordinates[2] = v[5][0] * f1;
      coordinates[3] = 0.0;
      return;
    }

    if (w[1] < 0.0 and v[1][0] > 0.0 and v[3][0] > 0.0 and v[5][1] > 0.0)
    {
      T f1 = 1.0 / (v[1][0] + v[3][0] + v[5][1]);
      coordinates[0] = v[1][0] * f1;
      coordinates[1] = v[3][0] * f1;
      coordinates[2] = 0.0;
      coordinates[3] = v[5][1] * f1;
      return;
    }

    if (w[2] < 0.0 and v[0][0] > 0.0 and v[3][1] > 0 and v[4][1] > 0.0)
    {
      T f1 = 1.0 / (v[0][0] + v[3][1] + v[4][1]);
      coordinates[0] = v[0][0] * f1;
      coordinates[1] = 0.0;
      coordinates[2] = v[3][1] * f1;
      coordinates[3] = v[4][1] * f1;
      return;
    }

    if (w[3] < 0.0 and v[0][1] > 0.0 and v[1][1] > 0.0 and v[2][1] > 0.0)
    {
      T f1 = 1.0 / (v[0][1] + v[1][1] + v[2][1]);
      coordinates[0] = 0.0;
      coordinates[1] = v[0][1] * f1;
      coordinates[2] = v[1][1] * f1;
      coordinates[3] = v[2][1] * f1;
      return;
    }

    // Point lies in interior of tetrahedron with these barycentric coordinates
    coordinates[0] = w[3] / wsum;
    coordinates[1] = w[2] / wsum;
    coordinates[2] = w[1] / wsum;
    coordinates[3] = w[0] / wsum;
    return;
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
  const U eps = 1000 * std::numeric_limits<U>::epsilon();

  // Initialise vector and simplex
  std::array<U, 3> v = {p[0] - q[0], p[1] - q[1], p[2] - q[2]};
  std::array<U, 12> s = {0}; // Max simplex is a tetrahedron
  s[0] = v[0];
  s[1] = v[1];
  s[2] = v[2];
  std::array<U, 4> lmn = {0}; // Scratch memory for barycentric
                              // coordinates of nearest simplex
  std::size_t simplex_size = 1;

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
    std::size_t m;
    for (m = 0; m < simplex_size; ++m)
    {
      auto it = std::next(s.begin(), 3 * m);
      if (std::equal(it, std::next(it, 3), w.begin(), w.end()))
        break;
    }

    if (m != simplex_size)
      break;

    // 1st exit condition (v - w).v = 0
    const U vnorm2 = impl_gjk::dot3(v, v);
    const U vw = vnorm2 - impl_gjk::dot3(v, w);
    if (vw < (eps * vnorm2) or vw < eps)
      break;

    SPDLOG_DEBUG("GJK: vw={}/{}", static_cast<double>(vw),
                 static_cast<double>(eps));

    // Add new vertex to simplex
    std::copy(w.begin(), w.end(), s.begin() + 3 * simplex_size);
    ++simplex_size;

    // Find nearest subset of simplex
    impl_gjk::nearest_simplex<U>(std::span(s.begin(), 3 * simplex_size), lmn);

    // Recompute v and keep points with non-zero values in lmn
    std::size_t j = 0;
    v = {0.0, 0.0, 0.0};
    for (std::size_t i = 0; i < simplex_size; ++i)
    {
      std::span<const U> sc(std::next(s.begin(), 3 * i), 3);
      if (lmn[i] > 0.0)
      {
        v[0] += lmn[i] * sc[0];
        v[1] += lmn[i] * sc[1];
        v[2] += lmn[i] * sc[2];
        if (i > j)
          std::copy(sc.begin(), sc.end(), std::next(s.begin(), 3 * j));
        ++j;
      }
    }
    SPDLOG_DEBUG("new simplex size={}", j);
    simplex_size = j;

    // 2nd exit condition - strict monotonicity
    // Floating point can cause the algorithm to stagnate. Then we terminate.
    const U vn = impl_gjk::dot3(v, v);
    if (vnorm2 <= vn)
      break;

    // 3nd exit condition - intersecting or touching
    if (vn < eps * eps)
      break;
  }

  if (k == maxk)
    throw std::runtime_error("GJK error - max iteration limit reached");
  return {static_cast<T>(v[0]), static_cast<T>(v[1]), static_cast<T>(v[2])};
}

/// @brief Compute the distance between a sequence of convex bodies `p0, ...,
/// pN` and `q`, each defined by a set of points.
///
/// Uses the Gilbert–Johnson–Keerthi (GJK) distance algorithm.
///
/// @param[in] bodies List of the list of points that make up each of  N bodies
/// considered as body 1. `shape=(num_bodies, (num_points_body_j, 3)`. Row-major
/// storage.
/// @param[in] q Body 2 list of points, `shape=(num_points, 3)`. Row-major
/// storage.
/// @tparam T Floating point type
/// @tparam U Floating point type used for geometry computations internally,
/// which should be higher precision than T, to maintain accuracy.
/// @return For each body in `p_j`, return the shortest distance vector to
/// body 2. Shape (num_points, 3).
template <std::floating_point T,
          typename U = boost::multiprecision::cpp_bin_float_double_extended>
std::vector<T>
compute_distances_gjk(const std::vector<std::span<const T>>& bodies,
                      std::span<const T> q, size_t num_threads)
{
  size_t total_size = bodies.size();
  num_threads = std::min(num_threads, total_size);

  std::vector<T> results(total_size * 3);
  auto compute_chunk
      = [&results, &bodies](size_t c0, size_t c1, std::span<const T> q_ref)
  {
    for (size_t i = c0; i < c1; ++i)
    {
      // Using U explicitly as the internal precision type
      std::array<T, 3> dist = compute_distance_gjk<T, U>(bodies[i], q_ref);
      results[3 * i + 0] = dist[0];
      results[3 * i + 1] = dist[1];
      results[3 * i + 2] = dist[2];
    }
  };

  if (num_threads <= 1)
  {
    compute_chunk(0, total_size, q);
  }
  else
  {
    std::vector<std::jthread> threads(num_threads);
    for (size_t i = 0; i < num_threads; ++i)
    {
      auto [c0, c1] = dolfinx::MPI::local_range(i, total_size, num_threads);
      threads[i] = std::jthread(compute_chunk, c0, c1, q);
    }
  }

  return results;
}

} // namespace dolfinx::geometry
