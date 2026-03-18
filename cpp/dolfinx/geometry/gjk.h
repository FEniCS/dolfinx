// Copyright (C) 2020-2026 Chris Richardson and Jørgen S. Dokken
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <boost/multiprecision/cpp_bin_float.hpp>
#include <cmath>
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

/// @brief Compute the four determinants, of 3x3 matrices given by all the
/// combinations of four vectors (0, 1, 2), (0, 1, 3), (0, 2, 3) and (1, 2, 3).
/// Equivalent to the determinant cofactors of an augmented 4x4 matrix of the
/// same vertices.
/// @param s Flattened array of four 3D vertices (row-major) 4x3.
/// @returns Cofactors/determinants for each set of three vertices
template <typename T>
inline std::array<T, 4> det4(const std::array<T, 12>& s)
{
  std::span<const T, 3> s0(s.begin(), 3);
  std::span<const T, 3> s1(s.begin() + 3, 3);
  std::span<const T, 3> s2(s.begin() + 6, 3);
  std::span<const T, 3> s3(s.begin() + 9, 3);

  std::array<T, 4> w;
  T c0 = s2[1] * s3[2] - s2[2] * s3[1];
  T c1 = s2[0] * s3[2] - s2[2] * s3[0];
  T c2 = s2[0] * s3[1] - s2[1] * s3[0];
  w[2] = -s0[0] * c0 + s0[1] * c1 - s0[2] * c2;
  w[3] = s1[0] * c0 - s1[1] * c1 + s1[2] * c2;

  c0 = s0[1] * s1[2] - s0[2] * s1[1];
  c1 = s0[0] * s1[2] - s0[2] * s1[0];
  c2 = s0[0] * s1[1] - s0[1] * s1[0];
  w[0] = -s2[0] * c0 + s2[1] * c1 - s2[2] * c2;
  w[1] = s3[0] * c0 - s3[1] * c1 + s3[2] * c2;

  return w;
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
/// @tparam T The scalar type of the coordinates.
/// @tparam simplex_size The number of points in the simplex (2,3 or 4)
/// @param s Simplex described by a set of points in 3D, row-major, flattened.
/// @param[in,out] coordinates Barycentric coordinates of the point in s
/// closest to the origin.
/// @note `s` may be an interval, a triangle or a tetrahedron.
template <typename T, std::size_t simplex_size>
void nearest_simplex(const std::array<T, 12>& s, std::array<T, 4>& coordinates)
{

  SPDLOG_DEBUG("GJK: nearest_simplex({})", simplex_size);

  if constexpr (simplex_size == 2)
  {
    // Simplex is an interval. Point may lie on the interval, or on either end.
    // Compute lm = dot(s0, ds / |ds|)
    std::span<const T, 3> s0(s.data(), 3);
    std::span<const T, 3> s1(s.data() + 3, 3);

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
  else if constexpr (simplex_size == 3)
  {
    // Simplex is a triangle. Point may lie in one of 7 regions (outside near a
    // vertex, outside near an edge, or on the interior)
    std::span<const T, 3> a(s.data(), 3);
    std::span<const T, 3> b(s.data() + 3, 3);
    std::span<const T, 3> c(s.data() + 6, 3);

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
  else if constexpr (simplex_size == 4)
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
    std::array<T, 4> w = det4(s);
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
  else
  {
    // Evaluated at compile-time instead of runtime!
    static_assert(simplex_size >= 2 && simplex_size <= 4,
                  "Number of rows defining simplex not supported.");
  }
}

/// @brief 'support' function, finds point p in bd which maximises p.v
/// @param bd Body described by set of 3D points, flattened
/// @param v A point in 3D
/// @returns Index of point p in `bd` which maximises p.v
template <typename T>
inline int support(std::span<const T> bd, const std::array<T, 3>& v)
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

  return i;
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

  constexpr int maxk = 15; // Maximum number of iterations of the GJK algorithm
  const U eps = 1000 * std::numeric_limits<U>::epsilon();

  // Initialize distance vector x_k
  std::array<U, 3> x_k = {static_cast<U>(p0[0]) - static_cast<U>(q0[0]),
                          static_cast<U>(p0[1]) - static_cast<U>(q0[1]),
                          static_cast<U>(p0[2]) - static_cast<U>(q0[2])};
  // Initialize simplex
  std::array<U, 12> s = {0}; // Max simplex is a tetrahedron
  s[0] = x_k[0];
  s[1] = x_k[1];
  s[2] = x_k[2];
  std::array<U, 4> lmn = {0}; // Scratch memory for barycentric
                              // coordinates of closest point in simplex
  std::size_t simplex_size = 1;
  // Begin GJK iteration
  int k;
  for (k = 0; k < maxk; ++k)
  {

    // Compute the squared norm of current iterate to normalize support search
    // in original precision
    const U x_norm2 = impl_gjk::dot3(x_k, x_k);
    std::array<U, 3> x_k_normalized = x_k;
    if (x_norm2 > eps * eps)
    {
      // ADL lookup:
      // Will use std::sqrt for double/float, and
      // boost::multiprecision::sqrt for U
      using std::sqrt;
      U inv_norm = U(1.0) / sqrt(x_norm2);
      x_k_normalized[0] *= inv_norm;
      x_k_normalized[1] *= inv_norm;
      x_k_normalized[2] *= inv_norm;
    }
    // Compute support point in original precision
    std::array<T, 3> dir_p = {static_cast<T>(-x_k_normalized[0]),
                              static_cast<T>(-x_k_normalized[1]),
                              static_cast<T>(-x_k_normalized[2])};
    std::array<T, 3> dir_q
        = {static_cast<T>(x_k_normalized[0]), static_cast<T>(x_k_normalized[1]),
           static_cast<T>(x_k_normalized[2])};
    int ip = impl_gjk::support(p0, dir_p);
    int iq = impl_gjk::support(q0, dir_q);

    // Only cast the winning support points to U
    std::array<U, 3> s_k
        = {static_cast<U>(p0[ip * 3]) - static_cast<U>(q0[iq * 3]),
           static_cast<U>(p0[ip * 3 + 1]) - static_cast<U>(q0[iq * 3 + 1]),
           static_cast<U>(p0[ip * 3 + 2]) - static_cast<U>(q0[iq * 3 + 2])};

    // Break if the newly found support point s_k is already in the simplex
    std::size_t m;
    for (m = 0; m < simplex_size; ++m)
    {
      auto it = std::next(s.begin(), 3 * m);
      if (std::equal(it, std::next(it, 3), s_k.begin(), s_k.end()))
        break;
    }

    if (m != simplex_size)
      break;

    // 1st exit condition: (x_k - s_k).x_k = 0
    const U xs_diff = x_norm2 - impl_gjk::dot3(x_k, s_k);
    if (xs_diff < (eps * x_norm2) or xs_diff < eps)
      break;

    SPDLOG_DEBUG("GJK: vw={}/{}", static_cast<double>(xs_diff),
                 static_cast<double>(eps));

    // Add new vertex to simplex
    std::ranges::copy(s_k, s.begin() + 3 * simplex_size);
    ++simplex_size;

    // Find nearest subset of simplex
    switch (simplex_size)
    {
    case 2:
      impl_gjk::nearest_simplex<U, 2>(s, lmn);
      break;
    case 3:
      impl_gjk::nearest_simplex<U, 3>(s, lmn);
      break;
    case 4:
      impl_gjk::nearest_simplex<U, 4>(s, lmn);
      break;
    default:
      throw std::runtime_error("Invalid simplex size");
    }

    // Recompute x_k and keep points with non-zero values in lmn
    std::size_t j = 0;
    x_k = {0.0, 0.0, 0.0};
    for (std::size_t i = 0; i < simplex_size; ++i)
    {
      std::span<const U> sc(std::next(s.begin(), 3 * i), 3);
      if (lmn[i] > 0.0)
      {
        x_k[0] += lmn[i] * sc[0];
        x_k[1] += lmn[i] * sc[1];
        x_k[2] += lmn[i] * sc[2];
        if (i > j)
          std::ranges::copy(sc, std::next(s.begin(), 3 * j));
        ++j;
      }
    }
    simplex_size = j;

    // 2nd exit condition - strict monotonicity
    const U x_next_norm2 = impl_gjk::dot3(x_k, x_k);
    if (x_norm2 <= x_next_norm2)
      break;

    // 3rd exit condition - intersecting or touching
    if (x_next_norm2 < eps * eps)
      break;
  }

  if (k == maxk)
    throw std::runtime_error("GJK error - max iteration limit reached");
  return {static_cast<T>(x_k[0]), static_cast<T>(x_k[1]),
          static_cast<T>(x_k[2])};
}

/// @brief Compute the distance between a sequence of convex bodies `p0, ...,
/// pN` and `q`, each defined by a set of points.
///
/// Uses the Gilbert–Johnson–Keerthi (GJK) distance algorithm.
///
/// @param[in] bodies List of the list of points that make up each of  N
/// bodies considered as body 1. `shape=(num_bodies, (num_points_body_j, 3)`.
/// Row-major storage.
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
