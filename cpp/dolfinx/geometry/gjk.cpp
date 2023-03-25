// Copyright (C) 2020 Chris Richardson and Mattia Montinari
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "gjk.h"
#include <dolfinx/common/math.h>
#include <numeric>
#include <stdexcept>

using namespace dolfinx;

//----------------------------------------------------------------------------
// std::array<double, 3> geometry::compute_distance_gjk(std::span<const double> p,
//                                                      std::span<const double> q)
// {
//   assert(p.size() % 3 == 0);
//   assert(q.size() % 3 == 0);

//   constexpr int maxk = 10; // Maximum number of iterations of the GJK algorithm

//   // Tolerance
//   constexpr double eps = 1e-12;

//   // Initialise vector and simplex
//   std::array<double, 3> v = {p[0] - q[0], p[1] - q[1], p[2] - q[2]};
//   std::vector<double> s = {v[0], v[1], v[2]};

//   // Begin GJK iteration
//   int k;
//   for (k = 0; k < maxk; ++k)
//   {
//     // Support function
//     std::array w1 = support(p, {-v[0], -v[1], -v[2]});
//     std::array w0 = support(q, {v[0], v[1], v[2]});
//     const std::array w = {w1[0] - w0[0], w1[1] - w0[1], w1[2] - w0[2]};

//     // Break if any existing points are the same as w
//     assert(s.size() % 3 == 0);
//     std::size_t m;
//     for (m = 0; m < s.size() / 3; ++m)
//     {
//       auto it = std::next(s.begin(), 3 * m);
//       if (std::equal(it, std::next(it, 3), w.begin(), w.end()))
//         break;
//     }

//     if (m != s.size() / 3)
//       break;

//     // 1st exit condition (v - w).v = 0
//     const double vnorm2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
//     const double vw = vnorm2 - (v[0] * w[0] + v[1] * w[1] + v[2] * w[2]);
//     if (vw < (eps * vnorm2) or vw < eps)
//       break;

//     // Add new vertex to simplex
//     s.insert(s.end(), w.begin(), w.end());

//     // Find nearest subset of simplex
//     auto [snew, vnew] = nearest_simplex(s);
//     s.assign(snew.data(), snew.data() + snew.size());
//     v = {vnew[0], vnew[1], vnew[2]};

//     // 2nd exit condition - intersecting or touching
//     if ((v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) < eps * eps)
//       break;
//   }

//   if (k == maxk)
//     throw std::runtime_error("GJK error: max iteration limit reached");

//   return v;
// }
//----------------------------------------------------------------------------
