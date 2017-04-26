// Copyright (C) 2003-2011 Anders Logg
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2003-02-06
// Last changed: 2011-07-01

#ifndef __BASIC_H
#define __BASIC_H

#include <cstddef>
#include <utility>
#include <dolfin/common/constants.h>

namespace dolfin
{
  /// Return a to the power n.
  /// NOTE: Overflow is not checked!
  /// @param a (std::size_t)
  ///   Value
  /// @param n (std::size_t)
  ///   Power
  /// @return std::size_t
  std::size_t ipow(std::size_t a, std::size_t n);

  /// Return a random number, uniformly distributed between [0.0, 1.0)
  /// @return double
  double rand();

  /// Seed random number generator
  /// @param s (std::size_t)
  ///   Seed value
  void seed(std::size_t s);

  /// Check whether x is close to x0 (to within DOLFIN_EPS)
  /// @param x (double)
  ///    First value
  /// @param x0 (double)
  ///    Second value
  /// @param eps (double)
  ///   Tolerance
  /// @return bool
  bool near(double x, double x0, double eps=DOLFIN_EPS);

  // FIXME: This should not involve eps. It is 'between' and
  //        not 'nearly between'.
  /// Check whether x is between x0 and x1 (inclusive, to within DOLFIN_EPS)
  /// @param x (double)
  ///   Value to check
  /// @param range (std::pair<double, double>)
  ///   Range to check
  /// @return bool
  bool between(double x, std::pair<double, double> range);

}

#endif
