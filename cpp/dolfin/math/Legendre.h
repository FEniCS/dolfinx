// Copyright (C) 2011 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <cstddef>

namespace dolfin
{

namespace math
{

/// Interface for computing Legendre polynomials via Boost.

class Legendre
{
public:
  /// Evaluate polynomial of order n at point x
  /// @param n (std::size_t)
  ///   Order
  /// @param x (double)
  ///   Point
  /// @return double
  ///   Legendre polynomial value at x
  static double eval(std::size_t n, double x);

  /// Evaluate first derivative of polynomial of order n at point x
  /// @param n (std::size_t)
  ///   Order
  /// @param x (double)
  ///   Point
  /// @return double
  ///   Legendre polynomial derivative value at x
  static double ddx(std::size_t n, double x);

  /// Evaluate second derivative of polynomial of order n at point x
  /// @param n (std::size_t)
  ///   Order
  /// @param x (double)
  ///   Point
  /// @return double
  ///   Legendre polynomial 2nd derivative value at x
  static double d2dx(std::size_t n, double x);
};
}
}