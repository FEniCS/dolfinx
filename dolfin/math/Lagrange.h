// Copyright (C) 2003-2005 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <vector>

#include <dolfin/common/Variable.h>
#include <dolfin/log/Event.h>

namespace dolfin
{
/// Lagrange polynomial (basis) with given degree q determined by
/// n = q + 1 nodal points.
///
/// Example: q = 1 (n = 2)
///
///   Lagrange p(1);
///   p.set(0, 0.0);
///   p.set(1, 1.0);
///
/// It is the callers responsibility that the points are distinct.
///
/// This creates a Lagrange polynomial (actually two Lagrange
/// polynomials):
///
///   p(0,x) = 1 - x   (one at x = 0, zero at x = 1)
///   p(1,x) = x       (zero at x = 0, one at x = 1)
///

class Lagrange : public Variable
{
public:
  /// Constructor
  Lagrange(std::size_t q);

  /// Copy constructor
  Lagrange(const Lagrange& p);

  /// Specify point
  /// @param i (std::size_t)
  /// @param x (double)
  void set(std::size_t i, double x);

  /// Return number of points
  /// @return std::size_t
  std::size_t size() const;

  /// Return degree
  /// @return std::size_t
  std::size_t degree() const;

  /// Return point
  /// @param i (std::size_t)
  double point(std::size_t i) const;

  /// Return value of polynomial i at given point x
  /// @param i (std::size_t)
  /// @param x (double)
  double operator()(std::size_t i, double x);

  /// Return value of polynomial i at given point x
  /// @param i (std::size_t)
  /// @param x (double)
  double eval(std::size_t i, double x);

  /// Return derivate of polynomial i at given point x
  /// @param i (std::size_t)
  /// @param x (double)
  double ddx(std::size_t i, double x);

  /// Return derivative q (a constant) of polynomial
  /// @param i (std::size_t)
  double dqdx(std::size_t i);

  /// Return informal string representation (pretty-print)
  /// @param verbose (bool)
  ///   Verbosity of output string
  std::string str(bool verbose) const;

private:
  void init();

  const std::size_t _q;

  // Counts the number of time set has been called to determine when
  // init should be called
  std::size_t counter;

  std::vector<double> points;
  std::vector<double> constants;

  Event instability_detected;
};
}


