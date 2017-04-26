// Copyright (C) 2003-2005 Anders Logg
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
// First added:  2003-06-12
// Last changed: 2009-09-08

#ifndef __LAGRANGE_H
#define __LAGRANGE_H

#include <vector>

#include <dolfin/log/Event.h>
#include <dolfin/common/Variable.h>

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
    double operator() (std::size_t i, double x);

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

#endif
