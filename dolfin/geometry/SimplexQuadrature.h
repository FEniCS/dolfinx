// Copyright (C) 2014 Anders Logg
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
// First added:  2014-02-24
// Last changed: 2016-05-04

#ifndef __SIMPLEX_QUADRATURE_H
#define __SIMPLEX_QUADRATURE_H

#include <vector>
#include "Point.h"

namespace dolfin
{

  // Forward declarations
  class Cell;

  class SimplexQuadrature
  {
  public:

    /// Compute quadrature rule for cell.
    ///
    /// *Arguments*
    ///     cell (Cell)
    ///         The cell.
    ///     order (std::size_t)
    ///         The order of convergence of the quadrature rule.
    ///
    /// *Returns*
    ///     std::pair<std::vector<double>, std::vector<double> >
    ///         A flattened array of quadrature points and a
    ///         corresponding array of quadrature weights.
    static std::pair<std::vector<double>, std::vector<double> >
    compute_quadrature_rule(const Cell& cell, std::size_t order);

    /// Compute quadrature rule for simplex.
    ///
    /// *Arguments*
    ///     coordinates (std::vector<Point>)
    ///         Vertex coordinates for the simplex
    ///     tdim (std::size_t)
    ///         The topological dimension of the simplex.
    ///     gdim (std::size_t)
    ///         The geometric dimension.
    ///     order (std::size_t)
    ///         The order of convergence of the quadrature rule.
    ///
    /// *Returns*
    ///     std::pair<std::vector<double>, std::vector<double>>
    ///         A flattened array of quadrature points and a
    ///         corresponding array of quadrature weights.
    static std::pair<std::vector<double>, std::vector<double>>
    compute_quadrature_rule(const std::vector<double>& coordinates,
                            std::size_t tdim,
                            std::size_t gdim,
                            std::size_t order);

    /// Compute quadrature rule for interval.
    ///
    /// *Arguments*
    ///     coordinates (std::vetor<Point>)
    ///         Vertex coordinates for the simplex
    ///     gdim (std::size_t)
    ///         The geometric dimension.
    ///     order (std::size_t)
    ///         The order of convergence of the quadrature rule.
    ///
    /// *Returns*
    ///     std::pair<std::vector<double>, std::vector<double>>
    ///         A flattened array of quadrature points and a
    ///         corresponding array of quadrature weights.
    static std::pair<std::vector<double>, std::vector<double>>
    compute_quadrature_rule_interval(const std::vector<double>& coordinates,
                                     std::size_t gdim,
                                     std::size_t order);

    /// Compute quadrature rule for triangle.
    ///
    /// *Arguments*
    ///     coordinates (std::vetor<Point>)
    ///         Vertex coordinates for the simplex
    ///     gdim (std::size_t)
    ///         The geometric dimension.
    ///     order (std::size_t)
    ///         The order of convergence of the quadrature rule.
    ///
    /// *Returns*
    ///     std::pair<std::vector<double>, std::vector<double>>
    ///         A flattened array of quadrature points and a
    ///         corresponding array of quadrature weights.
    static std::pair<std::vector<double>, std::vector<double>>
    compute_quadrature_rule_triangle(const std::vector<double>& coordinates,
                                     std::size_t gdim,
                                     std::size_t order);

    /// Compute quadrature rule for tetrahedron.
    ///
    /// *Arguments*
    ///     coordinates (std::vetor<Point>)
    ///         Vertex coordinates for the simplex
    ///     gdim (std::size_t)
    ///         The geometric dimension.
    ///     order (std::size_t)
    ///         The order of convergence of the quadrature rule.
    ///
    /// *Returns*
    ///     std::pair<std::vector<double>, std::vector<double>>
    ///         A flattened array of quadrature points and a
    ///         corresponding array of quadrature weights.
    static std::pair<std::vector<double>, std::vector<double>>
    compute_quadrature_rule_tetrahedron(const std::vector<double>& coordinates,
                                        std::size_t gdim,
                                        std::size_t order);

    // FIXME: Versions below should be removed when MultiMesh.cpp
    // FIXME: has been changed to work with vector<Point> instead of
    // FIXME: flattened arrays.

    static std::pair<std::vector<double>, std::vector<double>>
    compute_quadrature_rule(const double* coordinates,
                            std::size_t tdim,
                            std::size_t gdim,
                            std::size_t order);

    static std::pair<std::vector<double>, std::vector<double>>
    compute_quadrature_rule_interval(const double* coordinates,
                                     std::size_t gdim,
                                     std::size_t order);

    static std::pair<std::vector<double>, std::vector<double>>
    compute_quadrature_rule_triangle(const double* coordinates,
                                     std::size_t gdim,
                                     std::size_t order);

    static std::pair<std::vector<double>, std::vector<double>>
    compute_quadrature_rule_tetrahedron(const double* coordinates,
                                        std::size_t gdim,
                                        std::size_t order);

  };

}

#endif
