// Copyright (C) 2014 Mikael Mortensen
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

#ifndef __LAGRANGE_INTERPOLATOR_H
#define __LAGRANGE_INTERPOLATOR_H

#include <algorithm>
#include <map>
#include <unordered_map>
#include <vector>

namespace dolfin
{

  class Expression;
  class Function;
  class FunctionSpace;
  class GenericDofMap;
  class Mesh;

  /// This class interpolates efficiently from a GenericFunction to a
  /// Lagrange Function

  class LagrangeInterpolator
  {
  public:

    /// Interpolate Expression
    ///
    /// *Arguments*
    ///     u  (_Function_)
    ///         The resulting Function
    ///     u0 (_Expression_)
    ///         The Expression to be interpolated.
    static void interpolate(Function& u, const Expression& u0);

    /// Interpolate function (on possibly non-matching meshes)
    ///
    /// *Arguments*
    ///     u  (_Function_)
    ///         The resulting Function
    ///     u0 (_Function_)
    ///         The Function to be interpolated.
    static void interpolate(Function& u, const Function& u0);

  private:

    // Comparison operator for hashing coordinates. Note that two
    // coordinates are considered equal if equal to within specified
    // tolerance.
    struct lt_coordinate
    {
      lt_coordinate(double tolerance) : TOL(tolerance) {}

      bool operator() (const std::vector<double>& x,
                       const std::vector<double>& y) const
      {
        const std::size_t n = std::max(x.size(), y.size());
        for (std::size_t i = 0; i < n; ++i)
        {
          double xx = 0.0;
          double yy = 0.0;
          if (i < x.size())
            xx = x[i];
          if (i < y.size())
            yy = y[i];

          if (xx < (yy - TOL))
            return true;
          else if (xx > (yy + TOL))
            return false;
        }
        return false;
      }

      // Tolerance
      const double TOL;
    };

    // Create a map from coordinates to a list of dofs that share the
    // coordinate
    static std::map<std::vector<double>, std::vector<std::size_t>,
                    lt_coordinate>
    tabulate_coordinates_to_dofs(const FunctionSpace& V);

    // Create a map from dof to its component index in Mixed Space
    static void extract_dof_component_map(std::unordered_map<std::size_t,
                                          std::size_t>& dof_component_map,
                                          const FunctionSpace& V,
                                          int* component);

    // Return true if point lies within bounding box
    static bool in_bounding_box(const std::vector<double>& point,
                                const std::vector<double>& bounding_box,
                                const double tol);

  };

}

#endif
