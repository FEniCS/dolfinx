// Copyright (C) 2010 Marie E. Rognes
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
// First added:  2010-10-13
// Last changed: 2011-07-04

#ifndef __SPECIAL_FACET_FUNCTION_H
#define __SPECIAL_FACET_FUNCTION_H

#include <vector>
#include <dolfin/function/Expression.h>

namespace ufc
{
  class cell;
}

namespace dolfin
{
  template <typename T> class Array;
  class Function;

  /// A _SpecialFacetFunction_ is a representation of a global
  /// function that is in P(f) for each _Facet_ f in a _Mesh_
  /// for some _FunctionSpace_ P

  class SpecialFacetFunction : public Expression
  {
  public:

    /// Create (scalar-valued) SpecialFacetFunction
    ///
    /// *Arguments*
    ///     f_e (std::vector<_Function_>)
    ///        Separate _Function_s for each facet
    explicit SpecialFacetFunction(std::vector<Function>& f_e);

    /// Create (vector-valued) SpecialFacetFunction
    ///
    /// *Arguments*
    ///     f_e (std::vector<_Function_>)
    ///        Separate _Function_s for each facet
    ///
    ///     dim (int)
    ///         The value-dimension of the Functions
    SpecialFacetFunction(std::vector<Function>& f_e, std::size_t dim);

    /// Evaluate SpecialFacetFunction (cf _Expression_.eval)
    /// Evaluate function for given cell
    void eval(Array<double>& values, const Array<double>& x,
              const ufc::cell& cell) const;

    /// Extract sub-function i
    ///
    /// *Arguments*
    ///     i (int)
    ///        component
    ///
    /// *Returns*
    ///     _Function_
    Function& operator[] (std::size_t i) const;

  private:

    std::vector<Function>& _f_e;

  };

}
#endif
