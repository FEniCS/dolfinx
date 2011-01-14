// Copyright (C) 2010 Marie E. Rognes
// Licensed under the GNU LGPL Version 3.0 or any later version
//
// First added:  2010-10-13
// Last changed: 2011-01-04

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

  /// A _SpecialFacetFunction_ is a representation of a global
  /// function that is in P(f) for each _Facet_ f in a _Mesh_
  /// for some _FunctionSpace_ P

  template <class T> class Array;
  class Function;

  class SpecialFacetFunction : public Expression
  {

  public:

    /// Create (scalar-valued) SpecialFacetFunction
    ///
    /// *Arguments*
    ///     f_e (std::vector<_Function_*>)
    ///        Separate _Function_s for each facet
    SpecialFacetFunction(std::vector<Function>& f_e);

    /// Create (vector-valued) SpecialFacetFunction
    ///
    /// *Arguments*
    ///     f_e (std::vector<_Function_*>)
    ///        Separate _Function_s for each facet
    ///
    ///     dim (int)
    ///         The value-dimension of the Functions
    SpecialFacetFunction(std::vector<Function>& f_e, uint dim);

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
    ///     the sub-function (_Function_)
    Function& operator[] (uint i) const;

  private:

    std::vector<Function>& f_e;

  };

}
#endif
