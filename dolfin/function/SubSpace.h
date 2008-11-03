// Copyright (C) 2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-11-03
// Last changed: 2008-11-03

#ifndef __SUB_SPACE_H
#define __SUB_SPACE_H

#include <vector>
#include "FunctionSpace.h"

namespace dolfin
{

  /// This class represents a subspace (component) of a function space.
  ///
  /// The subspace is specified by an array of indices. For example,
  /// the array [3, 0, 2] specifies subspace 2 of subspace 0 of
  /// subspace 3.
  ///
  /// A typical example is the function space W = V x P for Stokes.
  /// Here, V = W[0] is the subspace for the velocity component and
  /// P = W[1] is the subspace for the pressure component. Furthermore,
  /// W[0][0] = V[0] is the first component of the velocity space etc.

  class SubSpace : public FunctionSpace
  {
  public:

    /// Create subspace for given component (one level)
    SubSpace(const FunctionSpace& V,
             uint component);

    /// Create subspace for given component (two levels)
    SubSpace(const FunctionSpace& V,
             uint component, uint sub_component);

    /// Create subspace for given component (n levels)
    SubSpace(const FunctionSpace& V,
             const std::vector<uint>& component);

  };

}

#endif
