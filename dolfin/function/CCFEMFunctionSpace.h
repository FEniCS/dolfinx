// Copyright (C) 2013 Anders Logg
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
// First added:  2013-08-05
// Last changed: 2013-08-06

#ifndef __CCFEM_FUNCTION_SPACE_H
#define __CCFEM_FUNCTION_SPACE_H

#include <vector>
#include <boost/shared_ptr.hpp>

namespace dolfin
{

  // Forward declarations
  class FunctionSpace;

  /// This class represents a cut and composite finite element
  /// function space (CCFEM) defined on one or more possibly
  /// intersecting meshes.
  ///
  /// FIXME: Document usage of class with add() followed by build()

  class CCFEMFunctionSpace
  {
  public:

    /// Create empty CCFEM function space
    CCFEMFunctionSpace();

    /// Destructor
    ~CCFEMFunctionSpace();

    /// Return dimension of CCFEM function space
    ///
    /// *Returns*
    ///     std::size_t
    ///         The dimension of the CCFEM function space.
    std::size_t dim() const;

    /// Add function space (shared pointer version)
    ///
    /// *Arguments*
    ///     function_space (_FunctionSpace_)
    ///         The function space.
    void add(boost::shared_ptr<const FunctionSpace> function_space);

    /// Add function space (reference version)
    ///
    /// *Arguments*
    ///     function_space (_FunctionSpace_)
    ///         The function space.
    void add(const FunctionSpace& mesh);

    /// Build CCFEM function space
    void build();

  private:

    // List of function spaces
    std::vector<boost::shared_ptr<const FunctionSpace> > _function_spaces;

    // Total dimension (sum of parts)
    std::size_t _dim;

  };

}

#endif
