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
// First added:  2013-09-25
// Last changed: 2014-03-03

#ifndef __CCFEM_FUNCTION_H
#define __CCFEM_FUNCTION_H

#include <memory>
#include <boost/ptr_container/ptr_map.hpp>

namespace dolfin
{

  // Forward declacations
  class CCFEMFunctionSpace;
  class GenericVector;
  class Function;

  /// This class represents a function on a cut and composite finite
  /// element function space (CCFEM) defined on one or more possibly
  /// intersecting meshes.

  class CCFEMFunction
  {
  public:

    /// Create CCFEM function on given CCFEM function space
    ///
    /// *Arguments*
    ///     V (_CCFEMFunctionSpace_)
    ///         The CCFEM function space.
    ///
    /// *Example*
    ///     .. code-block:: c++
    ///
    ///         CCFEMFunction u(V);
    ///
    explicit CCFEMFunction(const CCFEMFunctionSpace& V);

    /// Create CCFEM function on given CCFEM function space (shared
    /// pointer version)
    ///
    /// *Arguments*
    ///     V (_CCFEMFunctionSpace_)
    ///         The CCFEM function space.
    explicit CCFEMFunction(std::shared_ptr<const CCFEMFunctionSpace> V);

    /// Destructor
    virtual ~CCFEMFunction();

    /// Return function (part) number i
    ///
    /// *Returns*
    ///     _Function_
    ///         Function (part) number i
    const Function& part(std::size_t i) const;

    /// Return vector of expansion coefficients (non-const version)
    ///
    /// *Returns*
    ///     _GenericVector_
    ///         The vector of expansion coefficients.
    std::shared_ptr<GenericVector> vector();

    /// Return vector of expansion coefficients (const version)
    ///
    /// *Returns*
    ///     _GenericVector_
    ///         The vector of expansion coefficients (const).
    std::shared_ptr<const GenericVector> vector() const;

  private:

    // Initialize vector
    void init_vector();

    // Compute ghost indices
    void compute_ghost_indices(std::pair<std::size_t, std::size_t> range,
                               std::vector<la_index>& ghost_indices) const;

    // The function space
    std::shared_ptr<const CCFEMFunctionSpace> _function_space;

    // The vector of expansion coefficients (local)
    std::shared_ptr<GenericVector> _vector;

    // Cache of regular functions for the parts
    mutable boost::ptr_map<std::size_t, Function> _function_parts;

  };

}

#endif
