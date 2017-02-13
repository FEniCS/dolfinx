// Copyright (C) 2013-2016 Anders Logg
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
// Last changed: 2016-03-02

#ifndef __MULTI_MESH_FUNCTION_H
#define __MULTI_MESH_FUNCTION_H

#include <memory>
#include <boost/ptr_container/ptr_map.hpp>
#include <dolfin/common/Variable.h>

namespace dolfin
{

  // Forward declarations
  class MultiMeshFunctionSpace;
  class GenericVector;
  class Function;
  class MultiMeshFunction;

  /// This class represents a function on a cut and composite finite
  /// element function space (MultiMesh) defined on one or more possibly
  /// intersecting meshes.

  class MultiMeshFunction : public Variable
  {
  public:

    /// Constructor
    MultiMeshFunction();

    /// Create MultiMesh function on given MultiMesh function space
    ///
    /// *Arguments*
    ///     V (_MultiMeshFunctionSpace_)
    ///         The MultiMesh function space.
    explicit MultiMeshFunction(std::shared_ptr<const MultiMeshFunctionSpace> V);

    /// Create MultiMesh function on given MultiMesh function space with a given vector
    /// (shared data)
    ///
    /// *Warning: This constructor is intended for internal library use only*
    ///
    /// *Arguments*
    ///     V (_MultiMeshFunctionSpace_)
    ///         The multimesh function space.
    ///     x (_GenericVector_)
    ///         The vector.
    MultiMeshFunction(std::shared_ptr<const MultiMeshFunctionSpace> V,
		      std::shared_ptr<GenericVector> x);

    /// Destructor
    virtual ~MultiMeshFunction();

    /// Assign Function to part of a mesh
    ///
    /// *Arguments*
    ///     a (int)
    ///         Part mesh assigned to
    ///     V (_Function_)
    ///         The vector
    void assign_part(std::size_t a, Function& v);

    /// Extract the vector of a MultiMeshFunction to part of a mesh
    ///
    /// *Arguments*
    ///     part (int)
    ///         Part of function wanted for extraction
    ///     v (_Function_)
    ///         Vector to assign put extraction
    Function extract_vector(std::size_t part, Function& v);

    /// Return function (part) number i
    ///
    /// *Returns*
    ///     _Function_
    ///         Function (part) number i
    std::shared_ptr<const Function> part(std::size_t i) const;

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

    /// Return shared pointer to multi mesh function space
    ///
    /// *Returns*
    ///     _MultiMeshFunctionSpace_
    ///         Return the shared pointer.
    virtual std::shared_ptr<const MultiMeshFunctionSpace> function_space() const
    {
      return _function_space;
    }

  private:

    // Initialize vector
    void init_vector();

    // Compute ghost indices
    void compute_ghost_indices(std::pair<std::size_t, std::size_t> range,
                               std::vector<la_index>& ghost_indices) const;

    // The function space
    std::shared_ptr<const MultiMeshFunctionSpace> _function_space;

    // The vector of expansion coefficients (local)
    std::shared_ptr<GenericVector> _vector;

    // Cache of regular functions for the parts
    mutable std::map<std::size_t, std::shared_ptr<const Function> > _function_parts;

  };

}

#endif
