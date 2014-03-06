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
// Last changed: 2014-03-03

#ifndef __MultiMesh_FUNCTION_SPACE_H
#define __MultiMesh_FUNCTION_SPACE_H

#include <vector>
#include <map>
#include <memory>

namespace dolfin
{

  // Forward declarations
  class FunctionSpace;
  class MultiMeshDofMap;
  class MultiMesh;

  /// This class represents a function space on a multimesh. It may
  /// may be created from a set of standard function spaces by
  /// repeatedly calling add(), followed by a call to build(). Note
  /// that a multimesh function space is not useful and its data
  /// structures are empty until build() has been called.

  class MultiMeshFunctionSpace
  {
  public:

    /// Create empty multimesh function space
    MultiMeshFunctionSpace();

    /// Destructor
    ~MultiMeshFunctionSpace();

    /// Return dimension of the multimesh function space
    ///
    /// *Returns*
    ///     std::size_t
    ///         The dimension of the multimesh function space.
    std::size_t dim() const;

    /// Return multimesh
    ///
    /// *Returns*
    ///     _MultiMesh_
    ///         The multimesh.
    std::shared_ptr<const MultiMesh> multimesh() const;

    /// Return multimesh dofmap
    ///
    /// *Returns*
    ///     _MultiMeshDofMap_
    ///         The dofmap.
    std::shared_ptr<const MultiMeshDofMap> dofmap() const;

    /// Return the number of function spaces (parts) of the multimesh function space
    ///
    /// *Returns*
    ///     std::size_t
    ///         The number of function spaces (parts) of the multimesh function space.
    std::size_t num_parts() const;

    /// Return function space (part) number i
    ///
    /// *Arguments*
    ///     i (std::size_t)
    ///         The part number
    ///
    /// *Returns*
    ///     _FunctionSpace_
    ///         Function space (part) number i
    std::shared_ptr<const FunctionSpace> part(std::size_t i) const;

    /// Add function space (shared pointer version)
    ///
    /// *Arguments*
    ///     function_space (_FunctionSpace_)
    ///         The function space.
    void add(std::shared_ptr<const FunctionSpace> function_space);

    /// Add function space (reference version)
    ///
    /// *Arguments*
    ///     function_space (_FunctionSpace_)
    ///         The function space.
    void add(const FunctionSpace& function_space);

    /// Build multimesh function space
    void build();

  private:

    // List of function spaces
    std::vector<std::shared_ptr<const FunctionSpace> > _function_spaces;

    // Multimesh
    std::shared_ptr<MultiMesh> _multimesh;

    // Multimesh dofmap
    std::shared_ptr<MultiMeshDofMap> _dofmap;

    // Build multimesh
    void _build_multimesh();

    // Build dofmap
    void _build_dofmap();

  };

}

#endif
