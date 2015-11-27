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
// Last changed: 2015-11-12

#ifndef __MULTI_MESH_FUNCTION_SPACE_H
#define __MULTI_MESH_FUNCTION_SPACE_H

#include <vector>
#include <map>
#include <memory>

#include <dolfin/common/types.h>
#include <dolfin/common/Variable.h>
#include <dolfin/mesh/MultiMesh.h>

namespace dolfin
{

  // Forward declarations
  class FunctionSpace;
  class MultiMeshDofMap;

  /// This class represents a function space on a multimesh. It may
  /// may be created from a set of standard function spaces by
  /// repeatedly calling add(), followed by a call to build(). Note
  /// that a multimesh function space is not useful and its data
  /// structures are empty until build() has been called.

  class MultiMeshFunctionSpace : public Variable
  {
  public:

    /// Create multimesh function space on multimesh (referece version)
    MultiMeshFunctionSpace(const MultiMesh& multimesh);

    /// Create multimesh function space on multimesh (shared pointer version)
    MultiMeshFunctionSpace(std::shared_ptr<const MultiMesh> multimesh);

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

    /// Return view of multimesh function space for part number i.
    /// This function differs from the part() function in that it does
    /// not return the original function space for a part, but rather
    /// a view of the common multimesh function space (dofs global to
    /// the collection of parts).
    ///
    /// *Arguments*
    ///     i (std::size_t)
    ///         The part number
    ///
    /// *Returns*
    ///     _FunctionSpace_
    ///         Function space (part) number i
    std::shared_ptr<const FunctionSpace> view(std::size_t i) const;

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

    /// Build multimesh function space. This function uses offsets
    /// computed from the full function spaces on each part.
    void build(const std::vector<dolfin::la_index>& offsets);

  private:

    // Friends
    friend class MultiMeshSubSpace;

    // List of function spaces
    std::vector<std::shared_ptr<const FunctionSpace> > _function_spaces;

    // List of function space views
    std::vector<std::shared_ptr<const FunctionSpace> > _function_space_views;

    // Multimesh
    std::shared_ptr<const MultiMesh> _multimesh;

    // Multimesh dofmap
    std::shared_ptr<MultiMeshDofMap> _dofmap;

    // Quadrature order
    std::size_t _quadrature_order;

    // Build multimesh
    void _build_multimesh();

    // Build dofmap
    void _build_dofmap(const std::vector<dolfin::la_index>& offsets);

    // Build views
    void _build_views();

  };

}

#endif
