// Copyright (C) 2006 Johan Hoffman
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
// First added:  2006-12-20
// Last changed: 2006-12-20

#ifndef __MESHHIERARCHY_H
#define __MESHHIERARCHY_H

#include <dolfin/common/types.h>

namespace dolfin
{

  class Mesh;

  /// A MeshHierarchy is a set of Mesh objects M_k, each corresponding to
  /// a refinement level k, for k=0,1,...,k_{max}.
  /// M_k for k>0 contains mesh entities of codimension 0 that are not
  /// contained in M_{k-1}, together with associated mesh entities of
  /// codimension >0.
  ///
  /// For example, the MeshHierarchy may correspond to a set of successively
  /// refined finite element meshes T_k, k=0,1,...,k_{max}, where M_0
  /// corresponds to cells, nodes and edges of an unrefined initial mesh T_0,
  /// and M_k corresponds to the cells of the mesh T_k not contained in T_{k-1},
  /// together with its nodes and edges.

  class MeshHierarchy
  {
  public:

    /// Create mesh hierarcy with initial mesh
    MeshHierarchy(const Mesh& mesh);

    /// Create empty mesh hierarcy
    MeshHierarchy();

    /// Destructor
    ~MeshHierarchy();

    /// Initialize mesh hierarchy
    void init(const Mesh& mesh);

    /// Clear mesh hierarchy
    void clear();

    /// Return number of meshes in hierarchy
    int size();

    /// Add (finest) mesh to mesh hierarchy
    void add(const Mesh& mesh);

    /// Remove (finest) mesh from mesh hierarchy
    void remove();

    /// Return reduced mesh object corresponding to level k
    Mesh& operator() (uint k) const;

    /// Return full mesh object corresponding to level k
    Mesh& operator[] (uint k) const;

  private:

    /// std::vector of meshes
    Mesh* meshes;

    /// Number of meshes in mesh hierarchy
    uint num_meshes;

  };

}

#endif
