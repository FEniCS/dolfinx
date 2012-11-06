// Copyright (C) 2007 Anders Logg
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
// First added:  2007-03-01
// Last changed: 2012-11-05

#ifndef __UFC_MESH_H
#define __UFC_MESH_H

#include <ufc.h>
#include <dolfin/mesh/Mesh.h>

namespace dolfin
{

  /// This class is simple wrapper for a UFC mesh and provides
  /// a layer between a DOLFIN mesh and a UFC mesh.

  class UFCMesh : public ufc::mesh
  {
  public:

    /// Create empty UFC mesh
    UFCMesh() : ufc::mesh()
    {
      topological_dimension = 0;
      geometric_dimension = 0;
      num_entities = 0;
    }

    /// Create UFC mesh from DOLFIN mesh
    UFCMesh(const Mesh& mesh) : ufc::mesh()
    {
      // Clear old data
      clear();

      // Set topological and geometric dimensions
      topological_dimension = mesh.topology().dim();
      geometric_dimension = mesh.geometry().dim();

      // Set number of entities of each topological dimension, using
      // the number of global entities if available (in parallel)
      num_entities = new uint[topological_dimension + 1];
      for (uint d = 0; d <= topological_dimension; d++)
        num_entities[d] = mesh.size_global(d);
    }

    /// Copy constructor
    UFCMesh(const UFCMesh& mesh) : ufc::mesh()
    {
      topological_dimension = 0;
      geometric_dimension = 0;
      num_entities = 0;
      *this = mesh;
    }

    // Assignment operator
    const UFCMesh& operator= (const UFCMesh& mesh)
    {
      // Clear all data
      clear();

      // Set topological and geometric dimensions
      topological_dimension = mesh.topological_dimension;
      geometric_dimension = mesh.geometric_dimension;

      // Set number of entities of each dimension
      num_entities = new uint[topological_dimension + 1];
      for (uint d = 0; d <= topological_dimension; d++)
        num_entities[d] = mesh.num_entities[d];

      return *this;
    }

    /// Destructor
    ~UFCMesh()
    { clear(); }

  private:

    // Clear all data
    void clear()
    {
      topological_dimension = 0;
      geometric_dimension = 0;

      delete [] num_entities;
      num_entities = 0;
    }

  };

}

#endif
