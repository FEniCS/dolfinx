// Copyright (C) 2006 Anders Logg
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
// Modified by Garth N. Wells, 2008.
//
// First added:  2006-05-08
// Last changed: 2010-11-29

#ifndef __MESH_GEOMETRY_H
#define __MESH_GEOMETRY_H

#include <string>
#include <dolfin/log/log.h>
#include "Point.h"

namespace dolfin
{

  /// MeshGeometry stores the geometry imposed on a mesh. Currently,
  /// the geometry is represented by the set of coordinates for the
  /// vertices of a mesh, but other representations are possible.

  class Function;

  class MeshGeometry
  {
  public:

    /// Create empty set of coordinates
    MeshGeometry();

    /// Copy constructor
    MeshGeometry(const MeshGeometry& geometry);

    /// Destructor
    ~MeshGeometry();

    /// Assignment
    const MeshGeometry& operator= (const MeshGeometry& geometry);

    /// Return Euclidean dimension of coordinate system
    std::size_t dim() const
    { return _dim; }

    /// Return number of coordinates
    std::size_t size() const
    {
      dolfin_assert(coordinates.size() % _dim == 0);
      return coordinates.size()/_dim;
    }

    /// Return value of coordinate with local index n in direction i
    double& x(std::size_t n, std::size_t i)
    {
      dolfin_assert(n < local_index_to_position.size());
      dolfin_assert(i < _dim);
      return coordinates[local_index_to_position[n]*_dim + i];
    }

    /// Return value of coordinate with local index n in direction i
    double x(std::size_t n, std::size_t i) const
    {
      dolfin_assert(n < local_index_to_position.size());
      dolfin_assert(i < _dim);
      return coordinates[local_index_to_position[n]*_dim + i];
    }

    /// Return array of values for coordinate with local index n
    double* x(std::size_t n)
    {
      dolfin_assert(n < local_index_to_position.size());
      return &coordinates[local_index_to_position[n]*_dim];
    }

    /// Return array of values for coordinate with local index n
    const double* x(std::size_t n) const
    {
      dolfin_assert(n < local_index_to_position.size());
      return &coordinates[local_index_to_position[n]*_dim];
    }

    /// Return array of values for all coordinates
    std::vector<double>& x()
    { return coordinates; }

    /// Return array of values for all coordinates
    const std::vector<double>& x() const
    { return coordinates; }

    /// Return coordinate with local index n as a 3D point value
    Point point(std::size_t n) const;

    /// Clear all data
    void clear();

    /// Initialize coordinate list to given dimension and size
    void init(std::size_t dim, std::size_t size);

    /// Set value of coordinate
    //void set(std::size_t n, std::size_t i, double x);
    void set(std::size_t local_index, const std::vector<double>& x);

    /// Hash of coordinate values
    ///
    /// *Returns*
    ///     std::size_t
    ///         A tree-hashed value of the coordinates over all MPI processes
    ///
    std::size_t hash() const;

    /// Return informal string representation (pretty-print)
    std::string str(bool verbose) const;

  private:

    // Friends
    friend class BinaryFile;
    friend class MeshRenumbering;

    // Euclidean dimension
    std::size_t _dim;

    // Coordinates for all vertices stored as a contiguous array
    std::vector<double> coordinates;

    // Local coordinate indices (array position -> index)
    std::vector<unsigned int> position_to_local_index;

    // Local coordinate indices (local index -> array position)
    std::vector<unsigned int> local_index_to_position;

  };

}

#endif
