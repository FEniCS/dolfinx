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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Garth N. Wells, 2008.
//
// First added:  2006-05-08
// Last changed: 2010-11-29

#ifndef __MESH_GEOMETRY_H
#define __MESH_GEOMETRY_H

#include "Point.h"
#include <dolfin/common/types.h>

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
    uint dim() const { return _dim; }

    /// Return number of coordinates
    uint size() const { return _size; }

    /// Return value of coordinate n in direction i
    double& x(uint n, uint i) { assert(n < _size && i < _dim); return coordinates[n*_dim + i]; }

    /// Return value of coordinate n in direction i
    double x(uint n, uint i) const { assert(n < _size && i < _dim); return coordinates[n*_dim + i]; }

    /// Return array of values for coordinate n
    double* x(uint n) { return coordinates + n*_dim; }

    /// Return array of values for coordinate n
    const double* x(uint n) const { return coordinates + n*_dim; }

    /// Return array of values for all coordinates
    double* x() { return coordinates; }

    /// Return array of values for all coordinates
    const double* x() const { return coordinates; }

    /// Return array of values for higher order coordinate n
    double* higher_order_x(uint n) { return higher_order_coordinates + n*_dim; }

    /// Return array of values for higher order coordinate n
    const double* higher_order_x(uint n) const { return higher_order_coordinates + n*_dim; }

    /// Return array of values for all higher order coordinates
    double* higher_order_x() { return higher_order_coordinates; }

    /// Return array of values for all higher order coordinates
    const double* higher_order_x() const { return higher_order_coordinates; }

    /// Return number of vertices used (per cell) to represent the higher order geometry
    uint num_higher_order_vertices_per_cell() const { return _higher_order_num_dof; }

    /// Return array of higher order vertex indices for a specific higher order cell
    uint* higher_order_cell(uint c)
    { return (higher_order_cell_data + (c*_higher_order_num_dof)); }

    /// Return array of higher order vertex indices for a specific higher order cell
    const uint* higher_order_cell(uint c) const
    { return (higher_order_cell_data + (c*_higher_order_num_dof)); }

    /// Return array of values for all higher order cell data
    uint* higher_order_cells() { return higher_order_cell_data; }

    /// Return array of values for all higher order cell data
    const uint* higher_order_cells() const { return higher_order_cell_data; }

    /// Return coordinate n as a 3D point value
    Point point(uint n) const;

    /// Return pointer to boolean affine indicator array
    bool* affine_cell_bool() { return affine_cell; }

    /// Clear all data
    void clear();

    /// Initialize coordinate list to given dimension and size
    void init(uint dim, uint size);

    /// Initialize higher order coordinate list to given dimension and size
    void init_higher_order_vertices(uint dim, uint size_higher_order);

    /// Initialize higher order cell data list to given number of cells and dofs
    void init_higher_order_cells(uint num_cells, uint num_dof);

    /// Initialize the affine indicator array
    void init_affine_indicator(uint num_cells);

    /// set affine indicator at index i
    void set_affine_indicator(uint i, bool value);

    /// Set value of coordinate n in direction i
    void set(uint n, uint i, double x);

    /// Set value of higher order coordinate N in direction i
    void set_higher_order_coordinates(uint N, uint i, double x);

    /// Set higher order cell data for cell # N in direction i
    void set_higher_order_cell_data(uint N, std::vector<uint> vector_cell_data);

    /// Return informal string representation (pretty-print)
    std::string str(bool verbose) const;

  private:

    // Friends
    friend class BinaryFile;
    friend class MeshRenumbering;

    // Euclidean dimension
    uint _dim;

    // Number of coordinates
    uint _size;

    // Coordinates for all vertices stored as a contiguous array
    double* coordinates;

    // Number of higher order coordinates
    uint _size_higher_order;

    // Higher order mesh coordinates (stored just like coordinates)
    // note: this may seem redundant, but needs to stay this way!
    double* higher_order_coordinates;

    // should eventually have some kind of indicator for the TYPE of higher order cell data!
    // i.e. P2 Lagrange, etc...  For now we will assume P2 Lagrange only!

    // Higher order cell size info
    uint _higher_order_num_cells;
    uint _higher_order_num_dof;

    // Higher order cell data
    // note: this may seem redundant, but needs to stay this way!
    uint* higher_order_cell_data;

    // Boolean indicator for whether a cell is affinely mapped (or not), i.e. straight or not.
    // note: this is used in conjunction with the higher order stuff
    bool* affine_cell;

  };

}

#endif
