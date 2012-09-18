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
// Modified by Kristoffer Selim, 2008.
//
// First added:  2006-05-19
// Last changed: 2010-04-29

#include <dolfin/log/dolfin_log.h>
#include <dolfin/function/Function.h>
#include "MeshGeometry.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
MeshGeometry::MeshGeometry() : _dim(0), _size_higher_order(0),
  _higher_order_num_cells(0), _higher_order_num_dof(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
MeshGeometry::MeshGeometry(const MeshGeometry& geometry) : _dim(0),
  _size_higher_order(0), _higher_order_num_cells(0), _higher_order_num_dof(0)
{
  *this = geometry;
}
//-----------------------------------------------------------------------------
MeshGeometry::~MeshGeometry()
{
  clear();
}
//-----------------------------------------------------------------------------
const MeshGeometry& MeshGeometry::operator= (const MeshGeometry& geometry)
{
  // Clear old data if any
  clear();

  // Copy data
  _dim = geometry._dim;
  coordinates = geometry.coordinates;

  // higher order mesh data
  _size_higher_order         = geometry._size_higher_order;
  _higher_order_num_cells    = geometry._higher_order_num_cells;
  _higher_order_num_dof      = geometry._higher_order_num_dof;
  if ( (_size_higher_order > 0) && (_higher_order_num_cells > 0) && (_higher_order_num_dof > 0) )
  {
    /** COPY: higher order mesh data **/
    higher_order_coordinates = geometry.higher_order_coordinates;
    higher_order_cell_data = geometry.higher_order_cell_data;
    affine_cell = geometry.affine_cell;
  }
  else
  {
    _size_higher_order        = 0;
    _higher_order_num_cells   = 0;
    _higher_order_num_dof     = 0;
    higher_order_coordinates.clear();
    higher_order_cell_data.clear();
    affine_cell.clear();
  }

  return *this;
}
//-----------------------------------------------------------------------------
Point MeshGeometry::point(uint n) const
{
  double _x = 0.0;
  double _y = 0.0;
  double _z = 0.0;

  if ( _dim > 0 )
    _x = x(n, 0);
  if ( _dim > 1 )
    _y = x(n, 1);
  if ( _dim > 2 )
    _z = x(n, 2);

  Point p(_x, _y, _z);
  return p;
}
//-----------------------------------------------------------------------------
void MeshGeometry::clear()
{
  _dim  = 0;
  _size_higher_order      = 0;
  _higher_order_num_cells = 0;
  _higher_order_num_dof   = 0;

  coordinates.clear();
  higher_order_coordinates.clear();
  higher_order_cell_data.clear();
  affine_cell.clear();
}
//-----------------------------------------------------------------------------
void MeshGeometry::init(uint dim, uint size)
{
  // Delete old data if any
  clear();

  // Allocate new data
  coordinates.resize(dim*size);
  higher_order_coordinates.clear(); // this will be set by another routine

  // Save dimension and size
  _dim = dim;
  _size_higher_order = 0; // this will be set by another routine
}
//-----------------------------------------------------------------------------
void MeshGeometry::init_higher_order_vertices(uint dim, uint size_higher_order)
{
  // Allocate new data
  higher_order_coordinates.resize(dim*size_higher_order);

  // Save size
  _size_higher_order = size_higher_order;
}
//-----------------------------------------------------------------------------
void MeshGeometry::init_higher_order_cells(uint num_cells, uint num_dof)
{
  // Allocate new data
  higher_order_cell_data.resize(num_dof*num_cells);

  // Save size
  _higher_order_num_cells = num_cells;
  _higher_order_num_dof   = num_dof;
}
//-----------------------------------------------------------------------------
void MeshGeometry::init_affine_indicator(uint num_cells)
{
  affine_cell = std::vector<bool>(num_cells, true);
}
//-----------------------------------------------------------------------------
void MeshGeometry::set_affine_indicator(uint i, bool value)
{
  affine_cell[i] = value;
}
//-----------------------------------------------------------------------------
void MeshGeometry::set(uint n, uint i, double x)
{
  coordinates[n*_dim + i] = x;
}
//-----------------------------------------------------------------------------
void MeshGeometry::set_higher_order_coordinates(uint N, uint i, double x)
{
  higher_order_coordinates[N*_dim + i] = x;
}
//-----------------------------------------------------------------------------
void MeshGeometry::set_higher_order_cell_data(uint N, std::vector<uint> vector_cell_data)
{
  for (uint i = 0; i < _higher_order_num_dof; i++)
    higher_order_cell_data[N*_higher_order_num_dof + i] = vector_cell_data[i];
}
//-----------------------------------------------------------------------------
std::string MeshGeometry::str(bool verbose) const
{
  std::stringstream s;
  if (verbose)
  {
    s << str(false) << std::endl << std::endl;
    for (uint i = 0; i < size(); i++)
    {
      s << "  " << i << ":";
      for (uint d = 0; d < _dim; d++)
        s << " " << x(i, d);
      s << std::endl;
    }
    s << std::endl;
  }
  else
    s << "<MeshGeometry of dimension " << _dim << " and size " << size() << ">";

  return s.str();
}
//-----------------------------------------------------------------------------
