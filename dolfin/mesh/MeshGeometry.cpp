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
MeshGeometry::MeshGeometry() : _dim(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
MeshGeometry::MeshGeometry(const MeshGeometry& geometry) : _dim(0)
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
  // Copy data
  _dim = geometry._dim;
  coordinates = geometry.coordinates;
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
  coordinates.clear();
}
//-----------------------------------------------------------------------------
void MeshGeometry::init(uint dim, uint size)
{
  // Delete old data if any
  clear();

  // Allocate new data
  coordinates.resize(dim*size);

  // Allocate new data
  local_indices.resize(size);
  global_indices.resize(size);

  // Save dimension and size
  _dim = dim;
}
//-----------------------------------------------------------------------------
//void MeshGeometry::set(uint n, uint i, double x)
//{
//  coordinates[n*_dim + i] = x;
//  //local_indices[n.resize(size);
//  //global_indices.resize(size);
//}
//-----------------------------------------------------------------------------
void MeshGeometry::set(uint n, const std::vector<double>& x)
{
  //for (uint i = 0; i < x.size(); ++i)
  //  coordinates[n*_dim + i] = x[i];
  dolfin_assert(x.size() == _dim);
  std::copy(x.begin(), x.end(), coordinates.begin() + n*_dim);
  local_indices[n]  = n;
  global_indices[n] = n;
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
