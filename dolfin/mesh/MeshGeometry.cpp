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

#include <boost/functional/hash.hpp>

#include <dolfin/common/MPI.h>
#include <dolfin/log/dolfin_log.h>
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
  coordinates             = geometry.coordinates;
  position_to_local_index = geometry.position_to_local_index;
  local_index_to_position = geometry.local_index_to_position;

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
  position_to_local_index.clear();
  local_index_to_position.clear();
}
//-----------------------------------------------------------------------------
void MeshGeometry::init(uint dim, uint size)
{
  // Delete old data if any
  clear();

  // Allocate new data
  coordinates.resize(dim*size);

  // Allocate new data
  position_to_local_index.resize(size);
  local_index_to_position.resize(size);

  // Save dimension and size
  _dim = dim;
}
//-----------------------------------------------------------------------------
void MeshGeometry::set(uint local_index,
                       const std::vector<double>& x)
{
  dolfin_assert(x.size() == _dim);
  std::copy(x.begin(), x.end(), coordinates.begin() + local_index*_dim);

  dolfin_assert(local_index < position_to_local_index.size());
  position_to_local_index[local_index] = local_index;

  dolfin_assert(local_index < local_index_to_position.size());
  local_index_to_position[local_index] = local_index;

}
//-----------------------------------------------------------------------------
uint MeshGeometry::hash() const
{
  // Compute local hash
  boost::hash<std::vector<double> > dhash;
  const uint local_hash = dhash(coordinates);

  // Gather hash keys from all processes
  std::vector<uint> all_hashes;
  MPI::gather(local_hash, all_hashes);

  // Hash the received hash keys
  boost::hash<std::vector<uint> > uhash;
  uint global_hash = uhash(all_hashes);

  // Broadcast hash key
  MPI::broadcast(global_hash);

  return global_hash;
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
