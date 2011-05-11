// Copyright (C) 2006-2009 Anders Logg
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
// First added:  2006-05-08
// Last changed: 2010-11-25

#include <dolfin/log/dolfin_log.h>
#include <dolfin/common/utils.h>
#include "MeshConnectivity.h"
#include "MeshTopology.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
MeshTopology::MeshTopology()
  : _dim(0), num_entities(0), connectivity(0)

{
  // Do nothing
}
//-----------------------------------------------------------------------------
MeshTopology::MeshTopology(const MeshTopology& topology)
  : _dim(0), num_entities(0), connectivity(0)
{
  *this = topology;
}
//-----------------------------------------------------------------------------
MeshTopology::~MeshTopology()
{
  clear();
}
//-----------------------------------------------------------------------------
const MeshTopology& MeshTopology::operator= (const MeshTopology& topology)
{
  // Clear old data if any
  clear();

  // Allocate data
  _dim = topology._dim;
  num_entities = new uint[_dim + 1];
  connectivity = new MeshConnectivity**[_dim + 1];
  for (uint d0 = 0; d0 <= _dim; d0++)
  {
    connectivity[d0] = new MeshConnectivity*[_dim + 1];
    for (uint d1 = 0; d1 <= _dim; d1++)
      connectivity[d0][d1] = new MeshConnectivity(d0, d1);
  }

  // Copy data
  if (_dim > 0)
  {
    for (uint d = 0; d <= _dim; d++)
      num_entities[d] = topology.num_entities[d];
    for (uint d0 = 0; d0 <= _dim; d0++)
      for (uint d1 = 0; d1 <= _dim; d1++)
        *connectivity[d0][d1] = *topology.connectivity[d0][d1];
  }

  return *this;
}
//-----------------------------------------------------------------------------
void MeshTopology::clear()
{
  // Delete number of mesh entities
  delete [] num_entities;
  num_entities = 0;

  // Delete mesh connectivity
  if (connectivity)
  {
    for (uint d0 = 0; d0 <= _dim; d0++)
    {
      for (uint d1 = 0; d1 <= _dim; d1++)
        delete connectivity[d0][d1];
      delete [] connectivity[d0];
    }
    delete [] connectivity;
  }
  connectivity = 0;

  // Reset dimension
  _dim = 0;
}
//-----------------------------------------------------------------------------
void MeshTopology::clear(uint d0, uint d1)
{
  assert(d0 <= _dim);
  assert(d1 <= _dim);
  assert(connectivity[d0][d1]);
  connectivity[d0][d1]->clear();
}
//-----------------------------------------------------------------------------
void MeshTopology::init(uint dim)
{
  // Clear old data if any
  clear();

  // Initialize number of mesh entities
  num_entities = new uint[dim + 1];
  for (uint d = 0; d <= dim; d++)
    num_entities[d] = 0;

  // Initialize mesh connectivity
  connectivity = new MeshConnectivity**[dim + 1];
  for (uint d0 = 0; d0 <= dim; d0++)
  {
    connectivity[d0] = new MeshConnectivity*[dim + 1];
    for (uint d1 = 0; d1 <= dim; d1++)
      connectivity[d0][d1] = new MeshConnectivity(d0, d1);
  }

  // Save dimension
  _dim = dim;
}
//-----------------------------------------------------------------------------
void MeshTopology::init(uint dim, uint size)
{
  assert(num_entities);
  assert(dim <= _dim);

  num_entities[dim] = size;
}
//-----------------------------------------------------------------------------
std::string MeshTopology::str(bool verbose) const
{
  std::stringstream s;

  if (verbose)
  {
    s << str(false) << std::endl << std::endl;

    s << "  Number of entities:" << std::endl << std::endl;
    for (uint d = 0; d <= _dim; d++)
      s << "    dim = " << d << ": " << num_entities[d] << std::endl;
    s << std::endl;

    s << "  Connectivity matrix:" << std::endl << std::endl;
    s << "     ";
    for (uint d1 = 0; d1 <= _dim; d1++)
      s << " " << d1;
    s << std::endl;
    for (uint d0 = 0; d0 <= _dim; d0++)
    {
      s << "    " << d0;
      for (uint d1 = 0; d1 <= _dim; d1++)
      {
        if ( connectivity[d0][d1]->size() > 0 )
          s << " x";
        else
          s << " -";
      }
      s << std::endl;
    }
    s << std::endl;

    for (uint d0 = 0; d0 <= _dim; d0++)
    {
      for (uint d1 = 0; d1 <= _dim; d1++)
      {
        if ( connectivity[d0][d1]->size() == 0 )
          continue;
        s << indent(connectivity[d0][d1]->str(true));
        s << std::endl;
      }
    }
  }
  else
  {
    s << "<MeshTopology of dimension " << _dim << ">";
  }

  return s.str();
}
//-----------------------------------------------------------------------------
