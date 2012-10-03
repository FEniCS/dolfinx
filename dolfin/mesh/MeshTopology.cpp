// Copyright (C) 2006-2011 Anders Logg
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
// First added:  2006-05-08
// Last changed: 2011-11-15

#include <sstream>
#include <dolfin/log/log.h>
#include <dolfin/common/utils.h>
#include "MeshConnectivity.h"
#include "MeshTopology.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
MeshTopology::MeshTopology()

{
  // Do nothing
}
//-----------------------------------------------------------------------------
MeshTopology::MeshTopology(const MeshTopology& topology)
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

  // Copy data
  num_entities = topology.num_entities;
  connectivity = topology.connectivity;

  return *this;
}
//-----------------------------------------------------------------------------
dolfin::uint MeshTopology::dim() const
{
  return num_entities.size() - 1;
}
//-----------------------------------------------------------------------------
dolfin::uint MeshTopology::size(uint dim) const
{
  if (num_entities.size() == 0)
    return 0;

  dolfin_assert(dim < num_entities.size());
  return num_entities[dim];
}
//-----------------------------------------------------------------------------
void MeshTopology::clear()
{
  // Clear data
  num_entities.clear();
  connectivity.clear();
}
//-----------------------------------------------------------------------------
void MeshTopology::clear(uint d0, uint d1)
{
  dolfin_assert(d0 < connectivity.size());
  dolfin_assert(d1 < connectivity[d0].size());
  connectivity[d0][d1].clear();
}
//-----------------------------------------------------------------------------
void MeshTopology::init(uint dim)
{
  // Clear old data if any
  clear();

  // Initialize number of mesh entities
  num_entities = std::vector<uint>(dim + 1, 0);

  // Initialize mesh connectivity
  connectivity.resize(dim + 1);
  for (uint d0 = 0; d0 <= dim; d0++)
    for (uint d1 = 0; d1 <= dim; d1++)
      connectivity[d0].push_back(MeshConnectivity(d0, d1));
}
//-----------------------------------------------------------------------------
void MeshTopology::init(uint dim, uint size)
{
  dolfin_assert(dim < num_entities.size());
  num_entities[dim] = size;
}
//-----------------------------------------------------------------------------
dolfin::MeshConnectivity& MeshTopology::operator() (uint d0, uint d1)
{
  dolfin_assert(d0 < connectivity.size());
  dolfin_assert(d1 < connectivity[d0].size());
  return connectivity[d0][d1];
}
//-----------------------------------------------------------------------------
const dolfin::MeshConnectivity& MeshTopology::operator() (uint d0, uint d1) const
{
  dolfin_assert(d0 < connectivity.size());
  dolfin_assert(d1 < connectivity[d0].size());
  return connectivity[d0][d1];
}
//-----------------------------------------------------------------------------
std::string MeshTopology::str(bool verbose) const
{
  const uint _dim = num_entities.size() - 1;
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
        if ( !connectivity[d0][d1].empty() )
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
        if ( connectivity[d0][d1].empty() )
          continue;
        s << indent(connectivity[d0][d1].str(true));
        s << std::endl;
      }
    }
  }
  else
    s << "<MeshTopology of dimension " << _dim << ">";

  return s.str();
}
//-----------------------------------------------------------------------------
