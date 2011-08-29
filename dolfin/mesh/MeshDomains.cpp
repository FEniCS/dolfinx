// Copyright (C) 2011 Anders Logg
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
// First added:  2011-08-29
// Last changed: 2011-08-29

#include <dolfin/log/log.h>
#include "MeshDomains.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
MeshDomains::MeshDomains()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
MeshDomains::~MeshDomains()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
dolfin::uint MeshDomains::num_marked(uint dim) const
{
  assert(dim < _markers.size());
  return _markers[dim].size();
}
//-----------------------------------------------------------------------------
void MeshDomains::clear()
{
  _markers.clear();
}
//-----------------------------------------------------------------------------
void MeshDomains::init_subdomains(uint d)
{
  assert(_mesh);
  assert(d < _markers.size());
  assert(d < _subdomains.size());

  // Initialize entities of dimension d
  _mesh->init(d);

  // Initialize mesh function
  _subdomains[d].init(d);
  MeshFunction<uint>& mf = _subdomains[d];

  // Get mesh connectivity D --> d
  const uint D = _mesh->topology().dim();
  const MeshConnectivity& connectivity = _mesh->topology()(D, d);

  // Iterate over all markers
  for (uint i = 0; i < _markers[d].size(); i++)
  {
    // Get marker data
    const std::vector<uint>& marker = _markers[d][i];
    const uint cell_index   = marker[0];
    const uint local_entity = marker[1];
    const uint subdomain    = marker[2];

    // Get global facet index
    const uint global_entity = connectivity(cell_index)[local_entity];

    // Set boundary indicator for facet
    mf[global_entity] = subdomain;
  }
}
//-----------------------------------------------------------------------------
