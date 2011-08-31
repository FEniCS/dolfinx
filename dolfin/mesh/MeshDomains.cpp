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
// Last changed: 2011-08-31

#include <dolfin/log/log.h>
#include "Mesh.h"
#include "MeshFunction.h"
#include "MeshMarkers.h"
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
  assert(_markers[dim]);
  return _markers[dim]->size();
}
//-----------------------------------------------------------------------------
void MeshDomains::clear()
{
  _markers.clear();
}
//-----------------------------------------------------------------------------
void init_subdomains()
{
  // FIXME: initialize and call function in MeshMarkers

}
//-----------------------------------------------------------------------------
