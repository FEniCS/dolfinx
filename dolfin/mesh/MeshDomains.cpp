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
// Last changed: 2011-09-13

#include <dolfin/log/log.h>
#include "MeshFunction.h"
#include "MeshValueCollection.h"
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
dolfin::uint MeshDomains::dim() const
{
  if (_markers.size() == 0)
    return 0;
  else
    return _markers.size() - 1;
}
//-----------------------------------------------------------------------------
dolfin::uint MeshDomains::num_marked(uint dim) const
{
  assert(dim < _markers.size());
  assert(_markers[dim]);
  return _markers[dim]->size();
}
//-----------------------------------------------------------------------------
MeshValueCollection<unsigned int>& MeshDomains::markers(uint dim)
{
  assert(dim < _markers.size());
  assert(_markers[dim]);
  return *_markers[dim];
}
//-----------------------------------------------------------------------------
const MeshValueCollection<unsigned int>& MeshDomains::markers(uint dim) const
{
  assert(dim < _markers.size());
  assert(_markers[dim]);
  return *_markers[dim];
}
//-----------------------------------------------------------------------------
void MeshDomains::init(const Mesh& mesh, uint dim)
{
  init(reference_to_no_delete_pointer(mesh), dim);
}
//-----------------------------------------------------------------------------
void MeshDomains::init(boost::shared_ptr<const Mesh> mesh, uint dim)
{
  // Clear old data
  clear();

  // Add markers for each topological dimension. Notice that to save
  // space we don't initialize the MeshFunctions here, only the
  // MeshValueCollection which require minimal storage when empty.
  for (uint d = 0; d <= dim; d++)
  {
    boost::shared_ptr<MeshValueCollection<uint> > m(new MeshValueCollection<uint>(mesh, d));
    boost::shared_ptr<MeshFunction<uint> > f(new MeshFunction<uint>());

    _markers.push_back(m);
    _subdomains.push_back(f);
  }
}
//-----------------------------------------------------------------------------
void MeshDomains::clear()
{
  _markers.clear();
  _subdomains.clear();
}
//-----------------------------------------------------------------------------
void init_subdomains()
{
  // FIXME: initialize and call function in MeshValueCollection

}
//-----------------------------------------------------------------------------
