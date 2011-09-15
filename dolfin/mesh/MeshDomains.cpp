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
// Last changed: 2011-09-15

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
  if (_mesh)
    return _mesh->topology().dim();
  else
    return 0;
}
//-----------------------------------------------------------------------------
dolfin::uint MeshDomains::num_marked(uint dim) const
{
  assert(dim < _markers.size());
  assert(_markers[dim]);
  return _markers[dim]->size();
}
//-----------------------------------------------------------------------------
bool MeshDomains::is_empty() const
{
  uint size = 0;
  for (uint i = 0; i < _markers.size(); i++)
  {
    assert(_markers[i]);
    size += _markers[i]->size();
  }
  return size == 0;
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
boost::shared_ptr<MeshValueCollection<unsigned int> >
MeshDomains::markers_shared_ptr(uint dim)
{
  assert(dim < _markers.size());
  assert(_markers[dim]);
  return _markers[dim];
}
//-----------------------------------------------------------------------------
boost::shared_ptr<const MeshValueCollection<unsigned int> >
MeshDomains::markers_shared_ptr(uint dim) const
{
  assert(dim < _markers.size());
  assert(_markers[dim]);
  return _markers[dim];
}
//-----------------------------------------------------------------------------
boost::shared_ptr<const MeshFunction<dolfin::uint> >
MeshDomains::cell_domains() const
{
  // Check if data already exists
  if (_cell_domains)
    return _cell_domains;

  // Compute cell domains
  assert(_mesh);
  _cell_domains = boost::shared_ptr<MeshFunction<uint> >(new MeshFunction<uint>());
  _cell_domains->init(*_mesh, _mesh->topology().dim());
  init_domains(*_cell_domains);

  return _cell_domains;
}
//-----------------------------------------------------------------------------
boost::shared_ptr<const MeshFunction<dolfin::uint> >
MeshDomains::facet_domains() const
{
  // Check if data already exists
  if (_facet_domains)
    return _facet_domains;

  // Compute facet domains
  assert(_mesh);
  _facet_domains = boost::shared_ptr<MeshFunction<uint> >(new MeshFunction<uint>());
  _cell_domains->init(*_mesh, _mesh->topology().dim() - 1);
  init_domains(*_facet_domains);

  return _facet_domains;
}
//-----------------------------------------------------------------------------
void MeshDomains::init(const Mesh& mesh)
{
  init(reference_to_no_delete_pointer(mesh));
}
//-----------------------------------------------------------------------------
void MeshDomains::init(boost::shared_ptr<const Mesh> mesh)
{
  // Clear old data
  clear();

  // Store mesh
  _mesh = mesh;

  // Add markers for each topological dimension. Notice that to save
  // space we don't initialize the MeshFunctions here, only the
  // MeshValueCollection which require minimal storage when empty.
  for (uint d = 0; d <= mesh->topology().dim(); d++)
  {
    boost::shared_ptr<MeshValueCollection<uint> >
      m(new MeshValueCollection<uint>(mesh, d));
    _markers.push_back(m);
  }
}
//-----------------------------------------------------------------------------
void MeshDomains::clear()
{
  _markers.clear();
}
//-----------------------------------------------------------------------------
void MeshDomains::init_domains(MeshFunction<uint>& mesh_function) const
{
  assert(_mesh);

  // Get mesh connectivity D --> d
  const uint d = mesh_function.dim();
  const uint D = _mesh->topology().dim();
  assert(d <= D);
  const MeshConnectivity& connectivity = _mesh->topology()(D, d);
  assert(connectivity.size() > 0);

  // Get maximum value
  uint maxval = 0;
  std::map<std::pair<uint, uint>, uint> values = _markers[d]->values();
  typename std::map<std::pair<uint, uint>, uint>::const_iterator it;
  for (it = values.begin(); it != values.end(); ++it)
    maxval = std::max(maxval, it->second);

  // Set all values of mesh function to maximum value + 1
  mesh_function.set_all(maxval + 1);

  // Iterate over all values
  for (it = values.begin(); it != values.end(); ++it)
  {
    // Get marker data
    const uint cell_index = it->first.first;
    const uint local_entity = it->first.second;
    const uint value = it->second;

    // Get global entity index
    const uint entity_index = connectivity(cell_index)[local_entity];

    // Set value for entity
    mesh_function[entity_index] = value;
  }
}
//---------------------------------------------------------------------------
