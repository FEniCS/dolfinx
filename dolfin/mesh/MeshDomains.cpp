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
// Modified by Garth N. Wells, 2012
//
// First added:  2011-08-29
// Last changed: 2011-04-03

#include <limits>
#include <dolfin/common/MPI.h>
#include <dolfin/log/log.h>
#include "MeshFunction.h"
#include "MeshValueCollection.h"
#include "MeshDomains.h"

using namespace dolfin;

const std::size_t MeshDomains::default_unset_value = std::numeric_limits<std::size_t>::max();

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
std::size_t MeshDomains::max_dim() const
{
  if (!_markers.empty())
    return _markers.size() - 1;
  else
    return 0;
}
//-----------------------------------------------------------------------------
std::size_t MeshDomains::num_marked(std::size_t dim) const
{
  dolfin_assert(dim < _markers.size());
  dolfin_assert(_markers[dim]);
  return _markers[dim]->size();
}
//-----------------------------------------------------------------------------
bool MeshDomains::is_empty() const
{
  std::size_t size = 0;
  for (std::size_t i = 0; i < _markers.size(); i++)
  {
    dolfin_assert(_markers[i]);
    size += _markers[i]->size();
  }
  return size == 0;
}
//-----------------------------------------------------------------------------
boost::shared_ptr<MeshValueCollection<std::size_t> >
  MeshDomains::markers(std::size_t dim)
{
  dolfin_assert(dim < _markers.size());
  return _markers[dim];
}
//-----------------------------------------------------------------------------
boost::shared_ptr<const MeshValueCollection<std::size_t> >
  MeshDomains::markers(std::size_t dim) const
{
  dolfin_assert(dim < _markers.size());
  return _markers[dim];
}
//-----------------------------------------------------------------------------
std::vector<std::string> MeshDomains::marker_names(std::size_t dim) const
{
  dolfin_assert(dim < _named_markers.size());
  std::vector<std::string> names;
  boost::unordered_map<std::string, boost::shared_ptr<MeshValueCollection<std::size_t> > >::const_iterator m;
  for (m = _named_markers[dim].begin(); m != _named_markers[dim].end(); ++m)
    names.push_back(m->first);
  return names;
}
//-----------------------------------------------------------------------------
boost::shared_ptr<const MeshFunction<std::size_t> >
  MeshDomains::cell_domains(const Mesh& mesh, std::size_t unset_value) const
{
  // Check if any markers have been set
  const std::size_t D = mesh.topology().dim();
  dolfin_assert(D < _markers.size());

  // Create markers if mesh collection present
  if (!_markers[D]->empty() and !_cell_domains)
  {
    MeshValueCollection<std::size_t> domain = *(_markers[D]);
    _cell_domains = mesh_function(mesh, domain, unset_value);
  }

  return _cell_domains;
}
//-----------------------------------------------------------------------------
boost::shared_ptr<const MeshFunction<std::size_t> >
  MeshDomains::facet_domains(const Mesh& mesh, std::size_t unset_value) const
{
  // Check if any markers have been set
  const std::size_t D = mesh.topology().dim();
  dolfin_assert((D - 1) < _markers.size());

  // Create markers if mesh collection present
  if (!_markers[D - 1]->empty() and !_facet_domains)
  {
    const MeshValueCollection<std::size_t> domain = *(_markers[D - 1]);
    _facet_domains = mesh_function(mesh, domain, unset_value);
  }

  return _facet_domains;
}
//-----------------------------------------------------------------------------
boost::shared_ptr<MeshFunction<std::size_t> >
MeshDomains::mesh_function(const Mesh& mesh,
                           const MeshValueCollection<std::size_t>& collection,
                           std::size_t unset_value)
{
  // Get dimensions
  const std::size_t d = collection.dim();
  const std::size_t D = mesh.topology().dim();

  // Create MeshFunction
  boost::shared_ptr<MeshFunction<std::size_t> >
    mesh_function(new MeshFunction<std::size_t>(mesh, d, unset_value));

  // Get mesh connectivity D --> d
  dolfin_assert(d <= D);
  const MeshConnectivity& connectivity = mesh.topology()(D, d);
  dolfin_assert(D == d || !connectivity.empty());

  // Iterate over all values
  const std::map<std::pair<std::size_t, std::size_t>, std::size_t>& values = collection.values();
  std::map<std::pair<std::size_t, std::size_t>, std::size_t>::const_iterator it;
  for (it = values.begin(); it != values.end(); ++it)
  {
    // Get marker data
    const std::size_t cell_index = it->first.first;
    const std::size_t local_entity = it->first.second;
    const std::size_t value = it->second;

    // Get global entity index. Note that we ignore the local entity
    // index when the function is defined over cells.
    std::size_t entity_index = 0;
    if (d == D)
      entity_index = cell_index;
    else
      entity_index = connectivity(cell_index)[local_entity];

    // Check that value is not equal to the 'unset' value
    if (value == unset_value)
      warning("MeshValueCollection value entry is equal to %d, which is used to indicate an \"unset\" value.", value);

    // Set value for entity
    (*mesh_function)[entity_index] = value;
  }

  return mesh_function;
}
//-----------------------------------------------------------------------------
void MeshDomains::init(std::size_t dim)
{
  // Clear old data
  clear();

  // Add markers for each topological dimension
  for (std::size_t d = 0; d <= dim; d++)
  {
    boost::shared_ptr<MeshValueCollection<std::size_t> >
        m(new MeshValueCollection<std::size_t>(d));
    _markers.push_back(m);
  }
}
//-----------------------------------------------------------------------------
void MeshDomains::clear()
{
  _markers.clear();
  _named_markers.clear();
  _cell_domains.reset();
  _facet_domains.reset();
}
//-----------------------------------------------------------------------------
