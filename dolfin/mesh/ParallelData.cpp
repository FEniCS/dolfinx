// Copyright (C) 2011 Anders Logg and Garth N. Wells
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
// First added:  2011-01-17
// Last changed: 2011-01-17

#include "MeshFunction.h"
#include "ParallelData.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
ParallelData::ParallelData(const Mesh& mesh) : mesh(mesh),
    _exterior_facet(new MeshFunction<bool>(mesh))
{
  // Do nothing
}
//-----------------------------------------------------------------------------
ParallelData::ParallelData(const ParallelData& data) : mesh(data.mesh),
  _global_entity_indices(data._global_entity_indices),
  _shared_vertices(data._shared_vertices),
  _num_global_entities(data._num_global_entities),
  _exterior_facet(new MeshFunction<bool>(*data._exterior_facet))
{
  // Do nothing
}
//-----------------------------------------------------------------------------
ParallelData::~ParallelData()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
bool ParallelData::have_global_entity_indices(uint d) const
{
  if (_global_entity_indices.find(d) != _global_entity_indices.end())
    return true;
  else
    return false;
}
//-----------------------------------------------------------------------------
MeshFunction<dolfin::uint>& ParallelData::global_entity_indices(uint d)
{
  if (!have_global_entity_indices(d))
    _global_entity_indices[d] = MeshFunction<uint>(mesh, d);
  return _global_entity_indices.find(d)->second;
}
//-----------------------------------------------------------------------------
const MeshFunction<dolfin::uint>& ParallelData::global_entity_indices(uint d) const
{
  assert(have_global_entity_indices(d));
  return _global_entity_indices.find(d)->second;
}
//-----------------------------------------------------------------------------
std::vector<dolfin::uint> ParallelData::global_entity_indices_as_vector(uint d) const
{
  const MeshFunction<uint>& x = global_entity_indices(d);
  return std::vector<uint>(x.values(), x.values() + x.size());
}
//-----------------------------------------------------------------------------
std::map<dolfin::uint, std::vector<dolfin::uint> >& ParallelData::shared_vertices()
{
  return _shared_vertices;
}
//-----------------------------------------------------------------------------
const std::map<dolfin::uint, std::vector<dolfin::uint> >& ParallelData::shared_vertices() const
{
  return _shared_vertices;
}
//-----------------------------------------------------------------------------
MeshFunction<bool>& ParallelData::exterior_facet()
{
  assert(_exterior_facet);
  return *_exterior_facet;
}
//-----------------------------------------------------------------------------
const MeshFunction<bool>& ParallelData::exterior_facet() const
{
  assert(_exterior_facet);
  return *_exterior_facet;
}
//-----------------------------------------------------------------------------
std::vector<dolfin::uint>& ParallelData::num_global_entities()
{
  return _num_global_entities;
}
//-----------------------------------------------------------------------------
const std::vector<dolfin::uint>& ParallelData::num_global_entities() const
{
  return _num_global_entities;
}
//-----------------------------------------------------------------------------


/*
//-----------------------------------------------------------------------------
dolfin::uint ParallelData::num_colors(uint D, uint d, uint rho) const
{
  // Try to find in map
  tuple_type tuple(D, d, rho);
  colored_entities_map_type::const_iterator it = _colored_entities.find(tuple);

  // Return 0 if found
  if (it != _colored_entities.end())
    return 0;

  // Return size if found
  return it->second.size();
}
//-----------------------------------------------------------------------------
MeshFunction<dolfin::uint>& ParallelData::entity_colors(uint D, uint d, uint rho)
{
  // Try to find in map
  tuple_type tuple(D, d, rho);
  entity_colors_map_type::iterator it = _entity_colors.find(tuple);

  // Return if found
  if (it != _entity_colors.end())
    return it->second;

  // Create if not found
  _entity_colors[tuple] = MeshFunction<uint>(_mesh);
  it = _entity_colors.find(tuple);
  assert(it != _entity_colors.end());

  return it->second;
}
//-----------------------------------------------------------------------------
const MeshFunction<dolfin::uint>&
ParallelData::entity_colors(uint D, uint d, uint rho) const
{
  // Try to find in map
  tuple_type tuple(D, d, rho);
  entity_colors_map_type::const_iterator it = _entity_colors.find(tuple);

  // Return if found
  if (it != _entity_colors.end())
    return it->second;

  // Not found
  error("Missing colors for entities of dimension %d colored by entities of dimension %d and distance %d.",
        D, d, rho);
  return it->second;
}
//-----------------------------------------------------------------------------
std::vector<std::vector<dolfin::uint> >&
ParallelData::colored_entities(uint D, uint d, uint rho)
{
  // Try to find in map
  tuple_type tuple(D, d, rho);
  colored_entities_map_type::iterator it = _colored_entities.find(tuple);

  // Return if found
  if (it != _colored_entities.end())
    return it->second;

  // Create if not found
  _colored_entities[tuple] = std::vector<std::vector<uint> >();
  it = _colored_entities.find(tuple);
  assert(it != _colored_entities.end());

  return it->second;
}
//-----------------------------------------------------------------------------
const std::vector<std::vector<dolfin::uint> >&
ParallelData::colored_entities(uint D, uint d, uint rho) const
{
  // Try to find in map
  tuple_type tuple(D, d, rho);
  colored_entities_map_type::const_iterator it = _colored_entities.find(tuple);

  // Return if found
  if (it != _colored_entities.end())
    return it->second;

  // Not found
  error("Missing colors for entities of dimension %d colored by entities of dimension %d and distance %d.",
        D, d, rho);
  return it->second;
}
//-----------------------------------------------------------------------------
*/
