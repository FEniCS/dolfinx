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
// Last changed: 2011-11-15

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
  if (d == 0)
  {
    dolfin_error("ParallelData.cpp",
                 "checking for global entity indices ofdim 0",
                 "ParallelData no longer stores local-to-global maps for vertices. Global vertex indices are stored inMeshGeometry::local_to_global_indices");
  }

  if (_global_entity_indices.find(d) != _global_entity_indices.end())
    return true;
  else
    return false;
}
//-----------------------------------------------------------------------------
MeshFunction<dolfin::uint>& ParallelData::global_entity_indices(uint d)
{
  if (d == 0)
  {
    dolfin_error("ParallelData.cpp",
                 "get global entity indices for dim 0",
                 "Do not use ParallelData::global_entity_indices for vertices. Use Vertex::global_index()");
  }

  if (!have_global_entity_indices(d))
    _global_entity_indices[d] = MeshFunction<uint>(mesh, d);
  return _global_entity_indices.find(d)->second;
}
//-----------------------------------------------------------------------------
const MeshFunction<dolfin::uint>& ParallelData::global_entity_indices(uint d) const
{
  if (d == 0)
  {
    dolfin_error("ParallelData.cpp",
                 "get global entity indices for dim 0",
                 "Do not use ParallelData::global_entity_indices for vertices. Use Vertex::global_index() or MeshGeometry::local_to_global_indices");
  }

  dolfin_assert(have_global_entity_indices(d));
  return _global_entity_indices.find(d)->second;
}
//-----------------------------------------------------------------------------
std::vector<dolfin::uint> ParallelData::global_entity_indices_as_vector(uint d) const
{
  const MeshFunction<uint>& x = global_entity_indices(d);
  return std::vector<uint>(x.values(), x.values() + x.size());
}
//-----------------------------------------------------------------------------
const std::map<dolfin::uint, dolfin::uint>& ParallelData::global_to_local_entity_indices(uint d)
{
  if (d == 0)
  {
    dolfin_error("ParallelData.cpp",
                 "get global entity indices for dim 0",
                 "Do not use ParallelData::global_to_local_entity_indices for vertices. Use MeshGeometry instead");
  }

  std::map<uint, std::map<uint, uint> >::iterator it;
  it = _global_to_local_entity_indices.find(d);
  if (it == _global_to_local_entity_indices.end())
  {
    // Build data for map
    const MeshFunction<uint>& local_global = global_entity_indices(d);
    std::vector<std::pair<uint, uint> > data;
    for (uint i = 0; i < local_global.size(); ++i)
      data.push_back(std::make_pair(local_global[i], i));

    // Insert a map
    std::map<uint, uint> tmp;
    std::pair<std::map<uint, std::map<uint, uint> >::iterator, bool> ret;
    ret = _global_to_local_entity_indices.insert(std::make_pair(d, tmp));
    dolfin_assert(ret.second);
    ret.first->second.insert(data.begin(), data.end());
    it = ret.first;
    dolfin_assert(it->second.size() == local_global.size());
  }
  return it->second;
}
//-----------------------------------------------------------------------------
const std::map<dolfin::uint, dolfin::uint>& ParallelData::global_to_local_entity_indices(uint d) const
{
  if (d == 0)
  {
    dolfin_error("ParallelData.cpp",
                 "get global entity indices for dim 0",
                 "Do not use ParallelData::global_entity_indices for vertices. Use Vertex::global_index() or MeshGeometry::local_to_global_indices");
  }

  std::map<uint, std::map<uint, uint> >::const_iterator it;
  it = _global_to_local_entity_indices.find(d);
  if (it == _global_to_local_entity_indices.end())
  {
    dolfin_error("ParallelData.cpp",
                 "extract global-to-local entity indices",
                 "Global-to-local map has not been computed");
  }

  return it->second;
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
  dolfin_assert(_exterior_facet);
  return *_exterior_facet;
}
//-----------------------------------------------------------------------------
const MeshFunction<bool>& ParallelData::exterior_facet() const
{
  dolfin_assert(_exterior_facet);
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
  dolfin_assert(it != _entity_colors.end());

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
  dolfin_error("ParallelData.cpp",
               "access entity colors",
               "Missing colors for entities of dimension %d colored by entities of dimension %d and distance %d",
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
  dolfin_assert(it != _colored_entities.end());

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
  dolfin_error("ParallelData.cpp",
               "access colored entities",
               "Missing colors for entities of dimension %d colored by entities of dimension %d and distance %d",
               D, d, rho);
  return it->second;
}
//-----------------------------------------------------------------------------
*/
