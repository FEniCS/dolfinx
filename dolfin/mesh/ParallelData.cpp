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

#include "ParallelData.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
ParallelData::ParallelData(const Mesh& mesh) : mesh(mesh),
    _exterior_facet(mesh)
{
  // Do nothing
}
/*
//-----------------------------------------------------------------------------
ParallelData::~ParallelData()
{
  // Do nothing
}
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
