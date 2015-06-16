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
  return _markers[dim].size();
}
//-----------------------------------------------------------------------------
bool MeshDomains::is_empty() const
{
  std::size_t size = 0;
  for (std::size_t i = 0; i < _markers.size(); i++)
    size += _markers[i].size();
  return size == 0;
}
//-----------------------------------------------------------------------------
std::map<std::size_t, std::size_t>& MeshDomains::markers(std::size_t dim)
{
  dolfin_assert(dim < _markers.size());
  return _markers[dim];
}
//-----------------------------------------------------------------------------
const std::map<std::size_t, std::size_t>&
MeshDomains::markers(std::size_t dim) const
{
  dolfin_assert(dim < _markers.size());
  return _markers[dim];
}
//-----------------------------------------------------------------------------
bool MeshDomains::set_marker(std::pair<std::size_t, std::size_t> marker,
                             std::size_t dim)
{
  dolfin_assert(dim < _markers.size());

  // Check if key already present
  bool new_entity_index = false;
  auto it = _markers[dim].find(marker.first);
  if (it == _markers[dim].end())
  {
    new_entity_index = true;
  }

  _markers[dim][marker.first] = marker.second;
  return new_entity_index;
}
//-----------------------------------------------------------------------------
std::size_t MeshDomains::get_marker(std::size_t entity_index,
                                    std::size_t dim) const
{
  dolfin_assert(dim < _markers.size());
  std::map<std::size_t, std::size_t>::const_iterator it
    = _markers[dim].find(entity_index);
  if (it == _markers[dim].end())
  {
    dolfin_error("MeshDomains.cpp",
                 "get marker",
                 "Marked entity index does not exist in marked set");
  }

  return it->second;
}
//-----------------------------------------------------------------------------
const MeshDomains& MeshDomains::operator= (const MeshDomains& domains)
{
  // Copy data
  _markers = domains._markers;

  return *this;
}
//-----------------------------------------------------------------------------
void MeshDomains::init(std::size_t dim)
{
  // Clear old data
  clear();

  // Add markers for each topological dimension
  _markers.resize(dim + 1);
}
//-----------------------------------------------------------------------------
void MeshDomains::clear()
{
  _markers.clear();
}
//-----------------------------------------------------------------------------
