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
// Last changed: 2014-07-02

#include <numeric>
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
  : coloring(topology.coloring), num_entities(topology.num_entities),
    ghost_offset_index(topology.ghost_offset_index),
    global_num_entities(topology.global_num_entities),
    _global_indices(topology._global_indices),
    _shared_entities(topology._shared_entities),
    connectivity(topology.connectivity)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
MeshTopology::~MeshTopology()
{
  clear();
}
//-----------------------------------------------------------------------------
MeshTopology& MeshTopology::operator= (const MeshTopology& topology)
{
  // Public data
  coloring = topology.coloring;

  // Private data
  num_entities = topology.num_entities;
  global_num_entities = topology.global_num_entities;
  ghost_offset_index = topology.ghost_offset_index;
  _global_indices = topology._global_indices;
  _shared_entities = topology._shared_entities;
  connectivity = topology.connectivity;

  return *this;
}
//-----------------------------------------------------------------------------
std::size_t MeshTopology::dim() const
{
  return num_entities.size() - 1;
}
//-----------------------------------------------------------------------------
std::size_t MeshTopology::size(std::size_t dim) const
{
  if (num_entities.size() == 0)
    return 0;

  dolfin_assert(dim < num_entities.size());
  return num_entities[dim];
}
//-----------------------------------------------------------------------------
std::size_t MeshTopology::size_global(std::size_t dim) const
{
  if (global_num_entities.empty())
    return 0;

  dolfin_assert(dim < global_num_entities.size());
  return global_num_entities[dim];
}
//-----------------------------------------------------------------------------
std::size_t MeshTopology::ghost_offset(std::size_t dim) const
{
  if (ghost_offset_index.empty())
    return 0;

  dolfin_assert(dim < ghost_offset_index.size());
  return ghost_offset_index[dim];
}
//-----------------------------------------------------------------------------
void MeshTopology::clear()
{
  // Clear data
  coloring.clear();
  num_entities.clear();
  global_num_entities.clear();
  ghost_offset_index.clear();
  _global_indices.clear();
  _shared_entities.clear();
  connectivity.clear();
}
//-----------------------------------------------------------------------------
void MeshTopology::clear(std::size_t d0, std::size_t d1)
{
  dolfin_assert(d0 < connectivity.size());
  dolfin_assert(d1 < connectivity[d0].size());
  connectivity[d0][d1].clear();
}
//-----------------------------------------------------------------------------
void MeshTopology::init(std::size_t dim)
{
  // Clear old data if any
  clear();

  // Initialize number of mesh entities
  num_entities = std::vector<unsigned int>(dim + 1, 0);
  global_num_entities = std::vector<std::size_t>(dim + 1, 0);
  ghost_offset_index = std::vector<std::size_t>(dim + 1, 0);

  // Initialize storage for global indices
  _global_indices.resize(dim + 1);

  // Initialize mesh connectivity
  connectivity.resize(dim + 1);
  for (std::size_t d0 = 0; d0 <= dim; d0++)
    for (std::size_t d1 = 0; d1 <= dim; d1++)
      connectivity[d0].push_back(MeshConnectivity(d0, d1));
}
//-----------------------------------------------------------------------------
void MeshTopology::init(std::size_t dim, std::size_t local_size,
                        std::size_t global_size)
{
  dolfin_assert(dim < num_entities.size());
  num_entities[dim] = local_size;

  dolfin_assert(dim < global_num_entities.size());
  global_num_entities[dim] = global_size;

  // FIXME: Remove this when ghost/halo cells are supported
  // If mesh is local, make shared vertices empty
  if (dim == 0 && (local_size == global_size))
    shared_entities(0);
}
//-----------------------------------------------------------------------------
void MeshTopology::init_ghost(std::size_t dim, std::size_t index)
{
  dolfin_assert(dim < ghost_offset_index.size());
  ghost_offset_index[dim] = index;
}
//-----------------------------------------------------------------------------
void MeshTopology::init_global_indices(std::size_t dim, std::size_t size)
{
  dolfin_assert(dim < _global_indices.size());
  _global_indices[dim]
    = std::vector<std::size_t>(size, std::numeric_limits<std::size_t>::max());
}
//-----------------------------------------------------------------------------
dolfin::MeshConnectivity& MeshTopology::operator() (std::size_t d0,
                                                    std::size_t d1)
{
  dolfin_assert(d0 < connectivity.size());
  dolfin_assert(d1 < connectivity[d0].size());
  return connectivity[d0][d1];
}
//-----------------------------------------------------------------------------
const dolfin::MeshConnectivity& MeshTopology::operator() (std::size_t d0,
                                                          std::size_t d1) const
{
  dolfin_assert(d0 < connectivity.size());
  dolfin_assert(d1 < connectivity[d0].size());
  return connectivity[d0][d1];
}
//-----------------------------------------------------------------------------
std::map<std::int32_t, std::set<unsigned int>>&
  MeshTopology::shared_entities(unsigned int dim)
{
  dolfin_assert(dim <= this->dim());
  return _shared_entities[dim];
}
//-----------------------------------------------------------------------------
const std::map<std::int32_t, std::set<unsigned int>>&
  MeshTopology::shared_entities(unsigned int dim) const
{
  auto e = _shared_entities.find(dim);
  if (e == _shared_entities.end())
  {
    dolfin_error("MeshTopology.cpp",
                 "get shared mesh entities",
                 "Shared mesh entities have not been computed for dim %d", dim);
  }
  return e->second;
}
//-----------------------------------------------------------------------------
size_t MeshTopology::hash() const
{
  return (*this)(dim(), 0).hash();
}
//-----------------------------------------------------------------------------
std::string MeshTopology::str(bool verbose) const
{
  const std::size_t _dim = num_entities.size() - 1;
  std::stringstream s;
  if (verbose)
  {
    s << str(false) << std::endl << std::endl;

    s << "  Number of entities:" << std::endl << std::endl;
    for (std::size_t d = 0; d <= _dim; d++)
      s << "    dim = " << d << ": " << num_entities[d] << std::endl;
    s << std::endl;

    s << "  Connectivity matrix:" << std::endl << std::endl;
    s << "     ";
    for (std::size_t d1 = 0; d1 <= _dim; d1++)
      s << " " << d1;
    s << std::endl;
    for (std::size_t d0 = 0; d0 <= _dim; d0++)
    {
      s << "    " << d0;
      for (std::size_t d1 = 0; d1 <= _dim; d1++)
      {
        if (!connectivity[d0][d1].empty())
          s << " x";
        else
          s << " -";
      }
      s << std::endl;
    }
    s << std::endl;

    for (std::size_t d0 = 0; d0 <= _dim; d0++)
    {
      for (std::size_t d1 = 0; d1 <= _dim; d1++)
      {
        if (connectivity[d0][d1].empty())
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
