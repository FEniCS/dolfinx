// Copyright (C) 2013 Anders Logg
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
// First added:  2013-09-19
// Last changed: 2014-04-28

#include <dolfin/common/NoDeleter.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/function/MultiMeshFunctionSpace.h>
#include "MultiMeshDofMap.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
MultiMeshDofMap::MultiMeshDofMap()
{
  clear();
}
//-----------------------------------------------------------------------------
MultiMeshDofMap::MultiMeshDofMap(const MultiMeshDofMap& dofmap)
{
  _global_dimension = dofmap._global_dimension;
  _dofmaps = dofmap._dofmaps;
}
//-----------------------------------------------------------------------------
MultiMeshDofMap::~MultiMeshDofMap()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
std::size_t MultiMeshDofMap::num_parts() const
{
  return _dofmaps.size();
}
//-----------------------------------------------------------------------------
std::shared_ptr<const GenericDofMap> MultiMeshDofMap::part(std::size_t i) const
{
  dolfin_assert(i < _dofmaps.size());
  return _new_dofmaps[i];
}
//-----------------------------------------------------------------------------
void MultiMeshDofMap::add(std::shared_ptr<const GenericDofMap> dofmap)
{
  _dofmaps.push_back(dofmap);
  log(PROGRESS, "Added dofmap to MultiMesh dofmap; dofmap has %d part(s).",
      _dofmaps.size());
}
//-----------------------------------------------------------------------------
void MultiMeshDofMap::add(const GenericDofMap& dofmap)
{
  add(reference_to_no_delete_pointer(dofmap));
}
//-----------------------------------------------------------------------------
void MultiMeshDofMap::build(const MultiMeshFunctionSpace& function_space)
{
  // Compute global dimension
  begin(PROGRESS, "Computing total dimension.");
  _global_dimension = 0;
  for (std::size_t i = 0; i < num_parts(); i++)
  {
    const std::size_t d = _dofmaps[i]->global_dimension();
    _global_dimension += d;
    log(PROGRESS, "dim(V_%d) = %d", i, d);
  }
  end();
  log(PROGRESS, "Total global dimension is %d.", _global_dimension);

  // For now, we build the simplest possible dofmap by reusing the
  // dofmaps for each part and adding offsets in between.

  // Clear old dofmaps if any
  _new_dofmaps.clear();

  // Build dofmap
  _dofmap.clear();
  dolfin::la_index offset = 0;
  for (std::size_t part = 0; part < num_parts(); part++)
  {
    log(PROGRESS, "Computing dofs for part %d.", part);

    dolfin_assert(_dofmaps[part]);
    std::vector<std::vector<dolfin::la_index> > dofmap_part;

    // Get mesh on current part
    const Mesh& mesh = *function_space.part(part)->mesh();

    // Create new dofmap for part (copy of original dofmap)
    std::shared_ptr<GenericDofMap> new_dofmap = _dofmaps[part]->copy();
    _new_dofmaps.push_back(new_dofmap);

    // Add offset
    new_dofmap->add_offset(offset);

    // Modify all dofs in copy by adding an offset
    for (CellIterator cell(mesh); !cell.end(); ++cell)
    {
      // Get dofs from dofmap on part
      const std::vector<dolfin::la_index>& dofs
        = _dofmaps[part]->cell_dofs(cell->index());

      // Compute new dofs by adding offset
      std::vector<dolfin::la_index> new_dofs;
      for (std::size_t i = 0; i < dofs.size(); i++)
        new_dofs.push_back(dofs[i] + offset);

      // Store dofs for cell
      dofmap_part.push_back(new_dofs);
    }

    // Store dofs for part
    _dofmap.push_back(dofmap_part);

    // Increase offset
    offset += _dofmaps[part]->global_dimension();
  }
}
//-----------------------------------------------------------------------------
void MultiMeshDofMap::clear()
{
  _global_dimension = 0;
  _dofmaps.clear();
  _dofmap.clear();
}
//-----------------------------------------------------------------------------
std::size_t MultiMeshDofMap::global_dimension() const
{
  return _global_dimension;
}
//-----------------------------------------------------------------------------
std::pair<std::size_t, std::size_t> MultiMeshDofMap::ownership_range() const
{
  // FIXME: Does not run in parallel
  return std::make_pair<std::size_t, std::size_t>(0, global_dimension());
}
//-----------------------------------------------------------------------------
const boost::unordered_map<std::size_t, unsigned int>&
MultiMeshDofMap::off_process_owner() const
{
  // FIXME: Does not run in parallel
  return _dofmaps[0]->off_process_owner();
}
//-----------------------------------------------------------------------------
std::string MultiMeshDofMap::str(bool verbose) const
{
  std::stringstream s;
  s << "<MultiMeshDofMap with "
    << num_parts()
    << " parts and total global dimension "
    << global_dimension()
    << ">"
    << std::endl;
  return s.str();
}
//-----------------------------------------------------------------------------
