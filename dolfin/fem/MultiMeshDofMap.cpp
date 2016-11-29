// Copyright (C) 2013-2016 Anders Logg
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
// Last changed: 2016-03-02

#include <dolfin/common/types.h>
#include <dolfin/common/NoDeleter.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/function/MultiMeshFunctionSpace.h>
#include "DofMap.h"
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
  _index_map = dofmap._index_map;
  _original_dofmaps = dofmap._original_dofmaps;
  _new_dofmaps = dofmap._new_dofmaps;
}
//-----------------------------------------------------------------------------
MultiMeshDofMap::~MultiMeshDofMap()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
std::size_t MultiMeshDofMap::num_parts() const
{
  return _original_dofmaps.size();
}
//-----------------------------------------------------------------------------
std::shared_ptr<const GenericDofMap> MultiMeshDofMap::part(std::size_t i) const
{
  dolfin_assert(i < _new_dofmaps.size());
  return _new_dofmaps[i];
}
//-----------------------------------------------------------------------------
void MultiMeshDofMap::add(std::shared_ptr<const GenericDofMap> dofmap)
{
  _original_dofmaps.push_back(dofmap);
  log(PROGRESS, "Added dofmap to MultiMesh dofmap; dofmap has %d part(s).",
      _original_dofmaps.size());
}
//-----------------------------------------------------------------------------
void MultiMeshDofMap::build(const MultiMeshFunctionSpace& function_space,
                            const std::vector<dolfin::la_index>& offsets)
{
  // Compute global dimension
  begin(PROGRESS, "Computing total dimension.");
  std::size_t _global_dimension = 0;
  for (std::size_t i = 0; i < num_parts(); i++)
  {
    const std::size_t d = _original_dofmaps[i]->global_dimension();
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
  dolfin::la_index offset = 0;
  for (std::size_t part = 0; part < num_parts(); part++)
  {
    log(PROGRESS, "Computing dofs for part %d.", part);

    // Create new dofmap for part (copy of original dofmap)
    std::shared_ptr<GenericDofMap> new_dofmap = _original_dofmaps[part]->copy();
    _new_dofmaps.push_back(new_dofmap);

    // Choose offset, either using computed offset or given offset
    dolfin::la_index _offset = 0;
    if (offsets.size() > 0)
      _offset = offsets[part];
    else
      _offset = offset;

    // Add offset
    DofMap& dofmap = static_cast<DofMap&>(*new_dofmap);
    dofmap._multimesh_offset = _offset;
    for (auto it = dofmap._dofmap.begin(); it != dofmap._dofmap.end(); ++it)
      *it += _offset;

    // Increase offset
    offset += _original_dofmaps[part]->global_dimension();
  }

  _index_map.reset(new IndexMap(MPI_COMM_WORLD, _global_dimension, 1));
}
//-----------------------------------------------------------------------------
void MultiMeshDofMap::clear()
{
  _index_map.reset();
  _original_dofmaps.clear();
  _new_dofmaps.clear();
}
//-----------------------------------------------------------------------------
std::size_t MultiMeshDofMap::global_dimension() const
{
  return _index_map->size(IndexMap::MapSize::GLOBAL);
}
//-----------------------------------------------------------------------------
std::pair<std::size_t, std::size_t> MultiMeshDofMap::ownership_range() const
{
  // FIXME: Does not run in parallel
  return _index_map->local_range();
}
//-----------------------------------------------------------------------------
const std::vector<int>&
MultiMeshDofMap::off_process_owner() const
{
  // FIXME: Does not run in parallel
  return _index_map->off_process_owner();
}
//-----------------------------------------------------------------------------
std::shared_ptr<IndexMap> MultiMeshDofMap::index_map() const
{
  // FIXME: Does not run in parallel
  return _index_map;
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
std::vector<dolfin::la_index>
MultiMeshDofMap::inactive_dofs(MultiMesh multimesh, std::size_t part) const
{
  std::shared_ptr<const GenericDofMap> dofmap_part = this->part(part);

  // Get all dofs on covered cells
  std::vector<unsigned int> covered_cells = multimesh.covered_cells(part);

  std::vector<dolfin::la_index> covered_dofs;
  covered_dofs.reserve(dofmap_part->max_element_dofs() * covered_cells.size());
  for (unsigned int cell : covered_cells)
  {
    ArrayView<const dolfin::la_index> local_dofs = dofmap_part->cell_dofs(cell);
    std::copy(local_dofs.begin(), local_dofs.end(), std::back_inserter(covered_dofs));
  }
  // Sort and remove duplicates
  std::sort(covered_dofs.begin(), covered_dofs.end());
  covered_dofs.erase(std::unique(covered_dofs.begin(), covered_dofs.end()),
                      covered_dofs.end());

  // Get all dofs on cut cells
  std::vector<unsigned int> cut_cells = multimesh.cut_cells(part);

  std::vector<dolfin::la_index> cut_cell_dofs;
  cut_cell_dofs.reserve(dofmap_part->max_element_dofs() * cut_cells.size());
  for (unsigned int cell : cut_cells)
  {
    //ArrayView<const dolfin::la_index> local_dofs = dofmap_part->cell_dofs(cell);
    ArrayView<const dolfin::la_index> local_dofs = dofmap_part->cell_dofs(cell);
    std::copy(local_dofs.begin(), local_dofs.end(), std::back_inserter(cut_cell_dofs));
  }
  // Sort and remove duplicates
  std::sort(cut_cell_dofs.begin(), cut_cell_dofs.end());
  cut_cell_dofs.erase(std::unique(cut_cell_dofs.begin(), cut_cell_dofs.end()),
                      cut_cell_dofs.end());

  // Remove cut cell dofs from covered dofs
  std::vector<dolfin::la_index> _inactive_dofs;
  std::set_difference(covered_dofs.begin(), covered_dofs.end(),
                      cut_cell_dofs.begin(), cut_cell_dofs.end(),
                      std::inserter(_inactive_dofs, _inactive_dofs.begin()));
  return _inactive_dofs;
}
