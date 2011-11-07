// Copyright (C) 2007-2011 Anders Logg and Garth N. Wells
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
// Modified by Martin Alnes, 2008
// Modified by Kent-Andre Mardal, 2009
// Modified by Ola Skavhaug, 2009
// Modified by Niclas Jansson, 2009
//
// First added:  2007-03-01
// Last changed: 2011-10-31

#include <dolfin/common/MPI.h>
#include <dolfin/common/NoDeleter.h>
#include <dolfin/common/Set.h>
#include <dolfin/common/types.h>
#include <dolfin/log/LogStream.h>
#include <dolfin/mesh/BoundaryMesh.h>
#include <dolfin/mesh/MeshPartitioning.h>
#include "DofMapBuilder.h"
#include "UFCCell.h"
#include "UFCMesh.h"
#include "DofMap.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
DofMap::DofMap(boost::shared_ptr<const ufc::dofmap> ufc_dofmap,
               Mesh& dolfin_mesh) : _ufc_dofmap(ufc_dofmap->create()),
               ufc_offset(0), _is_view(false),
               _distributed(MPI::num_processes() > 1)
{
  assert(_ufc_dofmap);

  // Check for dimensional consistency between the dofmap and mesh
  check_dimensional_consistency(*_ufc_dofmap, dolfin_mesh);

  // Check that mesh has been ordered
  if (!dolfin_mesh.ordered())
     error("Mesh is not ordered according to the UFC numbering convention, consider calling mesh.order().");

  // Generate and number all mesh entities
  const uint D = dolfin_mesh.topology().dim();
  for (uint d = 1; d <= D; ++d)
  {
    if (_ufc_dofmap->needs_mesh_entities(d) || (_distributed && d == (D - 1)))
    {
      dolfin_mesh.init(d);
      if (_distributed)
        MeshPartitioning::number_entities(dolfin_mesh, d);
    }
  }

  // Create the UFC mesh
  const UFCMesh ufc_mesh(dolfin_mesh);

  // Initialize the UFC dofmap
  init_ufc_dofmap(*_ufc_dofmap, ufc_mesh, dolfin_mesh);

  // Build dof map
  DofMapBuilder::build(*this, dolfin_mesh, ufc_mesh, _distributed);
}
//-----------------------------------------------------------------------------
DofMap::DofMap(const DofMap& dofmap)
{
  error("DofMaps cannot be copied");
}
//-----------------------------------------------------------------------------
DofMap::DofMap(boost::shared_ptr<const ufc::dofmap> ufc_dofmap,
               const Mesh& dolfin_mesh) : _ufc_dofmap(ufc_dofmap->create()),
               ufc_offset(0), _is_view(false),
               _distributed(MPI::num_processes() > 1)
{
  assert(_ufc_dofmap);

  // Check for dimensional consistency between the dofmap and mesh
  check_dimensional_consistency(*_ufc_dofmap, dolfin_mesh);

  // Check that mesh has been ordered
  if (!dolfin_mesh.ordered())
     error("Mesh is not ordered according to the UFC numbering convention, consider calling mesh.order().");

  // Create the UFC mesh
  const UFCMesh ufc_mesh(dolfin_mesh);

  // Initialize the UFC dofmap
  init_ufc_dofmap(*_ufc_dofmap, ufc_mesh, dolfin_mesh);

  // Build dof map
  DofMapBuilder::build(*this, dolfin_mesh, ufc_mesh, _distributed);
}
//-----------------------------------------------------------------------------
DofMap::DofMap(const DofMap& parent_dofmap, const std::vector<uint>& component,
               const Mesh& mesh, bool distributed) : ufc_offset(0),
               _ownership_range(0, 0), _is_view(true),
               _distributed(distributed)
{
  // Ownership range is set to zero since dofmap is a view

  assert(component.size() > 0);

  // Create UFC mesh
  const UFCMesh ufc_mesh(mesh);

  // Initialise offset from parent
  uint offset = parent_dofmap.ufc_offset;

  // Get parent UFC dof map
  const ufc::dofmap& parent_ufc_dofmap = *(parent_dofmap._ufc_dofmap);

  // Extract ufc sub-dofmap from parent and get offset
  _ufc_dofmap.reset(extract_ufc_sub_dofmap(parent_ufc_dofmap, offset,
                                           component, ufc_mesh, mesh));
  assert(_ufc_dofmap);

  // Check for dimensional consistency between the dofmap and mesh
  check_dimensional_consistency(*_ufc_dofmap, mesh);

  // Set UFC offset
  this->ufc_offset = offset;

  // Initialise UFC dofmap
  init_ufc_dofmap(*_ufc_dofmap, ufc_mesh, mesh);

  // Resize dofmap data structure
  _dofmap.resize(mesh.num_cells());

  // Build sub-map based on UFC dofmap
  UFCCell ufc_cell(mesh);
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    const uint cell_index = cell->index();

    // Update to current cell
    ufc_cell.update(*cell);

    // Resize list for cell
    _dofmap[cell_index].resize(_ufc_dofmap->local_dimension(ufc_cell));

    // Tabulate sub-dofs on cell (using UFC map)
    _ufc_dofmap->tabulate_dofs(&_dofmap[cell_index][0], ufc_mesh, ufc_cell);

    // Add UFC offset
    for (uint i = 0; i < _dofmap[cell_index].size(); ++i)
      _dofmap[cell_index][i] += offset;
  }

  // Modify dofmap for non-UFC numbering
  ufc_map_to_dofmap.clear();
  _off_process_owner.clear();
  if (parent_dofmap.ufc_map_to_dofmap.size() > 0)
  {
    boost::unordered_map<uint, uint>::const_iterator ufc_to_current_dof;
    std::vector<std::vector<uint> >::iterator cell_map;
    std::vector<uint>::iterator dof;
    for (cell_map = _dofmap.begin(); cell_map != _dofmap.end(); ++cell_map)
    {
      for (dof = cell_map->begin(); dof != cell_map->end(); ++dof)
      {
        // Get dof index
        ufc_to_current_dof = parent_dofmap.ufc_map_to_dofmap.find(*dof);
        assert(ufc_to_current_dof != parent_dofmap.ufc_map_to_dofmap.end());

        // Add to ufc-to-current dof map
        ufc_map_to_dofmap.insert(*ufc_to_current_dof);

        // Set dof index
        *dof = ufc_to_current_dof->second;

        // Add to off-process dof owner map
        boost::unordered_map<uint, uint>::const_iterator parent_off_proc = parent_dofmap._off_process_owner.find(*dof);
        if (parent_off_proc != parent_dofmap._off_process_owner.end())
          _off_process_owner.insert(*parent_off_proc);
      }
    }
  }
}
//-----------------------------------------------------------------------------
DofMap::DofMap(boost::unordered_map<uint, uint>& collapsed_map,
               const DofMap& dofmap_view, const Mesh& mesh, bool distributed)
             : _ufc_dofmap(dofmap_view._ufc_dofmap->create()), ufc_offset(0),
               _is_view(false), _distributed(distributed)
{
  assert(_ufc_dofmap);

  // Check for dimensional consistency between the dofmap and mesh
  check_dimensional_consistency(*_ufc_dofmap, mesh);

  // Check that mesh has been ordered
  if (!mesh.ordered())
     error("Mesh is not ordered according to the UFC numbering convention, consider calling mesh.order().");

  // Create the UFC mesh
  const UFCMesh ufc_mesh(mesh);

  // Initialize the UFC dofmap
  init_ufc_dofmap(*_ufc_dofmap, ufc_mesh, mesh);

  // Build dof map
  DofMapBuilder::build(*this, mesh, ufc_mesh, _distributed);

  // Dimension checks
  assert(dofmap_view._dofmap.size() == mesh.num_cells());
  assert(global_dimension() == dofmap_view.global_dimension());
  assert(_dofmap.size() == mesh.num_cells());

  // FIXME: Could we use a std::vector instead of std::map if the collapsed
  //        dof map is contiguous (0, . . . , n)?

  // Build map from collapsed dof index to original dof index
  collapsed_map.clear();
  for (uint i = 0; i < mesh.num_cells(); ++i)
  {
    const std::vector<uint>& view_cell_dofs = dofmap_view._dofmap[i];
    const std::vector<uint>& cell_dofs = _dofmap[i];
    assert(view_cell_dofs.size() == cell_dofs.size());

    for (uint j = 0; j < view_cell_dofs.size(); ++j)
      collapsed_map[cell_dofs[j]] = view_cell_dofs[j];
  }
}
//-----------------------------------------------------------------------------
DofMap::~DofMap()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
bool DofMap::needs_mesh_entities(unsigned int d) const
{
  assert(_ufc_dofmap);
  return _ufc_dofmap->needs_mesh_entities(d);
}
//-----------------------------------------------------------------------------
unsigned int DofMap::global_dimension() const
{
  assert(_ufc_dofmap);
  assert(_ufc_dofmap->global_dimension() > 0);
  return _ufc_dofmap->global_dimension();
}
//-----------------------------------------------------------------------------
unsigned int DofMap::cell_dimension(uint cell_index) const
{
  assert(cell_index < _dofmap.size());
  return _dofmap[cell_index].size();
}
//-----------------------------------------------------------------------------
unsigned int DofMap::max_cell_dimension() const
{
  assert(_ufc_dofmap);
  return _ufc_dofmap->max_local_dimension();
}
//-----------------------------------------------------------------------------
unsigned int DofMap::geometric_dimension() const
{
  assert(_ufc_dofmap);
  return _ufc_dofmap->geometric_dimension();
}
//-----------------------------------------------------------------------------
unsigned int DofMap::num_facet_dofs() const
{
  assert(_ufc_dofmap);
  return _ufc_dofmap->num_facet_dofs();
}
//-----------------------------------------------------------------------------
std::pair<unsigned int, unsigned int> DofMap::ownership_range() const
{
  if (is_view())
    error("Cannot determine ownership range for sub-dofmaps.");

  return _ownership_range;
}
//-----------------------------------------------------------------------------
const boost::unordered_map<unsigned int, unsigned int>& DofMap::off_process_owner() const
{
  return _off_process_owner;
}
//-----------------------------------------------------------------------------
void DofMap::tabulate_facet_dofs(uint* dofs, uint local_facet) const
{
  assert(_ufc_dofmap);
  _ufc_dofmap->tabulate_facet_dofs(dofs, local_facet);
}
//-----------------------------------------------------------------------------
void DofMap::tabulate_coordinates(boost::multi_array<double, 2>& coordinates,
                                  const ufc::cell& ufc_cell) const
{
  // FIXME: This is a hack because UFC wants a double pointer for coordinates

  // Check dimensions
  if (coordinates.shape()[0] != cell_dimension(ufc_cell.index) ||
      coordinates.shape()[1] != geometric_dimension())
  {
    boost::multi_array<double, 2>::extent_gen extents;
    coordinates.resize(extents[cell_dimension(ufc_cell.index)]
		       [geometric_dimension()]);
  }

  // Set vertex coordinates
  const uint num_points = coordinates.size();
  std::vector<double*> coords(num_points);
  for (uint i = 0; i < num_points; ++i)
    coords[i] = &(coordinates[i][0]);

  // Tabulate coordinates
  _ufc_dofmap->tabulate_coordinates(&coords[0], ufc_cell);
}
//-----------------------------------------------------------------------------
void DofMap::tabulate_coordinates(boost::multi_array<double, 2>& coordinates,
                                  const Cell& cell) const
{
  UFCCell ufc_cell(cell);
  tabulate_coordinates(coordinates, ufc_cell);
}
//-----------------------------------------------------------------------------
DofMap* DofMap::copy(const Mesh& mesh) const
{
  boost::shared_ptr<const ufc::dofmap> ufc_dof_map(_ufc_dofmap->create());
  return new DofMap(ufc_dof_map, mesh);
}
//-----------------------------------------------------------------------------
DofMap* DofMap::extract_sub_dofmap(const std::vector<uint>& component,
                                   const Mesh& mesh) const
{
  return new DofMap(*this, component, mesh, _distributed);
}
//-----------------------------------------------------------------------------
DofMap* DofMap::collapse(boost::unordered_map<uint, uint>& collapsed_map,
                         const Mesh& mesh) const
{
  return new DofMap(collapsed_map, *this, mesh, _distributed);
}
//-----------------------------------------------------------------------------
ufc::dofmap* DofMap::extract_ufc_sub_dofmap(const ufc::dofmap& ufc_dofmap,
                                            uint& offset,
                                            const std::vector<uint>& component,
                                            const ufc::mesh ufc_mesh,
                                            const Mesh& dolfin_mesh)
{
  // Check if there are any sub systems
  if (ufc_dofmap.num_sub_dofmaps() == 0)
    error("Unable to extract sub system (there are no sub systems).");

  // Check that a sub system has been specified
  if (component.size() == 0)
    error("Unable to extract sub system (no sub system specified).");

  // Check the number of available sub systems
  if (component[0] >= ufc_dofmap.num_sub_dofmaps())
  {
    error("Unable to extract sub system %d (only %d sub systems defined).",
                  component[0], ufc_dofmap.num_sub_dofmaps());
  }

  // Add to offset if necessary
  for (uint i = 0; i < component[0]; i++)
  {
    // Extract sub dofmap
    boost::scoped_ptr<ufc::dofmap> ufc_tmp_dofmap(ufc_dofmap.create_sub_dofmap(i));
    assert(ufc_tmp_dofmap);

    // Initialise
    init_ufc_dofmap(*ufc_tmp_dofmap, ufc_mesh, dolfin_mesh);

    // Get offset
    offset += ufc_tmp_dofmap->global_dimension();
  }

  // Create UFC sub-system
  ufc::dofmap* sub_dofmap = ufc_dofmap.create_sub_dofmap(component[0]);
  assert(sub_dofmap);

  // Return sub-system if sub-sub-system should not be extracted, otherwise
  // recursively extract the sub sub system
  if (component.size() == 1)
    return sub_dofmap;
  else
  {
    std::vector<uint> sub_component;
    for (uint i = 1; i < component.size(); ++i)
      sub_component.push_back(component[i]);

    ufc::dofmap* sub_sub_dofmap = extract_ufc_sub_dofmap(*sub_dofmap, offset,
                                                     sub_component, ufc_mesh,
                                                     dolfin_mesh);
    delete sub_dofmap;
    return sub_sub_dofmap;
  }
}
//-----------------------------------------------------------------------------
void DofMap::init_ufc_dofmap(ufc::dofmap& dofmap,
                             const ufc::mesh ufc_mesh,
                             const Mesh& dolfin_mesh)
{
  // Check that we have all mesh entities
  for (uint d = 0; d <= dolfin_mesh.topology().dim(); ++d)
  {
    if (dofmap.needs_mesh_entities(d) && dolfin_mesh.num_entities(d) == 0)
      error("Unable to create function space, missing entities of dimension %d. Try calling mesh.init(%d).", d, d);
  }

  // Initialize UFC dof map
  const bool init_cells = dofmap.init_mesh(ufc_mesh);
  if (init_cells)
  {
    UFCCell ufc_cell(dolfin_mesh);
    for (CellIterator cell(dolfin_mesh); !cell.end(); ++cell)
    {
      ufc_cell.update(*cell);
      dofmap.init_cell(ufc_mesh, ufc_cell);
    }
    dofmap.init_cell_finalize();
  }
}
//-----------------------------------------------------------------------------
boost::unordered_set<dolfin::uint> DofMap::dofs() const
{
  // Build set of dofs
  boost::unordered_set<dolfin::uint> dof_list;
  std::vector<std::vector<uint> >::const_iterator cell_dofs;
  for (cell_dofs = _dofmap.begin(); cell_dofs != _dofmap.end(); ++cell_dofs)
    dof_list.insert(cell_dofs->begin(), cell_dofs->end());

  return dof_list;
}
//-----------------------------------------------------------------------------
void DofMap::renumber(const std::vector<uint>& renumbering_map)
{
  assert(global_dimension() == renumbering_map.size());

  // Update or build ufc-to-dofmap
  if (ufc_map_to_dofmap.size() == 0)
  {
    for (uint i = 0; i < _dofmap.size(); ++i)
      ufc_map_to_dofmap[i] = renumbering_map[i];
  }
  else
  {
    boost::unordered_map<dolfin::uint, uint>::iterator index_pair;
    for (index_pair = ufc_map_to_dofmap.begin(); index_pair != ufc_map_to_dofmap.end(); ++index_pair)
      index_pair->second = renumbering_map[index_pair->second];
  }

  // Re-number dofs for cell
  std::vector<std::vector<uint> >::iterator cell_map;
  std::vector<uint>::iterator dof;
  for (cell_map = _dofmap.begin(); cell_map != _dofmap.end(); ++cell_map)
    for (dof = cell_map->begin(); dof != cell_map->end(); ++dof)
      *dof = renumbering_map[*dof];
}
//-----------------------------------------------------------------------------
std::string DofMap::str(bool verbose) const
{
  // TODO: Display information on parallel stuff

  // Prefix with process number if running in parallel
  std::stringstream prefix;
  if (MPI::num_processes() > 1)
    prefix << "Process " << MPI::process_number() << ": ";

  std::stringstream s;
  s << prefix.str() << "<DofMap of global dimension " << global_dimension() << ">" << std::endl;
  if (verbose)
  {
    // Cell loop
    for (uint i = 0; i < _dofmap.size(); ++i)
    {
      s << prefix.str() << "Local cell index, cell dofmap dimension: " << i << ", " << _dofmap[i].size() << std::endl;

      // Local dof loop
      for (uint j = 0; j < _dofmap[i].size(); ++j)
        s << prefix.str() <<  "  " << "Local, global dof indices: " << j << ", " << _dofmap[i][j] << std::endl;
    }
  }

  return s.str();
}
//-----------------------------------------------------------------------------
void DofMap::check_dimensional_consistency(const ufc::dofmap& dofmap,
                                            const Mesh& mesh)
{
  // Check geometric dimension
  if (dofmap.geometric_dimension() != mesh.geometry().dim())
    error("Geometric dimension of the UFC dofmap and the Mesh do not match.");

  // Check topological dimension
  if (dofmap.topological_dimension() != mesh.topology().dim())
    error("Topological dimension of the UFC dofmap and the Mesh do not match.");
}
//-----------------------------------------------------------------------------
