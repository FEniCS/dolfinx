// Copyright (C) 2007-2011 Anders Logg and Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Martin Alnes, 2008
// Modified by Kent-Andre Mardal, 2009
// Modified by Ola Skavhaug, 2009
// Modified by Niclas Jansson, 2009
//
// First added:  2007-03-01
// Last changed: 2011-02-23

#include <dolfin/common/NoDeleter.h>
#include <dolfin/common/Set.h>
#include <dolfin/common/types.h>
#include <dolfin/log/LogStream.h>
#include <dolfin/common/MPI.h>
#include <dolfin/mesh/BoundaryMesh.h>
#include <dolfin/mesh/MeshData.h>
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
DofMap::DofMap(boost::shared_ptr<const ufc::dofmap> ufc_dofmap,
               const Mesh& dolfin_mesh) : _ufc_dofmap(ufc_dofmap->create()),
               ufc_offset(0), _is_view(false),
               _distributed(MPI::num_processes() > 1)
{
  assert(_ufc_dofmap);

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
               _is_view(true), _distributed(distributed)
{
  assert(component.size() > 0);

  // Create UFC mesh
  const UFCMesh ufc_mesh(mesh);

  // Initialise offset from parent
  uint offset = parent_dofmap.ufc_offset;

  // Get parent UFC dof map
  const ufc::dofmap& parent_ufc_dofmap = *(parent_dofmap._ufc_dofmap);

  // Extract UFC sub-dofmap from parent and get offset
  _ufc_dofmap.reset(extract_ufc_sub_dofmap(parent_ufc_dofmap, offset,
                                           component, ufc_mesh, mesh));
  assert(_ufc_dofmap);

  // Set UFC offset
  this->ufc_offset = offset;

  // Initialise ufc dofmap
  init_ufc_dofmap(*_ufc_dofmap, ufc_mesh, mesh);

  // Resize dofmap data structure
  dofmap.resize(mesh.num_cells());

  // Build sub-map based on UFC dofmap
  UFCCell ufc_cell(mesh);
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    const uint cell_index = cell->index();

    // Update to current cell
    ufc_cell.update(*cell);

    // Resize for list for cell
    dofmap[cell_index].resize(_ufc_dofmap->local_dimension(ufc_cell));

    // Tabulate sub-dofs on cell (using UFC map)
    _ufc_dofmap->tabulate_dofs(&dofmap[cell_index][0], ufc_mesh, ufc_cell);

    // Add UFC offset
    for (uint i = 0; i < dofmap[cell_index].size(); ++i)
      dofmap[cell_index][i] += offset;
  }

  // Modify dofmap for non-UFC numbering
  ufc_map_to_dofmap.clear();
  if (parent_dofmap.ufc_map_to_dofmap.size() > 0)
  {
    boost::unordered_map<uint, uint>::const_iterator ufc_to_current_dof;
    std::vector<std::vector<uint> >::iterator cell_map;
    std::vector<uint>::iterator dof;
    for (cell_map = dofmap.begin(); cell_map != dofmap.end(); ++cell_map)
    {
      for (dof = cell_map->begin(); dof != cell_map->end(); ++dof)
      {
        // Get dof index
        ufc_to_current_dof = parent_dofmap.ufc_map_to_dofmap.find(*dof);
        assert(ufc_to_current_dof != parent_dofmap.ufc_map_to_dofmap.end());

        // Add to map
        ufc_map_to_dofmap.insert(*ufc_to_current_dof);

        // Set dof index
        *dof = ufc_to_current_dof->second;
      }
    }
  }

  // Set local ownership range (set to zero since dofmap is a view)
  _ownership_range = std::make_pair(0, 0);

  // FIXME
  // Handle boost::unordered_map<uint, uint> _off_process_owner;

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
  assert(cell_index < dofmap.size());
  return dofmap[cell_index].size();
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
    error("Cannot determine ownership range for sub dofmaps.");

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
void DofMap::tabulate_coordinates(double** coordinates, const Cell& cell) const
{
  UFCCell ufc_cell(cell);
  tabulate_coordinates(coordinates, ufc_cell);
}
//-----------------------------------------------------------------------------
DofMap* DofMap::extract_sub_dofmap(const std::vector<uint>& component,
                                   const Mesh& dolfin_mesh) const
{
  return new DofMap(*this, component, dolfin_mesh, _distributed);
}
//-----------------------------------------------------------------------------
DofMap* DofMap::collapse(std::map<uint, uint>& collapsed_map,
                         const Mesh& dolfin_mesh) const
{
  assert(_ufc_dofmap);

  // Create new dof map (this sets ufc_offset = 0 and it will renumber the map
  // if runnning in parallel)
  boost::shared_ptr<const ufc::dofmap> wrapped_ufc_dofmap(_ufc_dofmap.get(), NoDeleter());
  DofMap* collapsed_dofmap = new DofMap(wrapped_ufc_dofmap, dolfin_mesh);

  // Dimension checks
  assert(collapsed_dofmap->global_dimension() == global_dimension());
  assert(collapsed_dofmap->dofmap.size() == dolfin_mesh.num_cells());
  assert(dofmap.size() == dolfin_mesh.num_cells());

  // FIXME: Could we use a std::vector instead of std::map if the collapsed
  //        dof map is contiguous (0, . . . ,n)?

  // Build map from collapsed dof index to original dof index
  collapsed_map.clear();
  for (uint i = 0; i < dolfin_mesh.num_cells(); ++i)
  {
    const std::vector<uint>& dofs = this->dofmap[i];
    const std::vector<uint>& collapsed_dofs = collapsed_dofmap->dofmap[i];
    assert(dofs.size() == collapsed_dofs.size());

    for (uint j = 0; j < dofs.size(); ++j)
      collapsed_map[collapsed_dofs[j]] = dofs[j];
  }

  // Create UFC mesh and cell
  UFCMesh ufc_mesh(dolfin_mesh);
  UFCCell ufc_cell(dolfin_mesh);

  // Build UFC-to-actual-dofs map
  std::vector<uint> ufc_dofs(collapsed_dofmap->max_cell_dimension());
  for (CellIterator cell(dolfin_mesh); !cell.end(); ++cell)
  {
    ufc_cell.update(*cell);

    // Tabulate UFC dofs (UFC map)
    collapsed_dofmap->_ufc_dofmap->tabulate_dofs(&ufc_dofs[0], ufc_mesh, ufc_cell);

    // Build UFC-to-actual-dofs map
    std::vector<uint>& collapsed_dofs = collapsed_dofmap->dofmap[cell->index()];
    for (uint j = 0; j < collapsed_dofs.size(); ++j)
      collapsed_dofmap->ufc_map_to_dofmap[ufc_dofs[j]] = collapsed_dofs[j];
  }

  // Reset offset of collapsed map
  collapsed_dofmap->ufc_offset = 0;

  // Set local ownership range

  // Update off-process owner

  return collapsed_dofmap;
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
    error("Unable to extract sub system %d (only %d sub systems defined).",
                  component[0], ufc_dofmap.num_sub_dofmaps());

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
  for (cell_dofs = dofmap.begin(); cell_dofs != dofmap.end(); ++cell_dofs)
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
    for (uint i = 0; i < dofmap.size(); ++i)
      ufc_map_to_dofmap[i] = renumbering_map[i];
  }
  else
  {
    boost::unordered_map<dolfin::uint, uint>::iterator index_pair;
    for (index_pair = ufc_map_to_dofmap.begin(); index_pair != ufc_map_to_dofmap.end(); ++index_pair)
      index_pair->second = renumbering_map[ index_pair->second ];
  }

  // Re-number dofs for cell
  for (uint i = 0; i < dofmap.size(); ++i)
  {
    for (uint j = 0; j < dofmap[i].size(); ++j)
      dofmap[i][j] = renumbering_map[ dofmap[i][j] ];
  }
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
    for (uint i = 0; i < dofmap.size(); ++i)
    {
      s << prefix.str() << "Local cell index, cell dofmap dimension: " << i << ", " << dofmap[i].size() << std::endl;

      // Local dof loop
      for (uint j = 0; j < dofmap[i].size(); ++j)
        s << prefix.str() <<  "  " << "Local, global dof indices: " << j << ", " << dofmap[i][j] << std::endl;
    }
  }

  return s.str();
}
//-----------------------------------------------------------------------------
