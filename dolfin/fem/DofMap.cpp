// Copyright (C) 2007-2010 Anders Logg and Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Martin Alnes, 2008
// Modified by Kent-Andre Mardal, 2009
// Modified by Ola Skavhaug, 2009
// Modified by Niclas Jansson, 2009
//
// First added:  2007-03-01
// Last changed: 2010-06-01

#include <dolfin/common/Set.h>
#include <dolfin/common/Timer.h>
#include <dolfin/log/LogStream.h>
#include <dolfin/main/MPI.h>
#include <dolfin/mesh/MeshPartitioning.h>
#include <dolfin/common/NoDeleter.h>
#include <dolfin/common/types.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/MeshData.h>
#include "UFCCell.h"
#include "UFCMesh.h"
#include "DofMapBuilder.h"
#include "DofMap.h"

#include <dolfin/mesh/BoundaryMesh.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
DofMap::DofMap(boost::shared_ptr<ufc::dof_map> ufc_dofmap, Mesh& dolfin_mesh)
             : _ufc_dofmap(ufc_dofmap), ufc_offset(0),
               _parallel(MPI::num_processes() > 1)
{
  assert(_ufc_dofmap);

  // Check that mesh has been ordered
  if (!dolfin_mesh.ordered())
     error("Mesh is not ordered according to the UFC numbering convention, consider calling mesh.order().");

  // Generate and number all mesh entities
  for (uint d = 1; d <= dolfin_mesh.topology().dim(); ++d)
  {
    if (_ufc_dofmap->needs_mesh_entities(d) ||
    	(_parallel && d == (dolfin_mesh.topology().dim() - 1)))
    {
      dolfin_mesh.init(d);
      if (_parallel)
        MeshPartitioning::number_entities(dolfin_mesh, d);
    }
  }

  // Create the UFC mesh
  UFCMesh ufc_mesh(dolfin_mesh);

  // Initialize the UFC dofmap
  init_ufc_dofmap(*_ufc_dofmap, ufc_mesh, dolfin_mesh);

  // Build dof map
  build(dolfin_mesh, ufc_mesh);
}
//-----------------------------------------------------------------------------
DofMap::DofMap(boost::shared_ptr<ufc::dof_map> ufc_dofmap,
               const Mesh& dolfin_mesh)
             : _ufc_dofmap(ufc_dofmap), ufc_offset(0),
               _parallel(MPI::num_processes() > 1)
{
  assert(_ufc_dofmap);

  // Check that mesh has been ordered
  if (!dolfin_mesh.ordered())
     error("Mesh is not ordered according to the UFC numbering convention, consider calling mesh.order().");

  // Create the UFC mesh
  UFCMesh ufc_mesh(dolfin_mesh);

  // Initialize the UFC dofmap
  init_ufc_dofmap(*_ufc_dofmap, ufc_mesh, dolfin_mesh);

  // Build dof map
  build(dolfin_mesh, ufc_mesh);
}
//-----------------------------------------------------------------------------
DofMap::DofMap(boost::shared_ptr<ufc::dof_map> ufc_dofmap,
               const UFCMesh& ufc_mesh)
             : _ufc_dofmap(ufc_dofmap), ufc_offset(0),
               _parallel(MPI::num_processes() > 1)

{
  assert(_ufc_dofmap);
}
//-----------------------------------------------------------------------------
DofMap::~DofMap()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
std::string DofMap::signature() const
{
  error("DofMap has been re-ordered. Cannot return signature string.");
  return _ufc_dofmap->signature();
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
unsigned int DofMap::local_dimension() const
{
  // FIXME: DofMap::dofs build a dolfin::Set each time, which is expensive
  return this->dofs(false).size();
}
//-----------------------------------------------------------------------------
unsigned int DofMap::dimension(uint cell_index) const
{
  return dofmap[cell_index].size();
}
//-----------------------------------------------------------------------------
unsigned int DofMap::max_local_dimension() const
{
  assert( _ufc_dofmap);
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
void DofMap::tabulate_dofs(uint* dofs, const Cell& cell) const
{
  // FIXME: Add assertion to test that this process has the dof.
  const uint cell_index = cell.index();
  std::copy(dofmap[cell_index].begin(), dofmap[cell_index].end(), dofs);
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
  assert(_ufc_dofmap);

  // Create UFC mesh
  UFCMesh ufc_mesh(dolfin_mesh);

  // Initialise offset
  uint offset = ufc_offset;

  // Recursively extract UFC sub-dofmap
  boost::shared_ptr<ufc::dof_map>
    ufc_sub_dof_map(extract_sub_dofmap(*_ufc_dofmap, offset, component, ufc_mesh, dolfin_mesh));

  // Initialise ufc sub-dofmap
  init_ufc_dofmap(*ufc_sub_dof_map, ufc_mesh, dolfin_mesh);

  // Create new dof map
  DofMap* sub_dofmap = new DofMap(ufc_sub_dof_map, ufc_mesh);

   // Create new dof map
  std::vector<std::vector<uint> >& sub_map = sub_dofmap->dofmap;
  sub_map.resize(dolfin_mesh.num_cells());

  // Build sub-map (based on UFC map)
  UFCCell ufc_cell(dolfin_mesh);
  for (CellIterator cell(dolfin_mesh); !cell.end(); ++cell)
  {
    const uint index = cell->index();

    // Update to current cell
    ufc_cell.update(*cell);

    // Resize for list for cell
    sub_map[index].resize( ufc_sub_dof_map->local_dimension(ufc_cell) );

    // Tabulate sub-dofs on cell (UFC map)
    ufc_sub_dof_map->tabulate_dofs(&sub_map[index][0], ufc_mesh, ufc_cell);

    // Add UFC offset
    for (uint i = 0; i < sub_map[index].size(); ++i)
      sub_map[index][i] += offset;
  }

  // Set UFC offset
  sub_dofmap->ufc_offset = offset;

  // Modify sub-map for non-UFC numbering
  if (ufc_map_to_dofmap.size() > 0)
  {
    for (uint i = 0; i < sub_map.size(); ++i)
    {
      for (uint j = 0; j < sub_map[i].size(); ++j)
      {
        std::map<uint, uint>::const_iterator new_dof_it = ufc_map_to_dofmap.find(sub_map[i][j]);
        assert(new_dof_it != ufc_map_to_dofmap.end());
        sub_map[i][j] = new_dof_it->second;
      }
    }

    // Copy of ufc-map-to-dofmap for new sub-dofmap
    sub_dofmap->ufc_map_to_dofmap = ufc_map_to_dofmap;
  }

  return sub_dofmap;
}
//-----------------------------------------------------------------------------
DofMap* DofMap::collapse(std::map<uint, uint>& collapsed_map,
                         const Mesh& dolfin_mesh) const
{
  assert(_ufc_dofmap);

  // Create new dof map (this set ufc_offset = 0 and it will renumber the map
  // if runnning in parallel)
  DofMap* collapsed_dof_map = new DofMap(_ufc_dofmap, dolfin_mesh);

  // Dimension checks
  assert(collapsed_dof_map->global_dimension() == global_dimension());
  assert(collapsed_dof_map->dofmap.size() == dolfin_mesh.num_cells());
  assert(dofmap.size() == dolfin_mesh.num_cells());

  // FIXME: Could we use a std::vector instead of std::map if the collapsed
  //        dof map is contiguous (0, . . . ,n)?

  // Build map from collapsed dof index to original dof index
  collapsed_map.clear();
  for (uint i = 0; i < dolfin_mesh.num_cells(); ++i)
  {
    const std::vector<uint> dofs = this->dofmap[i];
    const std::vector<uint> collapsed_dofs = collapsed_dof_map->dofmap[i];
    assert(dofs.size() == collapsed_dofs.size());

    for (uint j = 0; j < dofs.size(); ++j)
      collapsed_map[collapsed_dofs[j]] = dofs[j];
  }

  // Create UFC mesh and cell
  UFCMesh ufc_mesh(dolfin_mesh);
  UFCCell ufc_cell(dolfin_mesh);

  // Build UFC-to-actual-dofs map
  std::vector<uint> ufc_dofs(collapsed_dof_map->max_local_dimension());
  for (CellIterator cell(dolfin_mesh); !cell.end(); ++cell)
  {
    ufc_cell.update(*cell);

    // Tabulate UFC dofs (UFC map)
    collapsed_dof_map->_ufc_dofmap->tabulate_dofs(&ufc_dofs[0], ufc_mesh, ufc_cell);

    // Build UFC-to-actual-dofs map
    std::vector<uint>& collapsed_dofs = collapsed_dof_map->dofmap[cell->index()];
    for (uint j = 0; j < collapsed_dofs.size(); ++j)
      collapsed_dof_map->ufc_map_to_dofmap[ufc_dofs[j]] = collapsed_dofs[j];
  }

  // Reset offset of collapsed map
  collapsed_dof_map->ufc_offset = 0;

  return collapsed_dof_map;
}
//-----------------------------------------------------------------------------
ufc::dof_map* DofMap::extract_sub_dofmap(const ufc::dof_map& ufc_dofmap,
                                         uint& offset,
                                         const std::vector<uint>& component,
                                         const ufc::mesh ufc_mesh,
                                         const Mesh& dolfin_mesh)
{
  // Check if there are any sub systems
  if (ufc_dofmap.num_sub_dof_maps() == 0)
    error("Unable to extract sub system (there are no sub systems).");

  // Check that a sub system has been specified
  if (component.size() == 0)
    error("Unable to extract sub system (no sub system specified).");

  // Check the number of available sub systems
  if (component[0] >= ufc_dofmap.num_sub_dof_maps())
    error("Unable to extract sub system %d (only %d sub systems defined).",
                  component[0], ufc_dofmap.num_sub_dof_maps());

  // Add to offset if necessary
  for (uint i = 0; i < component[0]; i++)
  {
    // Extract sub dofmap
    boost::scoped_ptr<ufc::dof_map> ufc_tmp_dofmap(ufc_dofmap.create_sub_dof_map(i));
    assert(ufc_tmp_dofmap);

    // Initialise
    init_ufc_dofmap(*ufc_tmp_dofmap, ufc_mesh, dolfin_mesh);

    // Get offset
    offset += ufc_tmp_dofmap->global_dimension();
  }

  // Create UFC sub-system
  ufc::dof_map* sub_dof_map = ufc_dofmap.create_sub_dof_map(component[0]);

  // Return sub-system if sub-sub-system should not be extracted, otherwise
  // recursively extract the sub sub system
  if (component.size() == 1)
    return sub_dof_map;
  else
  {
    std::vector<uint> sub_component;
    for (uint i = 1; i < component.size(); i++)
      sub_component.push_back(component[i]);

    ufc::dof_map* sub_sub_dof_map = extract_sub_dofmap(*sub_dof_map, offset,
                                                       sub_component, ufc_mesh,
                                                       dolfin_mesh);
    delete sub_dof_map;
    return sub_sub_dof_map;
  }
}
//-----------------------------------------------------------------------------
void DofMap::build(const Mesh& dolfin_mesh, const UFCMesh& ufc_mesh)
{
  // Start timer for dofmap initialization
  Timer t0("Init dofmap");

  // Build dofmap from ufc::dofmap
  dofmap.resize(dolfin_mesh.num_cells());
  dolfin::UFCCell ufc_cell(dolfin_mesh);
  for (dolfin::CellIterator cell(dolfin_mesh); !cell.end(); ++cell)
  {
    // Update ufc cell
    ufc_cell.update(*cell);

    // Get standard local dimension
    const unsigned int local_dim = _ufc_dofmap->local_dimension(ufc_cell);
    dofmap[cell->index()].resize(local_dim);

    // Tabulate standard dof map
    _ufc_dofmap->tabulate_dofs(&dofmap[cell->index()][0], ufc_mesh, ufc_cell);
  }

  // Build (renumber) dofmap when running in parallel
  if (_parallel)
    DofMapBuilder::parallel_build(*this, dolfin_mesh);
}
//-----------------------------------------------------------------------------
void DofMap::init_ufc_dofmap(ufc::dof_map& dofmap,
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
dolfin::Set<dolfin::uint> DofMap::dofs(bool sort) const
{
  // Build set of dofs
  dolfin::Set<uint> dof_list;
  for (uint i = 0; i < dofmap.size(); ++i)
    for (uint j = 0; j < dofmap[i].size(); ++j)
      dof_list.insert(dofmap[i][j]);

  // Sort set of dofs if required
  if(sort)
    dof_list.sort();

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
      ufc_map_to_dofmap[i] = renumbering_map[ i ];
  }
  else
  {
    std::map<dolfin::uint, uint>::iterator index_pair;
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
