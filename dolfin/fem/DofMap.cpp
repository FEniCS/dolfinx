// Copyright (C) 2007-2016 Anders Logg and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <unordered_map>

#include "DofMap.h"
#include "DofMapBuilder.h"
#include <dolfin/common/MPI.h>
#include <dolfin/common/types.h>
#include <dolfin/la/PETScVector.h>
#include <dolfin/log/LogStream.h>
#include <dolfin/mesh/MeshIterator.h>
#include <dolfin/mesh/PeriodicBoundaryComputation.h>
#include <dolfin/mesh/Vertex.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
DofMap::DofMap(std::shared_ptr<const ufc::dofmap> ufc_dofmap, const Mesh& mesh)
    : _cell_dimension(0), _ufc_dofmap(ufc_dofmap), _is_view(false),
      _global_dimension(0), _ufc_offset(0),
      _index_map(new IndexMap(mesh.mpi_comm()))
{
  dolfin_assert(_ufc_dofmap);

  // Call dofmap builder
  DofMapBuilder::build(*this, mesh, std::shared_ptr<const SubDomain>());
}
//-----------------------------------------------------------------------------
DofMap::DofMap(std::shared_ptr<const ufc::dofmap> ufc_dofmap, const Mesh& mesh,
               std::shared_ptr<const SubDomain> constrained_domain)
    : _cell_dimension(0), _ufc_dofmap(ufc_dofmap), _is_view(false),
      _global_dimension(0), _ufc_offset(0),
      _index_map(new IndexMap(mesh.mpi_comm()))
{
  dolfin_assert(_ufc_dofmap);

  // Store constrained domain in base class
  this->constrained_domain = constrained_domain;

  // Call dofmap builder
  DofMapBuilder::build(*this, mesh, constrained_domain);
}
//-----------------------------------------------------------------------------
DofMap::DofMap(const DofMap& parent_dofmap,
               const std::vector<std::size_t>& component, const Mesh& mesh)
    : _cell_dimension(0), _ufc_dofmap(0), _is_view(true), _global_dimension(0),
      _ufc_offset(0), _index_map(parent_dofmap._index_map)
{
  // Build sub-dofmap
  DofMapBuilder::build_sub_map_view(*this, parent_dofmap, component, mesh);
}
//-----------------------------------------------------------------------------
DofMap::DofMap(std::unordered_map<std::size_t, std::size_t>& collapsed_map,
               const DofMap& dofmap_view, const Mesh& mesh)
    : _cell_dimension(0), _ufc_dofmap(dofmap_view._ufc_dofmap), _is_view(false),
      _global_dimension(0), _ufc_offset(0),
      _index_map(new IndexMap(mesh.mpi_comm()))
{
  dolfin_assert(_ufc_dofmap);

  // Check for dimensional consistency between the dofmap and mesh
  check_dimensional_consistency(*_ufc_dofmap, mesh);

  // Check that mesh has been ordered
  if (!mesh.ordered())
  {
    dolfin_error(
        "DofMap.cpp", "create mapping of degrees of freedom",
        "Mesh is not ordered according to the UFC numbering convention. "
        "Consider calling mesh.order()");
  }

  // Check dimensional consistency between UFC dofmap and the mesh
  check_provided_entities(*_ufc_dofmap, mesh);

  // Build new dof map
  DofMapBuilder::build(*this, mesh, constrained_domain);

  // Dimension sanity checks
  dolfin_assert(dofmap_view._dofmap.size()
                == mesh.num_cells() * dofmap_view._cell_dimension);
  dolfin_assert(global_dimension() == dofmap_view.global_dimension());
  dolfin_assert(_dofmap.size() == mesh.num_cells() * _cell_dimension);

  // FIXME: Could we use a std::vector instead of std::map if the
  //        collapsed dof map is contiguous (0, . . . , n)?

  // Build map from collapsed dof index to original dof index
  collapsed_map.clear();
  for (std::int64_t i = 0; i < mesh.num_cells(); ++i)
  {
    auto view_cell_dofs = dofmap_view.cell_dofs(i);
    auto cell_dofs = this->cell_dofs(i);
    dolfin_assert(view_cell_dofs.size() == cell_dofs.size());

    for (Eigen::Index j = 0; j < view_cell_dofs.size(); ++j)
      collapsed_map[cell_dofs[j]] = view_cell_dofs[j];
  }
}
//-----------------------------------------------------------------------------
DofMap::DofMap(const DofMap& dofmap) : _index_map(dofmap._index_map)
{
  // Copy data
  _dofmap = dofmap._dofmap;
  _cell_dimension = dofmap._cell_dimension;
  _ufc_dofmap = dofmap._ufc_dofmap;
  _num_mesh_entities_global = dofmap._num_mesh_entities_global;
  _ufc_local_to_local = dofmap._ufc_local_to_local;
  _is_view = dofmap._is_view;
  _global_dimension = dofmap._global_dimension;
  _ufc_offset = dofmap._ufc_offset;
  _shared_nodes = dofmap._shared_nodes;
  _neighbours = dofmap._neighbours;
  constrained_domain = dofmap.constrained_domain;
}
//-----------------------------------------------------------------------------
DofMap::~DofMap()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
std::int64_t DofMap::global_dimension() const { return _global_dimension; }
//-----------------------------------------------------------------------------
std::size_t DofMap::num_element_dofs(std::size_t cell_index) const
{
  return _cell_dimension;
}
//-----------------------------------------------------------------------------
std::size_t DofMap::max_element_dofs() const
{
  dolfin_assert(_ufc_dofmap);
  return _ufc_dofmap->num_element_dofs();
}
//-----------------------------------------------------------------------------
std::size_t DofMap::num_entity_dofs(std::size_t entity_dim) const
{
  dolfin_assert(_ufc_dofmap);
  return _ufc_dofmap->num_entity_dofs(entity_dim);
}
//-----------------------------------------------------------------------------
std::size_t DofMap::num_entity_closure_dofs(std::size_t entity_dim) const
{
  dolfin_assert(_ufc_dofmap);
  return _ufc_dofmap->num_entity_closure_dofs(entity_dim);
}
//-----------------------------------------------------------------------------
std::size_t DofMap::num_facet_dofs() const
{
  dolfin_assert(_ufc_dofmap);
  return _ufc_dofmap->num_facet_dofs();
}
//-----------------------------------------------------------------------------
std::array<std::int64_t, 2> DofMap::ownership_range() const
{
  auto block_range = _index_map->local_range();
  std::int64_t bs = _index_map->block_size();
  return {{bs * block_range[0], bs * block_range[1]}};
}
//-----------------------------------------------------------------------------
const std::unordered_map<int, std::vector<int>>& DofMap::shared_nodes() const
{
  return _shared_nodes;
}
//-----------------------------------------------------------------------------
const std::set<int>& DofMap::neighbours() const { return _neighbours; }
//-----------------------------------------------------------------------------
std::vector<dolfin::la_index_t> DofMap::entity_closure_dofs(
    const Mesh& mesh, std::size_t entity_dim,
    const std::vector<std::size_t>& entity_indices) const
{
  // Get some dimensions
  const std::size_t top_dim = mesh.topology().dim();
  const std::size_t dofs_per_entity = num_entity_closure_dofs(entity_dim);

  // Initialize entity to cell connections
  mesh.init(entity_dim, top_dim);

  // Allocate the the array to return
  const std::size_t num_marked_entities = entity_indices.size();
  std::vector<dolfin::la_index_t> entity_to_dofs(num_marked_entities
                                                 * dofs_per_entity);

  // Allocate data for tabulating local to local map
  std::vector<std::size_t> local_to_local_map(dofs_per_entity);

  // Iterate over entities
  std::size_t local_entity_ind = 0;
  for (std::size_t i = 0; i < num_marked_entities; ++i)
  {
    MeshEntity entity(mesh, entity_dim, entity_indices[i]);

    // Get the first cell connected to the entity
    const Cell cell(mesh, entity.entities(top_dim)[0]);

    // Find local entity number
    for (std::size_t local_i = 0; local_i < cell.num_entities(entity_dim);
         ++local_i)
    {
      if (cell.entities(entity_dim)[local_i] == entity.index())
      {
        local_entity_ind = local_i;
        break;
      }
    }

    // Get all cell dofs
    const auto cell_dof_list = cell_dofs(cell.index());

    // Tabulate local to local map of dofs on local entity
    tabulate_entity_closure_dofs(local_to_local_map, entity_dim,
                                 local_entity_ind);

    // Fill local dofs for the entity
    for (std::size_t local_dof = 0; local_dof < dofs_per_entity; ++local_dof)
    {
      // Map dofs
      const dolfin::la_index_t global_dof
          = cell_dof_list[local_to_local_map[local_dof]];
      entity_to_dofs[dofs_per_entity * i + local_dof] = global_dof;
    }
  }
  return entity_to_dofs;
}
//-----------------------------------------------------------------------------
std::vector<dolfin::la_index_t>
DofMap::entity_closure_dofs(const Mesh& mesh, std::size_t entity_dim) const
{
  // Get some dimensions
  const std::size_t top_dim = mesh.topology().dim();
  const std::size_t dofs_per_entity = num_entity_closure_dofs(entity_dim);
  const std::size_t num_mesh_entities = mesh.num_entities(entity_dim);

  // Initialize entity to cell connections
  mesh.init(entity_dim, top_dim);

  // Allocate the the array to return
  std::vector<dolfin::la_index_t> entity_to_dofs(num_mesh_entities
                                                 * dofs_per_entity);

  // Allocate data for tabulating local to local map
  std::vector<std::size_t> local_to_local_map(dofs_per_entity);

  // Iterate over entities
  std::size_t local_entity_ind = 0;
  for (auto& entity : MeshRange<MeshEntity>(mesh, entity_dim))
  {
    // Get the first cell connected to the entity
    const Cell cell(mesh, entity.entities(top_dim)[0]);

    // Find local entity number
    for (std::size_t local_i = 0; local_i < cell.num_entities(entity_dim);
         ++local_i)
    {
      if (cell.entities(entity_dim)[local_i] == entity.index())
      {
        local_entity_ind = local_i;
        break;
      }
    }

    // Get all cell dofs
    const auto cell_dof_list = cell_dofs(cell.index());

    // Tabulate local to local map of dofs on local entity
    tabulate_entity_closure_dofs(local_to_local_map, entity_dim,
                                 local_entity_ind);

    // Fill local dofs for the entity
    for (std::size_t local_dof = 0; local_dof < dofs_per_entity; ++local_dof)
    {
      // Map dofs
      const dolfin::la_index_t global_dof
          = cell_dof_list[local_to_local_map[local_dof]];
      entity_to_dofs[dofs_per_entity * entity.index() + local_dof] = global_dof;
    }
  }
  return entity_to_dofs;
}
//-----------------------------------------------------------------------------
std::vector<dolfin::la_index_t>
DofMap::entity_dofs(const Mesh& mesh, std::size_t entity_dim,
                    const std::vector<std::size_t>& entity_indices) const
{
  // Get some dimensions
  const std::size_t top_dim = mesh.topology().dim();
  const std::size_t dofs_per_entity = num_entity_dofs(entity_dim);

  // Initialize entity to cell connections
  mesh.init(entity_dim, top_dim);

  // Allocate the the array to return
  const std::size_t num_marked_entities = entity_indices.size();
  std::vector<dolfin::la_index_t> entity_to_dofs(num_marked_entities
                                                 * dofs_per_entity);

  // Allocate data for tabulating local to local map
  std::vector<std::size_t> local_to_local_map(dofs_per_entity);

  // Iterate over entities
  std::size_t local_entity_ind = 0;
  for (std::size_t i = 0; i < num_marked_entities; ++i)
  {
    MeshEntity entity(mesh, entity_dim, entity_indices[i]);

    // Get the first cell connected to the entity
    const Cell cell(mesh, entity.entities(top_dim)[0]);

    // Find local entity number
    for (std::size_t local_i = 0; local_i < cell.num_entities(entity_dim);
         ++local_i)
    {
      if (cell.entities(entity_dim)[local_i] == entity.index())
      {
        local_entity_ind = local_i;
        break;
      }
    }

    // Get all cell dofs
    const auto cell_dof_list = cell_dofs(cell.index());

    // Tabulate local to local map of dofs on local entity
    tabulate_entity_dofs(local_to_local_map, entity_dim, local_entity_ind);

    // Fill local dofs for the entity
    for (std::size_t local_dof = 0; local_dof < dofs_per_entity; ++local_dof)
    {
      // Map dofs
      const dolfin::la_index_t global_dof
          = cell_dof_list[local_to_local_map[local_dof]];
      entity_to_dofs[dofs_per_entity * i + local_dof] = global_dof;
    }
  }
  return entity_to_dofs;
}
//-----------------------------------------------------------------------------
std::vector<dolfin::la_index_t>
DofMap::entity_dofs(const Mesh& mesh, std::size_t entity_dim) const
{
  // Get some dimensions
  const std::size_t top_dim = mesh.topology().dim();
  const std::size_t dofs_per_entity = num_entity_dofs(entity_dim);
  const std::size_t num_mesh_entities = mesh.num_entities(entity_dim);

  // Initialize entity to cell connections
  mesh.init(entity_dim, top_dim);

  // Allocate the the array to return
  std::vector<dolfin::la_index_t> entity_to_dofs(num_mesh_entities
                                                 * dofs_per_entity);

  // Allocate data for tabulating local to local map
  std::vector<std::size_t> local_to_local_map(dofs_per_entity);

  // Iterate over entities
  std::size_t local_entity_ind = 0;
  for (auto& entity : MeshRange<MeshEntity>(mesh, entity_dim))
  {
    // Get the first cell connected to the entity
    const Cell cell(mesh, entity.entities(top_dim)[0]);

    // Find local entity number
    for (std::size_t local_i = 0; local_i < cell.num_entities(entity_dim);
         ++local_i)
    {
      if (cell.entities(entity_dim)[local_i] == entity.index())
      {
        local_entity_ind = local_i;
        break;
      }
    }

    // Get all cell dofs
    const auto cell_dof_list = cell_dofs(cell.index());

    // Tabulate local to local map of dofs on local entity
    tabulate_entity_dofs(local_to_local_map, entity_dim, local_entity_ind);

    // Fill local dofs for the entity
    for (std::size_t local_dof = 0; local_dof < dofs_per_entity; ++local_dof)
    {
      // Map dofs
      const dolfin::la_index_t global_dof
          = cell_dof_list[local_to_local_map[local_dof]];
      entity_to_dofs[dofs_per_entity * entity.index() + local_dof] = global_dof;
    }
  }
  return entity_to_dofs;
}
//-----------------------------------------------------------------------------
void DofMap::tabulate_facet_dofs(std::vector<std::size_t>& element_dofs,
                                 std::size_t cell_facet_index) const
{
  dolfin_assert(_ufc_dofmap);
  element_dofs.resize(_ufc_dofmap->num_facet_dofs());
  _ufc_dofmap->tabulate_facet_dofs(element_dofs.data(), cell_facet_index);
}
//-----------------------------------------------------------------------------
void DofMap::tabulate_entity_dofs(std::vector<std::size_t>& element_dofs,
                                  std::size_t entity_dim,
                                  std::size_t cell_entity_index) const
{
  dolfin_assert(_ufc_dofmap);
  if (_ufc_dofmap->num_entity_dofs(entity_dim) == 0)
    return;

  element_dofs.resize(_ufc_dofmap->num_entity_dofs(entity_dim));
  _ufc_dofmap->tabulate_entity_dofs(&element_dofs[0], entity_dim,
                                    cell_entity_index);
}
//-----------------------------------------------------------------------------
void DofMap::tabulate_entity_closure_dofs(
    std::vector<std::size_t>& element_dofs, std::size_t entity_dim,
    std::size_t cell_entity_index) const
{
  dolfin_assert(_ufc_dofmap);
  if (_ufc_dofmap->num_entity_closure_dofs(entity_dim) == 0)
    return;

  element_dofs.resize(_ufc_dofmap->num_entity_closure_dofs(entity_dim));
  _ufc_dofmap->tabulate_entity_closure_dofs(&element_dofs[0], entity_dim,
                                            cell_entity_index);
}
//-----------------------------------------------------------------------------
std::shared_ptr<GenericDofMap> DofMap::copy() const
{
  return std::shared_ptr<GenericDofMap>(new DofMap(*this));
}
//-----------------------------------------------------------------------------
std::shared_ptr<GenericDofMap> DofMap::create(const Mesh& new_mesh) const
{
  // Get underlying UFC dof map
  std::shared_ptr<const ufc::dofmap> ufc_dof_map(_ufc_dofmap);
  return std::shared_ptr<GenericDofMap>(new DofMap(ufc_dof_map, new_mesh));
}
//-----------------------------------------------------------------------------
std::shared_ptr<GenericDofMap>
DofMap::extract_sub_dofmap(const std::vector<std::size_t>& component,
                           const Mesh& mesh) const
{
  return std::shared_ptr<GenericDofMap>(new DofMap(*this, component, mesh));
}
//-----------------------------------------------------------------------------
std::shared_ptr<GenericDofMap>
DofMap::collapse(std::unordered_map<std::size_t, std::size_t>& collapsed_map,
                 const Mesh& mesh) const
{
  return std::shared_ptr<GenericDofMap>(new DofMap(collapsed_map, *this, mesh));
}
//-----------------------------------------------------------------------------
std::vector<dolfin::la_index_t> DofMap::dofs(const Mesh& mesh,
                                             std::size_t dim) const
{
  // FIXME: This function requires a special case when dim ==
  // mesh.topology().dim() because of they way DOLFIN handles d-d
  // connectivity. Change DOLFIN behaviour.

  // Check number of dofs per entity (on cell cell)
  const std::size_t num_dofs_per_entity = num_entity_dofs(dim);

  // Return empty vector if not dofs on requested entity
  if (num_dofs_per_entity == 0)
    return std::vector<dolfin::la_index_t>();

  // Vector to hold list of dofs
  std::vector<dolfin::la_index_t> dof_list(mesh.num_entities(dim)
                                           * num_dofs_per_entity);

  // Iterate over cells
  if (dim < mesh.topology().dim())
  {
    std::vector<std::size_t> entity_dofs_local;
    for (auto& c : MeshRange<Cell>(mesh))
    {
      // Get local-to-global dofmap for cell
      const auto cell_dof_list = cell_dofs(c.index());

      // Loop over all entities of dimension dim
      unsigned int local_index = 0;
      for (auto& e : EntityRange<MeshEntity>(c, dim))
      {
        // Tabulate cell-wise index of all dofs on entity
        tabulate_entity_dofs(entity_dofs_local, dim, local_index);

        // Get dof index and add to list
        for (std::size_t i = 0; i < entity_dofs_local.size(); ++i)
        {
          const std::size_t entity_dof_local = entity_dofs_local[i];
          const dolfin::la_index_t dof_index = cell_dof_list[entity_dof_local];
          dolfin_assert(e.index() * num_dofs_per_entity + i < dof_list.size());
          dof_list[e.index() * num_dofs_per_entity + i] = dof_index;
        }

        ++local_index;
      }
    }
  }
  else
  {
    std::vector<std::size_t> entity_dofs_local;
    for (auto& c : MeshRange<Cell>(mesh))
    {
      // Get local-to-global dofmap for cell
      const auto cell_dof_list = cell_dofs(c.index());

      // Tabulate cell-wise index of all dofs on entity
      tabulate_entity_dofs(entity_dofs_local, dim, 0);

      // Get dof index and add to list
      for (std::size_t i = 0; i < entity_dofs_local.size(); ++i)
      {
        const std::size_t entity_dof_local = entity_dofs_local[i];
        const dolfin::la_index_t dof_index = cell_dof_list[entity_dof_local];
        dolfin_assert(c.index() * num_dofs_per_entity + i < dof_list.size());
        dof_list[c.index() * num_dofs_per_entity + i] = dof_index;
      }
    }
  }

  return dof_list;
}
//-----------------------------------------------------------------------------
std::vector<dolfin::la_index_t> DofMap::dofs() const
{
  // Create vector to hold dofs
  std::vector<la_index_t> _dofs;
  _dofs.reserve(_dofmap.size() * max_element_dofs());

  const std::size_t bs = _index_map->block_size();
  const dolfin::la_index_t local_ownership_size
      = bs * _index_map->size(IndexMap::MapSize::OWNED);
  const std::size_t global_offset = bs * _index_map->local_range()[0];

  // Insert all dofs into a vector (will contain duplicates)
  for (auto dof : _dofmap)
  {
    if (dof >= 0 && dof < local_ownership_size)
      _dofs.push_back(dof + global_offset);
  }

  // Sort dofs (required to later remove duplicates)
  std::sort(_dofs.begin(), _dofs.end());

  // Remove duplicates
  _dofs.erase(std::unique(_dofs.begin(), _dofs.end()), _dofs.end());

  return _dofs;
}
//-----------------------------------------------------------------------------
void DofMap::set(PETScVector& x, double value) const
{
  dolfin_assert(_dofmap.size() % _cell_dimension == 0);
  const std::size_t num_cells = _dofmap.size() / _cell_dimension;

  std::vector<double> _value(_cell_dimension, value);
  for (std::size_t i = 0; i < num_cells; ++i)
  {
    auto dofs = cell_dofs(i);
    x.set_local(_value.data(), dofs.size(), dofs.data());
  }

  x.apply();
}
//-----------------------------------------------------------------------------
void DofMap::tabulate_local_to_global_dofs(
    std::vector<std::size_t>& local_to_global_map) const
{
  // FIXME: use IndexMap::local_to_global_index?

  const std::size_t bs = _index_map->block_size();
  const std::vector<std::size_t>& local_to_global_unowned
      = _index_map->local_to_global_unowned();
  const std::size_t local_ownership_size
      = bs * _index_map->size(IndexMap::MapSize::OWNED);
  local_to_global_map.resize(bs * _index_map->size(IndexMap::MapSize::ALL));

  const std::size_t global_offset = bs * _index_map->local_range()[0];
  for (std::size_t i = 0; i < local_ownership_size; ++i)
    local_to_global_map[i] = i + global_offset;

  for (std::size_t node = 0;
       node < _index_map->local_to_global_unowned().size(); ++node)
  {
    for (std::size_t component = 0; component < bs; ++component)
    {
      local_to_global_map[bs * node + component + local_ownership_size]
          = bs * local_to_global_unowned[node] + component;
    }
  }
}
//-----------------------------------------------------------------------------
void DofMap::check_provided_entities(const ufc::dofmap& dofmap,
                                     const Mesh& mesh)
{
  // Check that we have all mesh entities
  for (std::size_t d = 0; d <= mesh.topology().dim(); ++d)
  {
    if (dofmap.needs_mesh_entities(d) && mesh.num_entities(d) == 0)
    {
      dolfin_error(
          "DofMap.cpp", "initialize mapping of degrees of freedom",
          "Missing entities of dimension %d. Try calling mesh.init(%d)", d, d);
    }
  }
}
//-----------------------------------------------------------------------------
std::string DofMap::str(bool verbose) const
{
  std::stringstream s;
  s << "<DofMap of global dimension " << global_dimension() << ">" << std::endl;
  if (verbose)
  {
    // Cell loop
    dolfin_assert(_dofmap.size() % _cell_dimension == 0);
    const std::size_t ncells = _dofmap.size() / _cell_dimension;

    for (std::size_t i = 0; i < ncells; ++i)
    {
      s << "Local cell index, cell dofmap dimension: " << i << ", "
        << _cell_dimension << std::endl;

      // Local dof loop
      for (std::size_t j = 0; j < _cell_dimension; ++j)
      {
        s << "  "
          << "Local, global dof indices: " << j << ", "
          << _dofmap[i * _cell_dimension + j] << std::endl;
      }
    }
  }

  return s.str();
}
//-----------------------------------------------------------------------------
