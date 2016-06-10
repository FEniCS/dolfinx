// Copyright (C) 2007-2016 Anders Logg and Garth N. Wells
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
// Modified by Martin Alnes 2008-2015
// Modified by Kent-Andre Mardal 2009
// Modified by Ola Skavhaug 2009
// Modified by Niclas Jansson 2009
// Modified by Joachim B Haga 2012
// Modified by Mikael Mortensen 2012
// Modified by Jan Blechta 2013

#include <unordered_map>

#include <dolfin/common/MPI.h>
#include <dolfin/common/NoDeleter.h>
#include <dolfin/common/types.h>
#include <dolfin/la/GenericVector.h>
#include <dolfin/log/LogStream.h>
#include <dolfin/mesh/MeshEntityIterator.h>
#include <dolfin/mesh/PeriodicBoundaryComputation.h>
#include <dolfin/mesh/Vertex.h>
#include "DofMapBuilder.h"
#include "DofMap.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
DofMap::DofMap(std::shared_ptr<const ufc::dofmap> ufc_dofmap,
               const Mesh& mesh)
  : _cell_dimension(0), _ufc_dofmap(ufc_dofmap), _is_view(false),
    _global_dimension(0), _ufc_offset(0), _multimesh_offset(0),
    _index_map(new IndexMap(mesh.mpi_comm()))
{
  dolfin_assert(_ufc_dofmap);

  // Call dofmap builder
  DofMapBuilder::build(*this, mesh, std::shared_ptr<const SubDomain>());
}
//-----------------------------------------------------------------------------
DofMap::DofMap(std::shared_ptr<const ufc::dofmap> ufc_dofmap,
               const Mesh& mesh,
               std::shared_ptr<const SubDomain> constrained_domain)
  : _cell_dimension(0), _ufc_dofmap(ufc_dofmap), _is_view(false),
    _global_dimension(0), _ufc_offset(0), _multimesh_offset(0),
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
  : _cell_dimension(0), _ufc_dofmap(0), _is_view(true),
    _global_dimension(0), _ufc_offset(0), _multimesh_offset(0),
    _index_map(parent_dofmap._index_map)
{
  // Build sub-dofmap
  DofMapBuilder::build_sub_map_view(*this, parent_dofmap, component, mesh);
}
//-----------------------------------------------------------------------------
DofMap::DofMap(std::unordered_map<std::size_t, std::size_t>& collapsed_map,
               const DofMap& dofmap_view, const Mesh& mesh)
  : _cell_dimension(0), _ufc_dofmap(dofmap_view._ufc_dofmap), _is_view(false),
    _global_dimension(0), _ufc_offset(0), _multimesh_offset(0),
    _index_map(new IndexMap(mesh.mpi_comm()))
{
  dolfin_assert(_ufc_dofmap);

  // Check for dimensional consistency between the dofmap and mesh
  check_dimensional_consistency(*_ufc_dofmap, mesh);

  // Check that mesh has been ordered
  if (!mesh.ordered())
  {
     dolfin_error("DofMap.cpp",
                  "create mapping of degrees of freedom",
                  "Mesh is not ordered according to the UFC numbering convention. "
                  "Consider calling mesh.order()");
  }

  // Check dimensional consistency between UFC dofmap and the mesh
  check_provided_entities(*_ufc_dofmap, mesh);

  // Build new dof map
  DofMapBuilder::build(*this, mesh, constrained_domain);

  // Dimension sanity checks
  dolfin_assert(dofmap_view._dofmap.size()
                == mesh.num_cells()*dofmap_view._cell_dimension);
  dolfin_assert(global_dimension() == dofmap_view.global_dimension());
  dolfin_assert(_dofmap.size() == mesh.num_cells()*_cell_dimension);

  // FIXME: Could we use a std::vector instead of std::map if the
  //        collapsed dof map is contiguous (0, . . . , n)?

  // Build map from collapsed dof index to original dof index
  collapsed_map.clear();
  for (std::size_t i = 0; i < mesh.num_cells(); ++i)
  {
    const ArrayView<const dolfin::la_index> view_cell_dofs
      = dofmap_view.cell_dofs(i);
    const ArrayView<const dolfin::la_index> cell_dofs = this->cell_dofs(i);
    dolfin_assert(view_cell_dofs.size() == cell_dofs.size());

    for (std::size_t j = 0; j < view_cell_dofs.size(); ++j)
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
  _ufc_local_to_local= dofmap._ufc_local_to_local;
  _is_view = dofmap._is_view;
  _global_dimension = dofmap._global_dimension;
  _ufc_offset = dofmap._ufc_offset;
  _multimesh_offset = dofmap._multimesh_offset;
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
std::size_t DofMap::global_dimension() const
{
  return _global_dimension;
}
//-----------------------------------------------------------------------------
std::size_t DofMap::local_dimension(std::string type) const
{
  deprecation("DofMap::local_dimension", "2016.1",
              "Please use dofmap::index_map()->size() instead.");

  if (type == "owned")
    return _index_map->size(IndexMap::MapSize::OWNED);
  else if (type == "unowned")
    return _index_map->size(IndexMap::MapSize::UNOWNED);
  else if (type == "all")
    return _index_map->size(IndexMap::MapSize::ALL);
  else
  {
    dolfin_error("DofMap.cpp",
                 "get local dimension",
                 "Unknown type %s", type.c_str());
  }

  return 0;
}
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
std::size_t DofMap::num_entity_dofs(std::size_t dim) const
{
  dolfin_assert(_ufc_dofmap);
  return _ufc_dofmap->num_entity_dofs(dim);
}
//-----------------------------------------------------------------------------
std::size_t DofMap::num_facet_dofs() const
{
  dolfin_assert(_ufc_dofmap);
  return _ufc_dofmap->num_facet_dofs();
}
//-----------------------------------------------------------------------------
std::pair<std::size_t, std::size_t> DofMap::ownership_range() const
{
  return _index_map->local_range();
}
//-----------------------------------------------------------------------------
const std::unordered_map<int, std::vector<int>>& DofMap::shared_nodes() const
{
  return _shared_nodes;
}
//-----------------------------------------------------------------------------
const std::set<int>& DofMap::neighbours() const
{
  return _neighbours;
}
//-----------------------------------------------------------------------------
void
DofMap::tabulate_entity_dofs(std::vector<std::size_t>& dofs,
                             std::size_t dim, std::size_t local_entity) const
{
  dolfin_assert(_ufc_dofmap);
  if (_ufc_dofmap->num_entity_dofs(dim)==0)
    return;

  dofs.resize(_ufc_dofmap->num_entity_dofs(dim));
  _ufc_dofmap->tabulate_entity_dofs(&dofs[0], dim, local_entity);
}
//-----------------------------------------------------------------------------
void DofMap::tabulate_facet_dofs(std::vector<std::size_t>& dofs,
                                 std::size_t local_facet) const
{
  dolfin_assert(_ufc_dofmap);
  dofs.resize(_ufc_dofmap->num_facet_dofs());
  _ufc_dofmap->tabulate_facet_dofs(dofs.data(), local_facet);
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
  DofMap::collapse(std::unordered_map<std::size_t, std::size_t>&
                   collapsed_map,
                   const Mesh& mesh) const
{
  return std::shared_ptr<GenericDofMap>(new DofMap(collapsed_map,
                                                     *this, mesh));
}
//-----------------------------------------------------------------------------
std::vector<dolfin::la_index> DofMap::dofs(const Mesh& mesh,
                                           std::size_t dim) const
{
  // FIXME: This function requires a special case when dim ==
  // mesh.topology().dim() because of they way DOLFIN handles d-d
  // connectivity. Change DOLFIN behaviour.

  // Check number of dofs per entity (on cell cell)
  const std::size_t num_dofs_per_entity = num_entity_dofs(dim);

  // Return empty vector if not dofs on requested entity
  if (num_dofs_per_entity == 0)
    return std::vector<dolfin::la_index>();

  // Vector to hold list of dofs
  std::vector<dolfin::la_index>
    dof_list(mesh.num_entities(dim)*num_dofs_per_entity);

  // Iterate over cells
  if (dim < mesh.topology().dim())
  {
    std::vector<std::size_t> entity_dofs_local;
    for (CellIterator c(mesh); !c.end(); ++c)
    {
      // Get local-to-global dofmap for cell
      const auto cell_dof_list = cell_dofs(c->index());

      // Loop over all entities of dimension dim
      for (MeshEntityIterator e(*c, dim); !e.end(); ++e)
      {
        // Tabulate cell-wise index of all dofs on entity
        const std::size_t local_index = e.pos();
        tabulate_entity_dofs(entity_dofs_local, dim, local_index);

        // Get dof index and add to list
        for (std::size_t i = 0; i < entity_dofs_local.size(); ++i)
        {
          const std::size_t entity_dof_local = entity_dofs_local[i];
          const dolfin::la_index dof_index = cell_dof_list[entity_dof_local];
          dolfin_assert(e->index()*num_dofs_per_entity + i < dof_list.size());
          dof_list[e->index()*num_dofs_per_entity + i] = dof_index;
        }
      }
    }
  }
  else
  {
    std::vector<std::size_t> entity_dofs_local;
    for (CellIterator c(mesh); !c.end(); ++c)
    {
      // Get local-to-global dofmap for cell
      const auto cell_dof_list = cell_dofs(c->index());

      // Tabulate cell-wise index of all dofs on entity
      tabulate_entity_dofs(entity_dofs_local, dim, 0);

      // Get dof index and add to list
      for (std::size_t i = 0; i < entity_dofs_local.size(); ++i)
      {
        const std::size_t entity_dof_local = entity_dofs_local[i];
        const dolfin::la_index dof_index = cell_dof_list[entity_dof_local];
        dolfin_assert(c->index()*num_dofs_per_entity + i < dof_list.size());
        dof_list[c->index()*num_dofs_per_entity + i] = dof_index;
      }
    }
  }

  return dof_list;
}
//-----------------------------------------------------------------------------
std::vector<dolfin::la_index> DofMap::dofs() const
{
  // Create vector to hold dofs
  std::vector<la_index> _dofs;
  _dofs.reserve(_dofmap.size()*max_element_dofs());

  const dolfin::la_index local_ownership_size
    = _index_map->size(IndexMap::MapSize::OWNED);
  const std::size_t global_offset = _index_map->local_range().first;

  // Insert all dofs into a vector (will contain duplicates)
  for (auto dof : _dofmap)
  {
    if (dof >= 0 && dof < local_ownership_size)
      _dofs.push_back(dof + global_offset);
  }

  // Sort dofs (required to later remove duplicates)
  std::sort(_dofs.begin(), _dofs.end());

  // Remove duplicates
  _dofs.erase(std::unique(_dofs.begin(), _dofs.end() ), _dofs.end());

  return _dofs;
}
//-----------------------------------------------------------------------------
void DofMap::set(GenericVector& x, double value) const
{
  dolfin_assert(_dofmap.size() % _cell_dimension == 0);
  const std::size_t num_cells = _dofmap.size() / _cell_dimension;

  std::vector<double> _value(_cell_dimension, value);
  for (std::size_t i = 0; i < num_cells; ++i)
  {
    const ArrayView<const la_index> dofs = cell_dofs(i);
    x.set_local(_value.data(), dofs.size(), dofs.data());
  }
  x.apply("insert");
}
//-----------------------------------------------------------------------------
void DofMap::tabulate_local_to_global_dofs(std::vector<std::size_t>& local_to_global_map) const
{
  const std::size_t bs = _index_map->block_size();
  const std::vector<std::size_t>& local_to_global_unowned
    = _index_map->local_to_global_unowned();
  const std::size_t local_ownership_size
    = _index_map->size(IndexMap::MapSize::OWNED);
  local_to_global_map.resize(_index_map->size(IndexMap::MapSize::ALL));

  const std::size_t global_offset = _index_map->local_range().first;
  for (std::size_t i = 0; i < local_ownership_size; ++i)
    local_to_global_map[i] = i + global_offset;

  for (std::size_t node = 0;
       node < _index_map->local_to_global_unowned().size(); ++node)
  {
    for (std::size_t component = 0; component < bs; ++component)
    {
      local_to_global_map[bs*node + component + local_ownership_size]
        = bs*local_to_global_unowned[node] + component;
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
      dolfin_error("DofMap.cpp",
                   "initialize mapping of degrees of freedom",
                   "Missing entities of dimension %d. Try calling mesh.init(%d)", d, d);
    }
  }
}
//-----------------------------------------------------------------------------
std::string DofMap::str(bool verbose) const
{
  std::stringstream s;
  s << "<DofMap of global dimension " << global_dimension()
    << ">" << std::endl;
  if (verbose)
  {
    // Cell loop
    dolfin_assert(_dofmap.size()%_cell_dimension == 0);
    const std::size_t ncells = _dofmap.size() / _cell_dimension;

    for (std::size_t i = 0; i < ncells; ++i)
    {
      s << "Local cell index, cell dofmap dimension: " << i
        << ", " << _cell_dimension << std::endl;

      // Local dof loop
      for (std::size_t j = 0; j < _cell_dimension; ++j)
      {
        s <<  "  " << "Local, global dof indices: " << j
          << ", " << _dofmap[i*_cell_dimension + j] << std::endl;
      }
    }
  }

  return s.str();
}
//-----------------------------------------------------------------------------
void DofMap::check_dimensional_consistency(const ufc::dofmap& dofmap,
                                            const Mesh& mesh)
{
  // Check topological dimension
  if (dofmap.topological_dimension() != mesh.topology().dim())
  {
    dolfin_error("DofMap.cpp",
                 "create mapping of degrees of freedom",
                 "Topological dimension of the UFC dofmap (dim = %d) and the mesh (dim = %d) do not match",
                 dofmap.topological_dimension(),
                 mesh.topology().dim());
  }
}
//-----------------------------------------------------------------------------
