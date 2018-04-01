// Copyright (C) 2007-2016 Anders Logg and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "DofMap.h"
#include "DofMapBuilder.h"
#include <dolfin/common/MPI.h>
#include <dolfin/common/types.h>
#include <dolfin/la/PETScVector.h>
#include <dolfin/log/LogStream.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/MeshIterator.h>
#include <dolfin/mesh/PeriodicBoundaryComputation.h>
#include <dolfin/mesh/Vertex.h>
#include <unordered_map>

using namespace dolfin;
using namespace dolfin::fem;

//-----------------------------------------------------------------------------
DofMap::DofMap(std::shared_ptr<const ufc::dofmap> ufc_dofmap,
               const mesh::Mesh& mesh)
    : _cell_dimension(0), _ufc_dofmap(ufc_dofmap), _is_view(false),
      _global_dimension(0), _ufc_offset(0)
{
  assert(_ufc_dofmap);

  // Call dofmap builder
  DofMapBuilder::build(*this, mesh, std::shared_ptr<const mesh::SubDomain>());
}
//-----------------------------------------------------------------------------
DofMap::DofMap(std::shared_ptr<const ufc::dofmap> ufc_dofmap,
               const mesh::Mesh& mesh,
               std::shared_ptr<const mesh::SubDomain> constrained_domain)
    : _cell_dimension(0), _ufc_dofmap(ufc_dofmap), _is_view(false),
      _global_dimension(0), _ufc_offset(0)
{
  assert(_ufc_dofmap);

  // Store constrained domain in base class
  this->constrained_domain = constrained_domain;

  // Call dofmap builder
  DofMapBuilder::build(*this, mesh, constrained_domain);
}
//-----------------------------------------------------------------------------
DofMap::DofMap(const DofMap& parent_dofmap,
               const std::vector<std::size_t>& component,
               const mesh::Mesh& mesh)
    : _cell_dimension(0), _ufc_dofmap(0), _is_view(true), _global_dimension(0),
      _ufc_offset(0), _index_map(parent_dofmap._index_map)
{
  // Build sub-dofmap
  DofMapBuilder::build_sub_map_view(*this, parent_dofmap, component, mesh);
}
//-----------------------------------------------------------------------------
DofMap::DofMap(std::unordered_map<std::size_t, std::size_t>& collapsed_map,
               const DofMap& dofmap_view, const mesh::Mesh& mesh)
    : _cell_dimension(0), _ufc_dofmap(dofmap_view._ufc_dofmap), _is_view(false),
      _global_dimension(0), _ufc_offset(0)
{
  assert(_ufc_dofmap);

  // Check that mesh has been ordered
  if (!mesh.ordered())
  {
    log::dolfin_error(
        "DofMap.cpp", "create mapping of degrees of freedom",
        "mesh::Mesh is not ordered according to the UFC numbering convention. "
        "Consider calling mesh.order()");
  }

  // Check dimensional consistency between UFC dofmap and the mesh
  check_provided_entities(*_ufc_dofmap, mesh);

  // Build new dof map
  DofMapBuilder::build(*this, mesh, constrained_domain);

  // Dimension sanity checks
  assert(dofmap_view._dofmap.size()
                == mesh.num_cells() * dofmap_view._cell_dimension);
  assert(global_dimension() == dofmap_view.global_dimension());
  assert(_dofmap.size() == mesh.num_cells() * _cell_dimension);

  // FIXME: Could we use a std::vector instead of std::map if the
  //        collapsed dof map is contiguous (0, . . . , n)?

  // Build map from collapsed dof index to original dof index
  collapsed_map.clear();
  for (std::int64_t i = 0; i < mesh.num_cells(); ++i)
  {
    auto view_cell_dofs = dofmap_view.cell_dofs(i);
    auto cell_dofs = this->cell_dofs(i);
    assert(view_cell_dofs.size() == cell_dofs.size());

    for (Eigen::Index j = 0; j < view_cell_dofs.size(); ++j)
      collapsed_map[cell_dofs[j]] = view_cell_dofs[j];
  }
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
  assert(_ufc_dofmap);
  return _ufc_dofmap->num_element_dofs();
}
//-----------------------------------------------------------------------------
std::size_t DofMap::num_entity_dofs(std::size_t entity_dim) const
{
  assert(_ufc_dofmap);
  return _ufc_dofmap->num_entity_dofs(entity_dim);
}
//-----------------------------------------------------------------------------
std::size_t DofMap::num_facet_dofs() const
{
  assert(_ufc_dofmap);
  return _ufc_dofmap->num_facet_dofs();
}
//-----------------------------------------------------------------------------
std::array<std::int64_t, 2> DofMap::ownership_range() const
{
  assert(_index_map);
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
void DofMap::tabulate_facet_dofs(std::vector<std::size_t>& element_dofs,
                                 std::size_t cell_facet_index) const
{
  assert(_ufc_dofmap);
  element_dofs.resize(_ufc_dofmap->num_facet_dofs());
  _ufc_dofmap->tabulate_facet_dofs(element_dofs.data(), cell_facet_index);
}
//-----------------------------------------------------------------------------
void DofMap::tabulate_entity_dofs(std::vector<std::size_t>& element_dofs,
                                  std::size_t entity_dim,
                                  std::size_t cell_entity_index) const
{
  assert(_ufc_dofmap);
  if (_ufc_dofmap->num_entity_dofs(entity_dim) == 0)
    return;

  element_dofs.resize(_ufc_dofmap->num_entity_dofs(entity_dim));
  _ufc_dofmap->tabulate_entity_dofs(&element_dofs[0], entity_dim,
                                    cell_entity_index);
}
//-----------------------------------------------------------------------------
std::shared_ptr<GenericDofMap>
DofMap::extract_sub_dofmap(const std::vector<std::size_t>& component,
                           const mesh::Mesh& mesh) const
{
  return std::shared_ptr<GenericDofMap>(new DofMap(*this, component, mesh));
}
//-----------------------------------------------------------------------------
std::shared_ptr<GenericDofMap>
DofMap::collapse(std::unordered_map<std::size_t, std::size_t>& collapsed_map,
                 const mesh::Mesh& mesh) const
{
  return std::shared_ptr<GenericDofMap>(new DofMap(collapsed_map, *this, mesh));
}
//-----------------------------------------------------------------------------
void DofMap::set(la::PETScVector& x, double value) const
{
  assert(_dofmap.size() % _cell_dimension == 0);
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
void DofMap::check_provided_entities(const ufc::dofmap& dofmap,
                                     const mesh::Mesh& mesh)
{
  // Check that we have all mesh entities
  for (std::size_t d = 0; d <= mesh.topology().dim(); ++d)
  {
    if (dofmap.num_entity_dofs(d) > 0 && mesh.num_entities(d) == 0)
    {
      log::dolfin_error(
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
    assert(_dofmap.size() % _cell_dimension == 0);
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
