// Copyright (C) 2008-2020 Anders Logg and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "FunctionSpace.h"
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/UniqueIdGenerator.h>
#include <dolfinx/fem/CoordinateElement.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/FiniteElement.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/Topology.h>
#include <vector>

using namespace dolfinx;
using namespace dolfinx::fem;

namespace
{
Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>
internal_tabulate_dof_coordinates(
    std::shared_ptr<const mesh::Mesh> mesh,
    std::shared_ptr<const fem::FiniteElement> element,
    std::shared_ptr<const fem::DofMap> dofmap)
{
  // This function tabulates the DOF coordinates, with each coordinated
  // repeated the given number of times

  // Geometric dimension
  assert(mesh);
  assert(element);
  const int gdim = mesh->geometry().dim();
  const int tdim = mesh->topology().dim();

  // Get dofmap local size
  assert(dofmap);
  std::shared_ptr<const common::IndexMap> index_map = dofmap->index_map;
  const int index_map_bs = dofmap->index_map_bs();
  const int dofmap_bs = dofmap->bs();
  assert(index_map);

  const int element_block_size = element->block_size();
  const int scalar_dofs = element->space_dimension() / element_block_size;
  const std::int32_t num_dofs
      = index_map_bs * (index_map->size_local() + index_map->num_ghosts())
        / dofmap_bs;

  // Get the dof coordinates on the reference element
  if (!element->interpolation_ident())
  {
    throw std::runtime_error("Cannot evaluate dof coordinates - this element "
                             "does not have pointwise evaluation.");
  }
  const common::array2d<double> X = element->interpolation_points();

  // Get coordinate map
  const fem::CoordinateElement& cmap = mesh->geometry().cmap();

  // Prepare cell geometry
  const graph::AdjacencyList<std::int32_t>& x_dofmap
      = mesh->geometry().dofmap();
  // FIXME: Add proper interface for num coordinate dofs
  const int num_dofs_g = x_dofmap.num_links(0);

  // FIXME: Use eigen map for now.
  Eigen::Map<const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                Eigen::RowMajor>>
      x_g(mesh->geometry().x().data(), mesh->geometry().x().shape[0],
          mesh->geometry().x().shape[1]);

  // Array to hold coordinates to return
  Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor> coords
      = Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>::Zero(num_dofs,
                                                                       3);

  // Loop over cells and tabulate dofs
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> x(
      scalar_dofs, gdim);
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      coordinate_dofs(num_dofs_g, gdim);

  auto map = mesh->topology().index_map(tdim);
  assert(map);
  const int num_cells = map->size_local() + map->num_ghosts();

  for (int c = 0; c < num_cells; ++c)
  {
    // Extract cell geometry
    auto x_dofs = x_dofmap.links(c);
    for (int i = 0; i < num_dofs_g; ++i)
      coordinate_dofs.row(i) = x_g.row(x_dofs[i]).head(gdim);

    // Tabulate dof coordinates on cell
    cmap.push_forward(x, X, coordinate_dofs);

    // Get cell dofmap
    auto dofs = dofmap->cell_dofs(c);

    // Copy dof coordinates into vector
    for (std::size_t i = 0; i < dofs.size(); ++i)
      coords.row(dofs[i]).head(gdim) = x.row(i);
  }

  return coords;
}
} // namespace

//-----------------------------------------------------------------------------
FunctionSpace::FunctionSpace(std::shared_ptr<const mesh::Mesh> mesh,
                             std::shared_ptr<const fem::FiniteElement> element,
                             std::shared_ptr<const fem::DofMap> dofmap)
    : _mesh(mesh), _element(element), _dofmap(dofmap),
      _id(common::UniqueIdGenerator::id()), _root_space_id(_id)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
bool FunctionSpace::operator==(const FunctionSpace& V) const
{
  return _element == V._element and _mesh == V._mesh and _dofmap == V._dofmap;
}
//-----------------------------------------------------------------------------
bool FunctionSpace::operator!=(const FunctionSpace& V) const
{
  return !(*this == V);
}
//-----------------------------------------------------------------------------
std::shared_ptr<FunctionSpace>
FunctionSpace::sub(const std::vector<int>& component) const
{
  assert(_mesh);
  assert(_element);
  assert(_dofmap);

  // Check if sub space is already in the cache and not expired
  if (auto it = _subspaces.find(component); it != _subspaces.end())
  {
    if (auto s = it->second.lock())
      return s;
  }

  // Extract sub-element
  std::shared_ptr<const fem::FiniteElement> element
      = this->_element->extract_sub_element(component);

  // Extract sub dofmap
  auto dofmap
      = std::make_shared<fem::DofMap>(_dofmap->extract_sub_dofmap(component));

  // Create new sub space
  auto sub_space = std::make_shared<FunctionSpace>(_mesh, element, dofmap);

  // Set root space id and component w.r.t. root
  sub_space->_root_space_id = _root_space_id;
  sub_space->_component = _component;
  sub_space->_component.insert(sub_space->_component.end(), component.begin(),
                               component.end());

  // Insert new subspace into cache
  _subspaces.emplace(sub_space->_component, sub_space);

  return sub_space;
}
//-----------------------------------------------------------------------------
std::pair<std::shared_ptr<FunctionSpace>, std::vector<std::int32_t>>
FunctionSpace::collapse() const
{
  if (_component.empty())
    throw std::runtime_error("Function space is not a subspace");

  // Create collapsed DofMap
  std::shared_ptr<fem::DofMap> collapsed_dofmap;
  std::vector<std::int32_t> collapsed_dofs;
  std::tie(collapsed_dofmap, collapsed_dofs)
      = _dofmap->collapse(_mesh->mpi_comm(), _mesh->topology());

  // Create new FunctionSpace and return
  auto collapsed_sub_space
      = std::make_shared<FunctionSpace>(_mesh, _element, collapsed_dofmap);

  return {std::move(collapsed_sub_space), std::move(collapsed_dofs)};
}
//-----------------------------------------------------------------------------
std::vector<int> FunctionSpace::component() const { return _component; }
//-----------------------------------------------------------------------------
Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>
FunctionSpace::tabulate_dof_coordinates() const
{
  if (!_component.empty())
  {
    throw std::runtime_error(
        "Cannot tabulate coordinates for a FunctionSpace that is a subspace.");
  }

  return internal_tabulate_dof_coordinates(_mesh, _element, _dofmap);
}
//-----------------------------------------------------------------------------
Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>
FunctionSpace::tabulate_scalar_subspace_dof_coordinates() const
{
  if (!_component.empty())
  {
    throw std::runtime_error(
        "Cannot tabulate coordinates for a FunctionSpace that is a subspace.");
  }

  return internal_tabulate_dof_coordinates(_mesh, _element, _dofmap);
}
//-----------------------------------------------------------------------------
std::size_t FunctionSpace::id() const { return _id; }
//-----------------------------------------------------------------------------
std::shared_ptr<const mesh::Mesh> FunctionSpace::mesh() const { return _mesh; }
//-----------------------------------------------------------------------------
std::shared_ptr<const fem::FiniteElement> FunctionSpace::element() const
{
  return _element;
}
//-----------------------------------------------------------------------------
std::shared_ptr<const fem::DofMap> FunctionSpace::dofmap() const
{
  return _dofmap;
}
//-----------------------------------------------------------------------------
bool FunctionSpace::contains(const FunctionSpace& V) const
{
  // Is the root space same?
  if (_root_space_id != V._root_space_id)
    return false;

  // Is V possibly our superspace?
  if (_component.size() > V._component.size())
    return false;

  // Are our components same as leading components of V?
  if (!std::equal(_component.begin(), _component.end(), V._component.begin()))
    return false;

  // Ok, V is really our subspace
  return true;
}
//-----------------------------------------------------------------------------
std::array<std::vector<std::shared_ptr<const fem::FunctionSpace>>, 2>
fem::common_function_spaces(
    const std::vector<
        std::vector<std::array<std::shared_ptr<const FunctionSpace>, 2>>>& V)
{
  assert(!V.empty());
  std::vector<std::shared_ptr<const FunctionSpace>> spaces0(V.size(), nullptr);
  std::vector<std::shared_ptr<const FunctionSpace>> spaces1(V.front().size(),
                                                            nullptr);

  // Loop over rows
  for (std::size_t i = 0; i < V.size(); ++i)
  {
    // Loop over columns
    for (std::size_t j = 0; j < V[i].size(); ++j)
    {
      auto& V0 = V[i][j][0];
      auto& V1 = V[i][j][1];
      if (V0 and V1)
      {
        if (!spaces0[i])
          spaces0[i] = V0;
        else
        {
          if (spaces0[i] != V0)
            throw std::runtime_error("Mismatched test space for row.");
        }

        if (!spaces1[j])
          spaces1[j] = V1;
        else
        {
          if (spaces1[j] != V1)
            throw std::runtime_error("Mismatched trial space for column.");
        }
      }
    }
  }

  // Check there are no null entries
  if (std::find(spaces0.begin(), spaces0.end(), nullptr) != spaces0.end())
    throw std::runtime_error("Could not deduce all block test spaces.");
  if (std::find(spaces1.begin(), spaces1.end(), nullptr) != spaces1.end())
    throw std::runtime_error("Could not deduce all block trial spaces.");

  return {spaces0, spaces1};
}
//-----------------------------------------------------------------------------
