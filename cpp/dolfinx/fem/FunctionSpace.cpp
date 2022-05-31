// Copyright (C) 2008-2020 Anders Logg and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "FunctionSpace.h"
#include "CoordinateElement.h"
#include "DofMap.h"
#include "FiniteElement.h"
#include <boost/uuid/uuid_generators.hpp>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/Topology.h>
#include <vector>
#include <xtensor/xadapt.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>

using namespace dolfinx;
using namespace dolfinx::fem;

//-----------------------------------------------------------------------------
FunctionSpace::FunctionSpace(std::shared_ptr<const mesh::Mesh> mesh,
                             std::shared_ptr<const FiniteElement> element,
                             std::shared_ptr<const DofMap> dofmap)
    : _mesh(mesh), _element(element), _dofmap(dofmap),
      _id(boost::uuids::random_generator()()), _root_space_id(_id)
{
  // Do nothing
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
  std::shared_ptr<const FiniteElement> element
      = this->_element->extract_sub_element(component);

  // Extract sub dofmap
  auto dofmap
      = std::make_shared<DofMap>(_dofmap->extract_sub_dofmap(component));

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
bool FunctionSpace::contains(const FunctionSpace& V) const
{
  if (this == std::addressof(V))
  {
    // Spaces are the same (same memory address)
    return true;
  }
  else if (_root_space_id != V._root_space_id)
  {
    // Different root spaces
    return false;
  }
  else if (_component.size() > V._component.size())
  {
    // V is a superspace of *this
    return false;
  }
  else if (!std::equal(_component.begin(), _component.end(),
                       V._component.begin()))
  {
    // Components of 'this' are not the same as the leading components
    // of V
    return false;
  }
  else
  {
    // Ok, V is really our subspace
    return true;
  }
}
//-----------------------------------------------------------------------------
std::pair<FunctionSpace, std::vector<std::int32_t>>
FunctionSpace::collapse() const
{
  if (_component.empty())
    throw std::runtime_error("Function space is not a subspace");

  // Create collapsed DofMap
  auto [_collapsed_dofmap, collapsed_dofs]
      = _dofmap->collapse(_mesh->comm(), _mesh->topology());
  auto collapsed_dofmap
      = std::make_shared<DofMap>(std::move(_collapsed_dofmap));

  // Create new FunctionSpace and return
  return {FunctionSpace(_mesh, _element, collapsed_dofmap),
          std::move(collapsed_dofs)};
}
//-----------------------------------------------------------------------------
std::vector<int> FunctionSpace::component() const { return _component; }
//-----------------------------------------------------------------------------
std::vector<double>
FunctionSpace::tabulate_dof_coordinates(bool transpose) const
{
  if (!_component.empty())
  {
    throw std::runtime_error(
        "Cannot tabulate coordinates for a FunctionSpace that is a subspace.");
  }

  assert(_element);
  if (_element->is_mixed())
  {
    throw std::runtime_error(
        "Cannot tabulate coordinates for a mixed FunctionSpace.");
  }

  // Geometric dimension
  assert(_mesh);
  assert(_element);
  const std::size_t gdim = _mesh->geometry().dim();
  const int tdim = _mesh->topology().dim();

  // Get dofmap local size
  assert(_dofmap);
  std::shared_ptr<const common::IndexMap> index_map = _dofmap->index_map;
  assert(index_map);
  const int index_map_bs = _dofmap->index_map_bs();
  const int dofmap_bs = _dofmap->bs();

  const int element_block_size = _element->block_size();
  const std::size_t scalar_dofs
      = _element->space_dimension() / element_block_size;
  const std::int32_t num_dofs
      = index_map_bs * (index_map->size_local() + index_map->num_ghosts())
        / dofmap_bs;

  // Get the dof coordinates on the reference element
  if (!_element->interpolation_ident())
  {
    throw std::runtime_error("Cannot evaluate dof coordinates - this element "
                             "does not have pointwise evaluation.");
  }
  const xt::xtensor<double, 2>& X = _element->interpolation_points();

  // Get coordinate map
  const CoordinateElement& cmap = _mesh->geometry().cmap();

  // Prepare cell geometry
  const graph::AdjacencyList<std::int32_t>& x_dofmap
      = _mesh->geometry().dofmap();
  const std::size_t num_dofs_g = _mesh->geometry().cmap().dim();
  xtl::span<const double> x_g = _mesh->geometry().x();

  // Array to hold coordinates to return
  const std::size_t shape_c0 = transpose ? 3 : num_dofs;
  const std::size_t shape_c1 = transpose ? num_dofs : 3;
  std::vector<double> coords(shape_c0 * shape_c1, 0);

  // Loop over cells and tabulate dofs
  xt::xtensor<double, 2> x = xt::zeros<double>({scalar_dofs, gdim});
  xt::xtensor<double, 2> coordinate_dofs({num_dofs_g, gdim});

  auto map = _mesh->topology().index_map(tdim);
  assert(map);
  const int num_cells = map->size_local() + map->num_ghosts();

  xtl::span<const std::uint32_t> cell_info;
  if (_element->needs_dof_transformations())
  {
    _mesh->topology_mutable().create_entity_permutations();
    cell_info = xtl::span(_mesh->topology().get_cell_permutation_info());
  }

  const std::function<void(const xtl::span<double>&,
                           const xtl::span<const std::uint32_t>&, std::int32_t,
                           int)>
      apply_dof_transformation
      = _element->get_dof_transformation_function<double>();

  const xt::xtensor<double, 2> phi
      = xt::view(cmap.tabulate(0, X), 0, xt::all(), xt::all(), 0);

  for (int c = 0; c < num_cells; ++c)
  {
    // Extract cell geometry
    auto x_dofs = x_dofmap.links(c);
    for (std::size_t i = 0; i < x_dofs.size(); ++i)
    {
      std::copy_n(std::next(x_g.begin(), 3 * x_dofs[i]), gdim,
                  std::next(coordinate_dofs.begin(), i * gdim));
    }

    // Tabulate dof coordinates on cell
    cmap.push_forward(x, coordinate_dofs, phi);
    apply_dof_transformation(xtl::span(x.data(), x.size()),
                             xtl::span(cell_info.data(), cell_info.size()), c,
                             x.shape(1));

    // Get cell dofmap
    auto dofs = _dofmap->cell_dofs(c);

    // Copy dof coordinates into vector
    if (!transpose)
    {
      for (std::size_t i = 0; i < dofs.size(); ++i)
        for (std::size_t j = 0; j < gdim; ++j)
          coords[dofs[i] * 3 + j] = x(i, j);
    }
    else
    {
      for (std::size_t i = 0; i < dofs.size(); ++i)
        for (std::size_t j = 0; j < gdim; ++j)
          coords[j * num_dofs + dofs[i]] = x(i, j);
    }
  }

  return coords;
}
//-----------------------------------------------------------------------------
std::shared_ptr<const mesh::Mesh> FunctionSpace::mesh() const { return _mesh; }
//-----------------------------------------------------------------------------
std::shared_ptr<const FiniteElement> FunctionSpace::element() const
{
  return _element;
}
//-----------------------------------------------------------------------------
std::shared_ptr<const DofMap> FunctionSpace::dofmap() const { return _dofmap; }
//-----------------------------------------------------------------------------
std::array<std::vector<std::shared_ptr<const FunctionSpace>>, 2>
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
