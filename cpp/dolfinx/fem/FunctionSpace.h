// Copyright (C) 2008-2023 Anders Logg and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "CoordinateElement.h"
#include "DofMap.h"
#include "FiniteElement.h"
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/Topology.h>
#include <map>
#include <memory>
#include <vector>

namespace dolfinx::fem
{
/// @brief This class represents a finite element function space defined
/// by a mesh, a finite element, and a local-to-global map of the
/// degrees-of-freedom.
/// @tparam T The floating point (real) type of the mesh geometry and
/// the finite element basis.
template <std::floating_point T>
class FunctionSpace
{
public:
  /// Geometry type of the Mesh that the FunctionSpace is defined on.
  using geometry_type = T;

  /// @brief Create function space for given mesh, element and
  /// degree-of-freedom map.
  /// @param[in] mesh Mesh that the space is defined on.
  /// @param[in] element Finite element for the space.
  /// @param[in] dofmap Degree-of-freedom map for the space.
  FunctionSpace(std::shared_ptr<const mesh::Mesh<geometry_type>> mesh,
                std::shared_ptr<const FiniteElement<geometry_type>> element,
                std::shared_ptr<const DofMap> dofmap)
      : _mesh(mesh), _element(element), _dofmap(dofmap),
        _id(boost::uuids::random_generator()()), _root_space_id(_id)
  {
    // Do nothing
  }

  // Copy constructor (deleted)
  FunctionSpace(const FunctionSpace& V) = delete;

  /// Move constructor
  FunctionSpace(FunctionSpace&& V) = default;

  /// Destructor
  virtual ~FunctionSpace() = default;

  // Assignment operator (delete)
  FunctionSpace& operator=(const FunctionSpace& V) = delete;

  /// Move assignment operator
  FunctionSpace& operator=(FunctionSpace&& V) = default;

  /// @brief Create a subspace (view) for a specific component.
  ///
  /// @note If the subspace is re-used, for performance reasons the
  /// returned subspace should be stored by the caller to avoid repeated
  /// re-computation of the subspace.
  ///
  /// @param[in] component Subspace component.
  /// @return A subspace.
  FunctionSpace sub(const std::vector<int>& component) const
  {
    assert(_mesh);
    assert(_element);
    assert(_dofmap);

    // Check that component is valid
    if (component.empty())
      throw std::runtime_error("Component must be non-empty");

    // Extract sub-element
    auto element = this->_element->extract_sub_element(component);

    // Extract sub dofmap
    auto dofmap
        = std::make_shared<DofMap>(_dofmap->extract_sub_dofmap(component));

    // Create new sub space
    FunctionSpace sub_space(_mesh, element, dofmap);

    // Set root space id and component w.r.t. root
    sub_space._root_space_id = _root_space_id;
    sub_space._component = _component;
    sub_space._component.insert(sub_space._component.end(), component.begin(),
                                component.end());
    return sub_space;
  }

  /// @brief Check whether V is subspace of this, or this itself
  /// @param[in] V The space to be tested for inclusion
  /// @return True if V is contained in or is equal to this
  /// FunctionSpace
  bool contains(const FunctionSpace& V) const
  {
    if (this == std::addressof(V)) // Spaces are the same (same memory address)
      return true;
    else if (_root_space_id != V._root_space_id) // Different root spaces
      return false;
    else if (_component.size()
             > V._component.size()) // V is a superspace of *this
    {
      return false;
    }
    else if (!std::equal(_component.begin(), _component.end(),
                         V._component.begin())) // Components of 'this' are not
                                                // the same as the leading
                                                // components of V
    {
      return false;
    }
    else // Ok, V is really our subspace
      return true;
  }

  /// Collapse a subspace and return a new function space and a map from
  /// new to old dofs
  /// @return The new function space and a map from new to old dofs
  std::pair<FunctionSpace, std::vector<std::int32_t>> collapse() const
  {
    if (_component.empty())
      throw std::runtime_error("Function space is not a subspace");

    // Create collapsed DofMap
    auto [_collapsed_dofmap, collapsed_dofs]
        = _dofmap->collapse(_mesh->comm(), *_mesh->topology());
    auto collapsed_dofmap
        = std::make_shared<DofMap>(std::move(_collapsed_dofmap));

    return {FunctionSpace(_mesh, _element, collapsed_dofmap),
            std::move(collapsed_dofs)};
  }

  /// @brief Get the component with respect to the root superspace.
  /// @return The component with respect to the root superspace, i.e.
  /// `W.sub(1).sub(0) == [1, 0]`.
  std::vector<int> component() const { return _component; }

  /// @brief Indicate whether this function space represents a symmetric
  /// 2-tensor.
  bool symmetric() const
  {
    if (_element)
      return _element->symmetric();
    return false;
  }

  /// @brief Tabulate the physical coordinates of all dofs on this
  /// process.
  ///
  /// @todo Remove - see function in interpolate.h
  ///
  /// @param[in] transpose If false the returned data has shape
  /// `(num_points, 3)`, otherwise it is transposed and has shape `(3,
  /// num_points)`.
  /// @return The dof coordinates `[([x0, y0, z0], [x1, y1, z1], ...)`
  /// if `transpose` is false, and otherwise the returned data is
  /// transposed. Storage is row-major.
  std::vector<geometry_type> tabulate_dof_coordinates(bool transpose) const
  {
    if (!_component.empty())
    {
      throw std::runtime_error("Cannot tabulate coordinates for a "
                               "FunctionSpace that is a subspace.");
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
    const int tdim = _mesh->topology()->dim();

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
    const auto [X, Xshape] = _element->interpolation_points();

    // Get coordinate map
    const CoordinateElement<geometry_type>& cmap = _mesh->geometry().cmap();

    // Prepare cell geometry
    auto x_dofmap = _mesh->geometry().dofmap();
    const std::size_t num_dofs_g = cmap.dim();
    std::span<const geometry_type> x_g = _mesh->geometry().x();

    // Array to hold coordinates to return
    const std::size_t shape_c0 = transpose ? 3 : num_dofs;
    const std::size_t shape_c1 = transpose ? num_dofs : 3;
    std::vector<geometry_type> coords(shape_c0 * shape_c1, 0);

    using mdspan2_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        geometry_type,
        MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>;
    using cmdspan4_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        const geometry_type,
        MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 4>>;

    // Loop over cells and tabulate dofs
    std::vector<geometry_type> x_b(scalar_dofs * gdim);
    mdspan2_t x(x_b.data(), scalar_dofs, gdim);

    std::vector<geometry_type> coordinate_dofs_b(num_dofs_g * gdim);
    mdspan2_t coordinate_dofs(coordinate_dofs_b.data(), num_dofs_g, gdim);

    auto map = _mesh->topology()->index_map(tdim);
    assert(map);
    const int num_cells = map->size_local() + map->num_ghosts();

    std::span<const std::uint32_t> cell_info;
    if (_element->needs_dof_transformations())
    {
      _mesh->topology_mutable()->create_entity_permutations();
      cell_info = std::span(_mesh->topology()->get_cell_permutation_info());
    }

    const std::array<std::size_t, 4> phi_shape
        = cmap.tabulate_shape(0, Xshape[0]);
    std::vector<geometry_type> phi_b(
        std::reduce(phi_shape.begin(), phi_shape.end(), 1, std::multiplies{}));
    cmdspan4_t phi_full(phi_b.data(), phi_shape);
    cmap.tabulate(0, X, Xshape, phi_b);
    auto phi = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
        phi_full, 0, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent,
        MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent, 0);

    // TODO: Check transform
    // Basis function reference-to-conforming transformation function
    auto apply_dof_transformation
        = _element->template dof_transformation_fn<geometry_type>(
            doftransform::standard);

    for (int c = 0; c < num_cells; ++c)
    {
      // Extract cell geometry 'dofs'
      auto x_dofs = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
          x_dofmap, c, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
      for (std::size_t i = 0; i < x_dofs.size(); ++i)
        for (std::size_t j = 0; j < gdim; ++j)
          coordinate_dofs(i, j) = x_g[3 * x_dofs[i] + j];

      // Tabulate dof coordinates on cell
      cmap.push_forward(x, coordinate_dofs, phi);
      apply_dof_transformation(
          x_b, std::span(cell_info.data(), cell_info.size()), c, x.extent(1));

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

  /// The mesh
  std::shared_ptr<const mesh::Mesh<geometry_type>> mesh() const
  {
    return _mesh;
  }

  /// The finite element
  std::shared_ptr<const FiniteElement<geometry_type>> element() const
  {
    return _element;
  }

  /// The dofmap
  std::shared_ptr<const DofMap> dofmap() const { return _dofmap; }

private:
  // The mesh
  std::shared_ptr<const mesh::Mesh<geometry_type>> _mesh;

  // The finite element
  std::shared_ptr<const FiniteElement<geometry_type>> _element;

  // The dofmap
  std::shared_ptr<const DofMap> _dofmap;

  // The component w.r.t. to root space
  std::vector<int> _component;

  // Unique identifier for the space and for its root space
  boost::uuids::uuid _id;
  boost::uuids::uuid _root_space_id;
};

/// @brief Extract FunctionSpaces for (0) rows blocks and (1) columns
/// blocks from a rectangular array of (test, trial) space pairs.
///
/// The test space must be the same for each row and the trial spaces
/// must be the same for each column. Raises an exception if there is an
/// inconsistency. e.g. if each form in row i does not have the same
/// test space then an exception is raised.
///
/// @param[in] V Vector function spaces for (0) each row block and (1)
/// each column block
template <dolfinx::scalar T>
std::array<std::vector<std::shared_ptr<const FunctionSpace<T>>>, 2>
common_function_spaces(
    const std::vector<
        std::vector<std::array<std::shared_ptr<const FunctionSpace<T>>, 2>>>& V)
{
  assert(!V.empty());
  std::vector<std::shared_ptr<const FunctionSpace<T>>> spaces0(V.size(),
                                                               nullptr);
  std::vector<std::shared_ptr<const FunctionSpace<T>>> spaces1(V.front().size(),
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

  // Check that there are no null entries
  if (std::find(spaces0.begin(), spaces0.end(), nullptr) != spaces0.end())
    throw std::runtime_error("Could not deduce all block test spaces.");
  if (std::find(spaces1.begin(), spaces1.end(), nullptr) != spaces1.end())
    throw std::runtime_error("Could not deduce all block trial spaces.");

  return {spaces0, spaces1};
}

/// Type deduction
template <typename U, typename V, typename W>
FunctionSpace(U mesh, V element, W dofmap)
    -> FunctionSpace<typename std::remove_cvref<
        typename U::element_type>::type::geometry_type::value_type>;

} // namespace dolfinx::fem
