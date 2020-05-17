// Copyright (C) 2008-2019 Anders Logg and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "FunctionSpace.h"
#include "Function.h"
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/UniqueIdGenerator.h>
#include <dolfinx/common/types.h>
#include <dolfinx/common/utils.h>
#include <dolfinx/fem/CoordinateElement.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/FiniteElement.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/Topology.h>
#include <vector>

using namespace dolfinx;
using namespace dolfinx::function;

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
  // Compare pointers to shared objects
  return _element.get() == V._element.get() and _mesh.get() == V._mesh.get()
         and _dofmap.get() == V._dofmap.get();
}
//-----------------------------------------------------------------------------
bool FunctionSpace::operator!=(const FunctionSpace& V) const
{
  // Compare pointers to shared objects
  return !(*this == V);
}
//-----------------------------------------------------------------------------
std::int64_t FunctionSpace::dim() const
{
  assert(_dofmap);
  assert(_dofmap->element_dof_layout);
  if (_dofmap->element_dof_layout->is_view())
  {
    throw std::runtime_error("FunctionSpace dimension not supported for "
                             "sub-functions");
  }

  assert(_dofmap->index_map);
  return _dofmap->index_map->size_global() * _dofmap->index_map->block_size();
}
//-----------------------------------------------------------------------------
void FunctionSpace::interpolate_from_any(
    Eigen::Ref<Eigen::Array<PetscScalar, Eigen::Dynamic, 1>>
        expansion_coefficients,
    const Function& v) const
{
  assert(v.function_space());
  if (!v.function_space()->has_element(*_element))
  {
    throw std::runtime_error("Restricting finite elements function in "
                             "different elements not supported.");
  }

  assert(_mesh);
  assert(v.function_space()->mesh());
  if (_mesh->id() != v.function_space()->mesh()->id())
  {
    throw std::runtime_error(
        "Interpolation on different meshes not supported (yet).");
  }

  const int tdim = _mesh->topology().dim();

  // Get dofmaps
  assert(v.function_space());
  std::shared_ptr<const fem::DofMap> dofmap_v = v.function_space()->dofmap();
  assert(dofmap_v);
  auto map = _mesh->topology().index_map(tdim);
  assert(map);

  // Iterate over mesh and interpolate on each cell
  assert(_dofmap);
  la::VecReadWrapper v_vector_wrap(v.vector().vec());
  Eigen::Map<const Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> v_array
      = v_vector_wrap.x;
  const int num_cells = map->size_local() + map->num_ghosts();
  for (int c = 0; c < num_cells; ++c)
  {
    auto dofs_v = dofmap_v->cell_dofs(c);
    auto cell_dofs = _dofmap->cell_dofs(c);
    assert(dofs_v.size() == cell_dofs.size());
    for (Eigen::Index i = 0; i < dofs_v.size(); ++i)
      expansion_coefficients[cell_dofs[i]] = v_array[dofs_v[i]];
  }
}
//-----------------------------------------------------------------------------
void FunctionSpace::interpolate(
    Eigen::Ref<Eigen::Array<PetscScalar, Eigen::Dynamic, 1>>
        expansion_coefficients,
    const Function& v) const
{
  assert(_mesh);
  assert(_element);

  // Check that function ranks match
  if (_element->value_rank() != v.value_rank())
  {
    throw std::runtime_error("Cannot interpolate function into function space. "
                             "Rank of function ("
                             + std::to_string(v.value_rank())
                             + ") does not match rank of function space ("
                             + std::to_string(_element->value_rank()) + ")");
  }

  // Check that function dimension match
  for (int i = 0; i < _element->value_rank(); ++i)
  {
    if (_element->value_dimension(i) != v.value_dimension(i))
    {
      throw std::runtime_error(
          "Cannot interpolate function into function space. "
          "Dimension "
          + std::to_string(i) + " of function ("
          + std::to_string(v.value_dimension(i)) + ") does not match dimension "
          + std::to_string(i) + " of function space("
          + std::to_string(_element->value_dimension(i)) + ")");
    }
  }

  interpolate_from_any(expansion_coefficients, v);
}
//-----------------------------------------------------------------------------
void FunctionSpace::interpolate(
    Eigen::Ref<Eigen::Array<PetscScalar, Eigen::Dynamic, 1>> coefficients,
    const std::function<Eigen::Array<PetscScalar, Eigen::Dynamic,
                                     Eigen::Dynamic, Eigen::RowMajor>(
        const Eigen::Ref<const Eigen::Array<double, 3, Eigen::Dynamic,
                                            Eigen::RowMajor>>&)>& f) const
{
  // Evaluate expression at dof points
  const Eigen::Array<double, 3, Eigen::Dynamic, Eigen::RowMajor> x
      = tabulate_dof_coordinates().transpose();
  const Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic,
                     Eigen::RowMajor>
      values = f(x);

  assert(_element);
  std::vector<int> vshape(_element->value_rank(), 1);
  for (std::size_t i = 0; i < vshape.size(); ++i)
    vshape[i] = _element->value_dimension(i);
  const int value_size = std::accumulate(std::begin(vshape), std::end(vshape),
                                         1, std::multiplies<>());

  // Note: pybind11 maps 1D NumPy arrays to column vectors for
  // Eigen::Array<PetscScalar, Eigen::Dynamic,Eigen::Dynamic, Eigen::RowMajor>
  // types, therefore we need to handle vectors as a special case.
  if (values.cols() == 1 and values.rows() != 1)
  {
    if (values.rows() != x.cols())
    {
      throw std::runtime_error("Number of computed values is not equal to the "
                               "number of evaluation points. (1)");
    }
    interpolate(coefficients, values);
  }
  else
  {
    if (values.rows() != value_size)
      throw std::runtime_error("Values shape is incorrect. (2)");

    if (values.cols() != x.cols())
    {
      throw std::runtime_error("Number of computed values is not equal to the "
                               "number of evaluation points. (2)");
    }

    interpolate(coefficients, values.transpose());
  }
}
//-----------------------------------------------------------------------------
void FunctionSpace::interpolate_c(
    Eigen::Ref<Eigen::Array<PetscScalar, Eigen::Dynamic, 1>> coefficients,
    const interpolation_function& f) const
{
  // Build list of points at which to evaluate the Expression
  const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor> x
      = tabulate_dof_coordinates();

  // Evaluate expression at points
  assert(_element);
  std::vector<int> vshape(_element->value_rank(), 1);
  for (std::size_t i = 0; i < vshape.size(); ++i)
    vshape[i] = _element->value_dimension(i);
  const int value_size = std::accumulate(std::begin(vshape), std::end(vshape),
                                         1, std::multiplies<>());
  Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      values(x.rows(), value_size);
  f(values, x);

  interpolate(coefficients, values);
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
  _subspaces.insert(std::pair<std::vector<int>, std::shared_ptr<FunctionSpace>>(
      sub_space->_component, sub_space));

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

  return std::pair(std::move(collapsed_sub_space), std::move(collapsed_dofs));
}
//-----------------------------------------------------------------------------
std::vector<int> FunctionSpace::component() const { return _component; }
//-----------------------------------------------------------------------------
Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>
FunctionSpace::tabulate_dof_coordinates() const
{
  // Geometric dimension
  assert(_mesh);
  assert(_element);
  const int gdim = _mesh->geometry().dim();
  const int tdim = _mesh->topology().dim();

  if (!_component.empty())
  {
    throw std::runtime_error(
        "Cannot tabulate coordinates for a FunctionSpace that is a subspace.");
  }

  // Get local size
  assert(_dofmap);
  std::shared_ptr<const common::IndexMap> index_map = _dofmap->index_map;
  assert(index_map);
  int bs = index_map->block_size();
  std::int32_t local_size
      = bs * (index_map->size_local() + index_map->num_ghosts());

  // Dof coordinate on reference element
  const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& X
      = _element->dof_reference_coordinates();

  // Get coordinate map
  const fem::CoordinateElement& cmap = _mesh->geometry().cmap();

  // Prepare cell geometry
  const graph::AdjacencyList<std::int32_t>& x_dofmap
      = _mesh->geometry().dofmap();

  // FIXME: Add proper interface for num coordinate dofs
  const int num_dofs_g = x_dofmap.num_links(0);
  const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
      x_g
      = _mesh->geometry().x();

  // Array to hold coordinates to return
  Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor> x
      = Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>::Zero(
          local_size, 3);

  // Loop over cells and tabulate dofs
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      coordinates(_element->space_dimension(), gdim);
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      coordinate_dofs(num_dofs_g, gdim);

  auto map = _mesh->topology().index_map(tdim);
  assert(map);
  const int num_cells = map->size_local() + map->num_ghosts();
  for (int c = 0; c < num_cells; ++c)
  {
    // Update cell
    auto x_dofs = x_dofmap.links(c);
    for (int i = 0; i < num_dofs_g; ++i)
      coordinate_dofs.row(i) = x_g.row(x_dofs[i]).head(gdim);

    // Get local-to-global map
    auto dofs = _dofmap->cell_dofs(c);

    // Tabulate dof coordinates on cell
    cmap.push_forward(coordinates, X, coordinate_dofs);

    // Copy dof coordinates into vector
    for (Eigen::Index i = 0; i < dofs.size(); ++i)
    {
      const std::int32_t dof = dofs[i];
      x.row(dof).head(gdim) = coordinates.row(i);
    }
  }

  return x;
}
//-----------------------------------------------------------------------------
void FunctionSpace::set_x(
    Eigen::Ref<Eigen::Array<PetscScalar, Eigen::Dynamic, 1>> x,
    PetscScalar value, int component) const
{
  assert(_mesh);
  assert(_dofmap);
  assert(_element);

  const int gdim = _mesh->geometry().dim();
  const int tdim = _mesh->topology().dim();
  std::vector<PetscScalar> x_values;

  // Prepare cell geometry
  const graph::AdjacencyList<std::int32_t>& x_dofmap
      = _mesh->geometry().dofmap();

  // FIXME: Add proper interface for num coordinate dofs
  const int num_dofs_g = x_dofmap.num_links(0);
  const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
      x_g
      = _mesh->geometry().x();

  // Dof coordinate on reference element
  const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& X
      = _element->dof_reference_coordinates();

  // Get coordinate map
  const fem::CoordinateElement& cmap = _mesh->geometry().cmap();

  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      coordinates(_element->space_dimension(), _mesh->geometry().dim());
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      coordinate_dofs(num_dofs_g, gdim);

  auto map = _mesh->topology().index_map(tdim);
  assert(map);
  const int num_cells = map->size_local() + map->num_ghosts();
  for (int c = 0; c < num_cells; ++c)
  {
    // Update UFC cell
    auto x_dofs = x_dofmap.links(c);
    for (int i = 0; i < num_dofs_g; ++i)
      coordinate_dofs.row(i) = x_g.row(x_dofs[i]).head(gdim);

    // Get cell local-to-global map
    auto dofs = _dofmap->cell_dofs(c);

    // Tabulate dof coordinates
    cmap.push_forward(coordinates, X, coordinate_dofs);

    assert(coordinates.rows() == dofs.size());
    assert(component < (int)coordinates.cols());

    // Copy coordinate (it may be possible to avoid this)
    for (Eigen::Index i = 0; i < coordinates.rows(); ++i)
      x[dofs[i]] = value * coordinates(i, component);
  }
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
  for (std::size_t i = 0; i < _component.size(); ++i)
  {
    if (_component[i] != V._component[i])
      return false;
  }

  // Ok, V is really our subspace
  return true;
}
//-----------------------------------------------------------------------------
void FunctionSpace::interpolate(
    Eigen::Ref<Eigen::Array<PetscScalar, Eigen::Dynamic, 1>> coefficients,
    const Eigen::Ref<const Eigen::Array<PetscScalar, Eigen::Dynamic,
                                        Eigen::Dynamic, Eigen::RowMajor>>&
        values) const
{
  assert(_mesh);
  assert(_element);
  assert(_dofmap);
  const int tdim = _mesh->topology().dim();

  // Note: the following does not exploit any block structure, e.g. for
  // vector Lagrange, which leads to a lot of redundant evaluations.
  // E.g., for a vector Lagrange element the vector-valued expression is
  // evaluted three times at the some point.

  const int value_size = values.cols();

  // FIXME: Dummy coordinate dofs - should limit the interpolation to
  // Lagrange, in which case we don't need coordinate dofs in
  // FiniteElement::transform_values.
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      coordinate_dofs;

  // FIXME: It would be far more elegant and efficient to avoid the need
  // to loop over cells to set the expansion corfficients. Would be much
  // better if the expansion coefficients could be passed straight into
  // Expresion::eval.

  // Loop over cells
  const int ndofs = _element->space_dimension();
  Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      values_cell(ndofs, value_size);
  assert(_dofmap->element_dof_layout);
  std::vector<PetscScalar> cell_coefficients(
      _dofmap->element_dof_layout->num_dofs());

  auto map = _mesh->topology().index_map(tdim);
  assert(map);
  const int num_cells = map->size_local() + map->num_ghosts();
  for (int c = 0; c < num_cells; ++c)
  {
    // Get dofmap for cell
    auto cell_dofs = _dofmap->cell_dofs(c);
    for (Eigen::Index i = 0; i < cell_dofs.rows(); ++i)
    {
      for (Eigen::Index j = 0; j < value_size; ++j)
        values_cell(i, j) = values(cell_dofs[i], j);

      // FIXME: For vector-valued Lagrange, this function 'throws away'
      // the redundant expression evaluations. It should really be made
      // not necessary.
      _element->transform_values(cell_coefficients.data(), values_cell,
                                 coordinate_dofs);

      // Copy into expansion coefficient array
      for (Eigen::Index i = 0; i < cell_dofs.rows(); ++i)
        coefficients[cell_dofs[i]] = cell_coefficients[i];
    }
  }
}
//-----------------------------------------------------------------------------
