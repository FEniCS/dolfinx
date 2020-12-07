// Copyright (C) 2008-2020 Anders Logg and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "FiniteElement.h"
#include <dolfinx/common/log.h>
#include <dolfinx/mesh/utils.h>
#include <functional>
#include <libtab.h>
#include <ufc.h>

using namespace dolfinx;
using namespace dolfinx::fem;

//-----------------------------------------------------------------------------
FiniteElement::FiniteElement(const ufc_finite_element& ufc_element)
    : _signature(ufc_element.signature), _family(ufc_element.family),
      _tdim(ufc_element.topological_dimension),
      _space_dim(ufc_element.space_dimension),
      _value_size(ufc_element.value_size),
      _reference_value_size(ufc_element.reference_value_size),
      _hash(std::hash<std::string>{}(_signature)),
      _transform_reference_basis_derivatives(
          ufc_element.transform_reference_basis_derivatives),
      _transform_values(ufc_element.transform_values),
      _block_size(ufc_element.block_size),
      _interpolate_into_cell(ufc_element.interpolate_into_cell),
      _interpolation_points(ufc_element.num_interpolation_points,
                            ufc_element.topological_dimension),
      _needs_permutation_data(ufc_element.needs_permutation_data)
{
  int n = 0;
  for (int p = 0; p < ufc_element.num_interpolation_points; ++p)
    for (int d = 0; d < ufc_element.topological_dimension; ++d)
      _interpolation_points(p, d) = ufc_element.interpolation_points[n++];

  // Store dof coordinates on reference element if they exist
  assert(ufc_element.tabulate_reference_dof_coordinates);
  _refX.resize(_space_dim, _tdim);
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> X(
      _space_dim, _tdim);
  if (ufc_element.tabulate_reference_dof_coordinates(X.data()) != -1)
    _refX = X;

  // FIXME: this should really be fixed in ffcx.
  _refX.conservativeResize(_space_dim / _block_size, _tdim);

  const ufc_shape _shape = ufc_element.cell_shape;
  switch (_shape)
  {
  case interval:
    _cell_shape = mesh::CellType::interval;
    break;
  case triangle:
    _cell_shape = mesh::CellType::triangle;
    break;
  case quadrilateral:
    _cell_shape = mesh::CellType::quadrilateral;
    break;
  case tetrahedron:
    _cell_shape = mesh::CellType::tetrahedron;
    break;
  case hexahedron:
    _cell_shape = mesh::CellType::hexahedron;
    break;
  default:
    throw std::runtime_error(
        "Unknown UFC cell type when building FiniteElement.");
  }
  assert(mesh::cell_dim(_cell_shape) == _tdim);

  _libtab_element
      = std::make_shared<const libtab::FiniteElement>(libtab::create_element(
          _family, mesh::to_string(_cell_shape), ufc_element.degree));

  // Copy over "dof coordinates" from libtab (only for Lagrange, so far)
  _refX = _libtab_element->points();

  // Fill value dimension
  for (int i = 0; i < ufc_element.value_rank; ++i)
    _value_dimension.push_back(ufc_element.value_dimension(i));

  // Create all sub-elements
  for (int i = 0; i < ufc_element.num_sub_elements; ++i)
  {
    ufc_finite_element* ufc_sub_element = ufc_element.create_sub_element(i);
    _sub_elements.push_back(std::make_shared<FiniteElement>(*ufc_sub_element));
    std::free(ufc_sub_element);
  }
}
//-----------------------------------------------------------------------------
std::string FiniteElement::signature() const { return _signature; }
//-----------------------------------------------------------------------------
mesh::CellType FiniteElement::cell_shape() const { return _cell_shape; }
//-----------------------------------------------------------------------------
int FiniteElement::space_dimension() const { return _space_dim; }
//-----------------------------------------------------------------------------
int FiniteElement::value_size() const { return _value_size; }
//-----------------------------------------------------------------------------
int FiniteElement::reference_value_size() const
{
  return _reference_value_size;
}
//-----------------------------------------------------------------------------
int FiniteElement::value_rank() const { return _value_dimension.size(); }
//-----------------------------------------------------------------------------
int FiniteElement::block_size() const { return _block_size; }
//-----------------------------------------------------------------------------
int FiniteElement::value_dimension(int i) const
{
  if (i >= (int)_value_dimension.size())
    return 1;
  return _value_dimension.at(i);
}
//-----------------------------------------------------------------------------
std::string FiniteElement::family() const { return _family; }
//-----------------------------------------------------------------------------
void FiniteElement::evaluate_reference_basis(
    Eigen::Tensor<double, 3, Eigen::RowMajor>& reference_values,
    const Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                        Eigen::RowMajor>>& X) const
{
  Eigen::ArrayXXd libtab_data = _libtab_element->tabulate(0, X)[0];

  assert(reference_values.dimension(0) == X.rows());
  assert(reference_values.dimension(1)
         == libtab_data.cols() / _reference_value_size);
  assert(reference_values.dimension(2) == _reference_value_size);
  for (int p = 0; p < X.rows(); ++p)
    for (int d = 0; d < libtab_data.cols() / _reference_value_size; ++d)
      for (int v = 0; v < _reference_value_size; ++v)
        reference_values(p, d, v)
            = libtab_data(p, d * _reference_value_size + v);
}
//-----------------------------------------------------------------------------
void FiniteElement::evaluate_reference_basis_derivatives(
    Eigen::Tensor<double, 4, Eigen::RowMajor>& reference_values, int order,
    const Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                        Eigen::RowMajor>>& X) const
{
  assert(order == 1); // TODO: fix this for order > 1

  std::vector<Eigen::ArrayXXd> libtab_data = _libtab_element->tabulate(1, X);
  for (int p = 0; p < X.rows(); ++p)
    for (int d = 0; d < libtab_data[0].cols() / _reference_value_size; ++d)
      for (int v = 0; v < _reference_value_size; ++v)
        for (std::size_t deriv = 0; deriv < libtab_data.size() - 1; ++deriv)
          reference_values(p, d, v, deriv)
              = libtab_data[deriv](p, d * _reference_value_size + v);
}
//-----------------------------------------------------------------------------
void FiniteElement::transform_reference_basis(
    Eigen::Tensor<double, 3, Eigen::RowMajor>& values,
    const Eigen::Tensor<double, 3, Eigen::RowMajor>& reference_values,
    const Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                        Eigen::RowMajor>>& X,
    const Eigen::Tensor<double, 3, Eigen::RowMajor>& J,
    const Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, 1>>& detJ,
    const Eigen::Tensor<double, 3, Eigen::RowMajor>& K,
    const std::uint32_t permutation_info) const
{
  assert(_transform_reference_basis_derivatives);
  const int num_points = X.rows();

  int ret = _transform_reference_basis_derivatives(
      values.data(), 0, num_points, reference_values.data(), X.data(), J.data(),
      detJ.data(), K.data(), permutation_info);
  if (ret == -1)
  {
    throw std::runtime_error("Generated code returned error "
                             "in transform_reference_basis_derivatives");
  }
}
//-----------------------------------------------------------------------------
void FiniteElement::transform_reference_basis_derivatives(
    Eigen::Tensor<double, 4, Eigen::RowMajor>& values, std::size_t order,
    const Eigen::Tensor<double, 4, Eigen::RowMajor>& reference_values,
    const Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                        Eigen::RowMajor>>& X,
    const Eigen::Tensor<double, 3, Eigen::RowMajor>& J,
    const Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, 1>>& detJ,
    const Eigen::Tensor<double, 3, Eigen::RowMajor>& K,
    const std::uint32_t permutation_info) const
{
  assert(_transform_reference_basis_derivatives);
  const int num_points = X.rows();
  int ret = _transform_reference_basis_derivatives(
      values.data(), order, num_points, reference_values.data(), X.data(),
      J.data(), detJ.data(), K.data(), permutation_info);
  if (ret == -1)
  {
    throw std::runtime_error("Generated code returned error "
                             "in transform_reference_basis_derivatives");
  }
}
//-----------------------------------------------------------------------------
const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
FiniteElement::dof_reference_coordinates() const
{
  if (_refX.size() == 0)
  {
    throw std::runtime_error(
        "Dof reference coordinates do not exist for this element.");
  }

  return _refX;
}
//-----------------------------------------------------------------------------
void FiniteElement::transform_values(
    ufc_scalar_t* reference_values,
    const Eigen::Ref<const Eigen::Array<ufc_scalar_t, Eigen::Dynamic,
                                        Eigen::Dynamic, Eigen::RowMajor>>&
        physical_values,
    const Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                        Eigen::RowMajor>>& coordinate_dofs)
    const
{
  assert(_transform_values);
  _transform_values(reference_values, physical_values.data(),
                    coordinate_dofs.data(), nullptr);
}

//-----------------------------------------------------------------------------
int FiniteElement::num_sub_elements() const { return _sub_elements.size(); }
//-----------------------------------------------------------------------------
std::size_t FiniteElement::hash() const { return _hash; }
//-----------------------------------------------------------------------------
std::shared_ptr<const FiniteElement>
FiniteElement::extract_sub_element(const std::vector<int>& component) const
{
  // Recursively extract sub element
  std::shared_ptr<const FiniteElement> sub_finite_element
      = extract_sub_element(*this, component);
  DLOG(INFO) << "Extracted finite element for sub-system: "
             << sub_finite_element->signature().c_str();
  return sub_finite_element;
}
//-----------------------------------------------------------------------------
std::shared_ptr<const FiniteElement>
FiniteElement::extract_sub_element(const FiniteElement& finite_element,
                                   const std::vector<int>& component)
{
  // Check that a sub system has been specified
  if (component.empty())
  {
    throw std::runtime_error(
        "Cannot extract subsystem of finite element. No system was specified");
  }

  // Check if there are any sub systems
  if (finite_element.num_sub_elements() == 0)
  {
    throw std::runtime_error(
        "Cannot extract subsystem of finite element. There are no subsystems.");
  }

  // Check the number of available sub systems
  if (component[0] >= finite_element.num_sub_elements())
  {
    throw std::runtime_error(
        "Cannot extract subsystem of finite element. Requested "
        "subsystem out of range.");
  }

  // Get sub system
  std::shared_ptr<const FiniteElement> sub_element
      = finite_element._sub_elements[component[0]];
  assert(sub_element);

  // Return sub system if sub sub system should not be extracted
  if (component.size() == 1)
    return sub_element;

  // Otherwise, recursively extract the sub sub system
  const std::vector<int> sub_component(component.begin() + 1, component.end());

  return extract_sub_element(*sub_element, sub_component);
}
//-----------------------------------------------------------------------------
Eigen::ArrayXXd FiniteElement::interpolation_points() const
{
  return _interpolation_points;
}
//-----------------------------------------------------------------------------
Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
FiniteElement::interpolate_into_cell(
    const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        values,
    const std::uint32_t cell_permutation) const
{
  Eigen::Array<double, Eigen::Dynamic, 1> output_values(_space_dim / _block_size);
  _interpolate_into_cell(output_values.data(), values.data(), cell_permutation);
  return output_values;
}
//-----------------------------------------------------------------------------
bool FiniteElement::needs_permutation_data() const
{
  return _needs_permutation_data;
}
//-----------------------------------------------------------------------------
