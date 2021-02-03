// Copyright (C) 2008-2020 Anders Logg and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "FiniteElement.h"
#include <basix.h>
#include <dolfinx/common/log.h>
#include <dolfinx/mesh/utils.h>
#include <functional>
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
      _apply_dof_transformation(ufc_element.apply_dof_transformation),
      _apply_dof_transformation_to_scalar(
          ufc_element.apply_dof_transformation_to_scalar),
      _bs(ufc_element.block_size),
      _interpolation_is_ident(ufc_element.interpolation_is_identity),
      _needs_permutation_data(ufc_element.needs_permutation_data)
{
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

  static const std::map<ufc_shape, std::string> ufc_to_cell
      = {{vertex, "point"},
         {interval, "interval"},
         {triangle, "triangle"},
         {tetrahedron, "tetrahedron"},
         {quadrilateral, "quadrilateral"},
         {hexahedron, "hexahedron"}};
  const std::string cell_shape = ufc_to_cell.at(ufc_element.cell_shape);

  const std::string family = ufc_element.family;

  // FIXME: Add element 'handle' to UFC and do not use fragile strings
  if (family == "mixed element")
  {
    // basix does not support mixed elements, so the subelements should be
    // handled separately
    // This will cause an error, if actually used
    _basix_element_handle = -1;
  }
  else
  {
    _basix_element_handle = basix::register_element(
        family.c_str(), cell_shape.c_str(), ufc_element.degree);
    _interpolation_matrix = basix::interpolation_matrix(_basix_element_handle);
  }

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
std::string FiniteElement::signature() const noexcept { return _signature; }
//-----------------------------------------------------------------------------
mesh::CellType FiniteElement::cell_shape() const noexcept
{
  return _cell_shape;
}
//-----------------------------------------------------------------------------
int FiniteElement::space_dimension() const noexcept { return _space_dim; }
//-----------------------------------------------------------------------------
int FiniteElement::value_size() const noexcept { return _value_size; }
//-----------------------------------------------------------------------------
int FiniteElement::reference_value_size() const noexcept
{
  return _reference_value_size;
}
//-----------------------------------------------------------------------------
int FiniteElement::value_rank() const noexcept
{
  return _value_dimension.size();
}
//-----------------------------------------------------------------------------
int FiniteElement::block_size() const noexcept { return _bs; }
//-----------------------------------------------------------------------------
int FiniteElement::value_dimension(int i) const
{
  if (i >= (int)_value_dimension.size())
    return 1;
  else
    return _value_dimension.at(i);
}
//-----------------------------------------------------------------------------
std::string FiniteElement::family() const noexcept { return _family; }
//-----------------------------------------------------------------------------
void FiniteElement::evaluate_reference_basis(
    std::vector<double>& reference_values,
    const Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                        Eigen::RowMajor>>& X) const
{
  const Eigen::ArrayXXd basix_data
      = basix::tabulate(_basix_element_handle, 0, X)[0];

  const int scalar_reference_value_size = _reference_value_size / _bs;

  assert(basix_data.cols() % scalar_reference_value_size == 0);
  const int scalar_dofs = basix_data.cols() / scalar_reference_value_size;

  assert((int)reference_values.size()
         == X.rows() * scalar_dofs * scalar_reference_value_size);

  assert(basix_data.rows() == X.rows());

  for (int p = 0; p < X.rows(); ++p)
    for (int d = 0; d < scalar_dofs; ++d)
      for (int v = 0; v < scalar_reference_value_size; ++v)
        reference_values[(p * scalar_dofs + d) * scalar_reference_value_size
                         + v]
            = basix_data(p, d + scalar_dofs * v);
}
//-----------------------------------------------------------------------------
void FiniteElement::evaluate_reference_basis_derivatives(
    std::vector<double>& values, int order,
    const Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                        Eigen::RowMajor>>& X) const
{
  // TODO: fix this for order > 1
  if (order != 1)
  {
    throw std::runtime_error(
        "FiniteElement::evaluate_reference_basis_derivatives only supports "
        "order 1 at the moment.");
  }

  const std::vector<Eigen::ArrayXXd> basix_data
      = basix::tabulate(_basix_element_handle, 1, X);
  for (int p = 0; p < X.rows(); ++p)
    for (int d = 0; d < basix_data[0].cols() / _reference_value_size; ++d)
      for (int v = 0; v < _reference_value_size; ++v)
        for (std::size_t deriv = 0; deriv < basix_data.size() - 1; ++deriv)
          values[(p * basix_data[0].cols() + d * _reference_value_size + v)
                     * (basix_data.size() - 1)
                 + deriv]
              = basix_data[deriv](p, d * _reference_value_size + v);
}
//-----------------------------------------------------------------------------
void FiniteElement::transform_reference_basis(
    std::vector<double>& values, const std::vector<double>& reference_values,
    const Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                        Eigen::RowMajor>>& X,
    const std::vector<double>& J, const tcb::span<const double>& detJ,
    const std::vector<double>& K) const
{
  const int num_points = X.rows();
  const int scalar_dim = _space_dim / _bs;
  const int value_size = _value_size / _bs;
  const int size_per_point = scalar_dim * value_size;
  const int Jsize = J.dimension(1) * J.dimension(2);
  assert(X.cols() == J.dimension(2));
  assert(values.dimension(0) == num_points);
  assert(values.dimension(1) * values.dimension(2) == size_per_point);

  Eigen::Map<const Eigen::ArrayXd> J_unwrapped(J.data(), J.size());
  Eigen::Map<const Eigen::ArrayXd> K_unwrapped(K.data(), K.size());
  Eigen::Map<const Eigen::ArrayXd> reference_values_unwrapped(
      reference_values.data(), reference_values.size());

  Eigen::Map<Eigen::ArrayXd> values_unwrapped(values.data(), values.size());

  for (int pt = 0; pt < num_points; ++pt)
  {
    for (int d = 0; d < scalar_dim; ++d)
    {
      values_unwrapped.block(pt * size_per_point + d * value_size, 0,
                             value_size, 1)
          = basix::map_push_forward(
              _basix_element_handle,
              reference_values_unwrapped.block(
                  pt * size_per_point + d * value_size, 0, value_size, 1),
              Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic,
                                             Eigen::Dynamic, Eigen::RowMajor>>(
                  J_unwrapped.block(Jsize * pt, 0, Jsize, 1).data(),
                  J.dimension(1), J.dimension(2)),
              detJ[pt],
              Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic,
                                             Eigen::Dynamic, Eigen::RowMajor>>(
                  K_unwrapped.block(Jsize * pt, 0, Jsize, 1).data(),
                  J.dimension(1), J.dimension(2)));
    }
  }
}
//-----------------------------------------------------------------------------
void FiniteElement::transform_reference_basis_derivatives(
    std::vector<double>& values, std::size_t order,
    const std::vector<double>& reference_values,
    const Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                        Eigen::RowMajor>>& X,
    const std::vector<double>& J, const tcb::span<const double>& detJ,
    const std::vector<double>& K) const
{
  throw std::runtime_error("Transforming basis derivatives is not implemented yet.");
  std::cout << values(0,0,0,0) + order + reference_values(0,0,0,0) + X(0,0) + J(0,0,0) + detJ[0] + K(0,0,0);
}
//-----------------------------------------------------------------------------
int FiniteElement::num_sub_elements() const noexcept
{
  return _sub_elements.size();
}
//-----------------------------------------------------------------------------
std::size_t FiniteElement::hash() const noexcept { return _hash; }
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
bool FiniteElement::interpolation_ident() const noexcept
{
  return _interpolation_is_ident;
}
//-----------------------------------------------------------------------------
Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
FiniteElement::interpolation_points() const noexcept
{
  return basix::points(_basix_element_handle);
}
//-----------------------------------------------------------------------------
bool FiniteElement::needs_permutation_data() const noexcept
{
  return _needs_permutation_data;
}
//-----------------------------------------------------------------------------
