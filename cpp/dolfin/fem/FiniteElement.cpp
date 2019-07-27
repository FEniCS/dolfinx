// Copyright (C) 2008-2011 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "FiniteElement.h"
#include <dolfin/common/log.h>
#include <dolfin/mesh/utils.h>
#include <functional>
#include <memory>
#include <ufc.h>

using namespace dolfin;
using namespace dolfin::fem;

//-----------------------------------------------------------------------------
FiniteElement::FiniteElement(const ufc_finite_element& ufc_element)
    : _signature(ufc_element.signature), _family(ufc_element.family),
      _tdim(ufc_element.topological_dimension),
      _space_dim(ufc_element.space_dimension),
      _value_size(ufc_element.value_size),
      _reference_value_size(ufc_element.reference_value_size),
      _degree(ufc_element.degree), _hash(std::hash<std::string>{}(_signature)),
      _evaluate_reference_basis(ufc_element.evaluate_reference_basis),
      _evaluate_reference_basis_derivatives(
          ufc_element.evaluate_reference_basis_derivatives),
      _transform_reference_basis_derivatives(
          ufc_element.transform_reference_basis_derivatives),
      _transform_values(ufc_element.transform_values)
{
  // Store dof coordinates on reference element
  assert(ufc_element.tabulate_reference_dof_coordinates);
  _refX.resize(_space_dim, _tdim);
  if (ufc_element.tabulate_reference_dof_coordinates(_refX.data()) == -1)
  {
    throw std::runtime_error(
        "Generated code returned error in tabulate_reference_dof_coordinates");
  }

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
// std::size_t FiniteElement::topological_dimension() const { return _tdim; }
//-----------------------------------------------------------------------------
std::size_t FiniteElement::space_dimension() const { return _space_dim; }
//-----------------------------------------------------------------------------
std::size_t FiniteElement::value_size() const { return _value_size; }
//-----------------------------------------------------------------------------
std::size_t FiniteElement::reference_value_size() const
{
  return _reference_value_size;
}
//-----------------------------------------------------------------------------
int FiniteElement::value_rank() const { return _value_dimension.size(); }
//-----------------------------------------------------------------------------
int FiniteElement::value_dimension(int i) const
{
  if (i >= (int)_value_dimension.size())
    return 1;
  return _value_dimension[i];
}
//-----------------------------------------------------------------------------
std::size_t FiniteElement::degree() const { return _degree; }
//-----------------------------------------------------------------------------
std::string FiniteElement::family() const { return _family; }
//-----------------------------------------------------------------------------
void FiniteElement::evaluate_reference_basis(
    Eigen::Tensor<double, 3, Eigen::RowMajor>& reference_values,
    const Eigen::Ref<const EigenRowArrayXXd> X) const
{
  assert(_evaluate_reference_basis);
  const int num_points = X.rows();
  int ret = _evaluate_reference_basis(reference_values.data(), num_points,
                                      X.data());
  if (ret == -1)
  {
    throw std::runtime_error("Generated code returned error "
                             "in evaluate_reference_basis");
  }
}
//-----------------------------------------------------------------------------
void FiniteElement::transform_reference_basis(
    Eigen::Tensor<double, 3, Eigen::RowMajor>& values,
    const Eigen::Tensor<double, 3, Eigen::RowMajor>& reference_values,
    const Eigen::Ref<const EigenRowArrayXXd> X,
    const Eigen::Tensor<double, 3, Eigen::RowMajor>& J,
    const Eigen::Ref<const EigenArrayXd> detJ,
    const Eigen::Tensor<double, 3, Eigen::RowMajor>& K) const
{
  assert(_transform_reference_basis_derivatives);
  const int num_points = X.rows();
  int ret = _transform_reference_basis_derivatives(
      values.data(), 0, num_points, reference_values.data(), X.data(), J.data(),
      detJ.data(), K.data(), 1);
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
    const Eigen::Ref<const EigenRowArrayXXd> X,
    const Eigen::Tensor<double, 3, Eigen::RowMajor>& J,
    const Eigen::Ref<const EigenArrayXd> detJ,
    const Eigen::Tensor<double, 3, Eigen::RowMajor>& K) const
{
  assert(_transform_reference_basis_derivatives);
  const int num_points = X.rows();
  int ret = _transform_reference_basis_derivatives(
      values.data(), order, num_points, reference_values.data(), X.data(),
      J.data(), detJ.data(), K.data(), 1);
  if (ret == -1)
  {
    throw std::runtime_error("Generated code returned error "
                             "in transform_reference_basis_derivatives");
  }
}
//-----------------------------------------------------------------------------
const EigenRowArrayXXd& FiniteElement::dof_reference_coordinates() const
{
  return _refX;
}
//-----------------------------------------------------------------------------
void FiniteElement::transform_values(
    PetscScalar* reference_values,
    const Eigen::Ref<const Eigen::Array<PetscScalar, Eigen::Dynamic,
                                        Eigen::Dynamic, Eigen::RowMajor>>&
        physical_values,
    const Eigen::Ref<const EigenRowArrayXXd>& coordinate_dofs) const
{
  assert(_transform_values);
  _transform_values(reference_values, physical_values.data(),
                    coordinate_dofs.data(), 1, nullptr);
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
  // Check if there are any sub systems
  if (finite_element.num_sub_elements() == 0)
  {
    throw std::runtime_error(
        "Cannot extract subsystem of finite element. There are no subsystems.");
  }

  // Check that a sub system has been specified
  if (component.empty())
  {
    throw std::runtime_error(
        "Cannot extract subsystem of finite element. No system was specified");
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
