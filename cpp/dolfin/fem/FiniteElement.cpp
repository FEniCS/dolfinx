// Copyright (C) 2008-2011 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "FiniteElement.h"
#include <dolfin/common/utils.h>
#include <memory>
#include <ufc.h>
// #include <spdlog/spdlog.h>

using namespace dolfin;
using namespace dolfin::fem;

//-----------------------------------------------------------------------------
FiniteElement::FiniteElement(std::shared_ptr<const ufc_finite_element> element)
    : _ufc_element(element), _hash(common::hash_local(signature()))
{
  // Store dof coordinates on reference element
  assert(_ufc_element);
  _refX.resize(this->space_dimension(), this->topological_dimension());
  int ret = _ufc_element->tabulate_reference_dof_coordinates(_refX.data());
  if (ret == -1)
  {
    throw std::runtime_error(
        "Generated code returned error in tabulate_reference_dof_coordinates");
  }
}
//-----------------------------------------------------------------------------
std::string FiniteElement::signature() const
{
  assert(_ufc_element);
  assert(_ufc_element->signature);
  return _ufc_element->signature;
}
//-----------------------------------------------------------------------------
CellType FiniteElement::cell_shape() const
{
  assert(_ufc_element);
  const ufc_shape _shape = _ufc_element->cell_shape;
  switch (_shape)
  {
  case interval:
    return CellType::interval;
  case triangle:
    return CellType::triangle;
  case quadrilateral:
    return CellType::quadrilateral;
  case tetrahedron:
    return CellType::tetrahedron;
  case hexahedron:
    return CellType::hexahedron;
  default:
    throw std::runtime_error("Unknown UFC cell type");
    return CellType::interval;
  }
}
//-----------------------------------------------------------------------------
std::size_t FiniteElement::topological_dimension() const
{
  assert(_ufc_element);
  return _ufc_element->topological_dimension;
}
//-----------------------------------------------------------------------------
std::size_t FiniteElement::space_dimension() const
{
  assert(_ufc_element);
  return _ufc_element->space_dimension;
}
//-----------------------------------------------------------------------------
std::size_t FiniteElement::value_size() const
{
  assert(_ufc_element);
  return _ufc_element->value_size;
}
std::size_t FiniteElement::reference_value_size() const
{
  assert(_ufc_element);
  return _ufc_element->reference_value_size;
}
//-----------------------------------------------------------------------------
std::size_t FiniteElement::value_rank() const
{
  assert(_ufc_element);
  return _ufc_element->value_rank;
}
//-----------------------------------------------------------------------------
std::size_t FiniteElement::value_dimension(std::size_t i) const
{
  assert(_ufc_element);
  return _ufc_element->value_dimension(i);
}
//-----------------------------------------------------------------------------
std::size_t FiniteElement::degree() const
{
  assert(_ufc_element);
  return _ufc_element->degree;
}
//-----------------------------------------------------------------------------
std::string FiniteElement::family() const
{
  assert(_ufc_element);
  return _ufc_element->family;
}
//-----------------------------------------------------------------------------
void FiniteElement::evaluate_reference_basis(
    Eigen::Tensor<double, 3, Eigen::RowMajor>& reference_values,
    const Eigen::Ref<const EigenRowArrayXXd> X) const
{
  assert(_ufc_element);
  std::size_t num_points = X.rows();
  int ret = _ufc_element->evaluate_reference_basis(reference_values.data(),
                                                   num_points, X.data());
  if (ret == -1)
    throw std::runtime_error("Generated code returned error "
                             "in evaluate_reference_basis");
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
  assert(_ufc_element);
  std::size_t num_points = X.rows();
  int ret = _ufc_element->transform_reference_basis_derivatives(
      values.data(), 0, num_points, reference_values.data(), X.data(), J.data(),
      detJ.data(), K.data(), 1);
  if (ret == -1)
    throw std::runtime_error("Generated code returned error "
                             "in transform_reference_basis_derivatives");
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
  assert(_ufc_element);
  std::size_t num_points = X.rows();
  int ret = _ufc_element->transform_reference_basis_derivatives(
      values.data(), order, num_points, reference_values.data(), X.data(),
      J.data(), detJ.data(), K.data(), 1);
  if (ret == -1)
    throw std::runtime_error("Generated code returned error "
                             "in transform_reference_basis_derivatives");
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
  assert(_ufc_element);
  _ufc_element->transform_values(reference_values, physical_values.data(),
                                 coordinate_dofs.data(), 1, nullptr);
}

//-----------------------------------------------------------------------------
std::size_t FiniteElement::num_sub_elements() const
{
  assert(_ufc_element);
  return _ufc_element->num_sub_elements;
}

//-----------------------------------------------------------------------------
std::size_t FiniteElement::hash() const { return _hash; }
//-----------------------------------------------------------------------------
std::unique_ptr<FiniteElement>
FiniteElement::create_sub_element(std::size_t i) const
{
  assert(_ufc_element);
  std::shared_ptr<ufc_finite_element> ufc_element(
      _ufc_element->create_sub_element(i), free);
  return std::make_unique<FiniteElement>(ufc_element);
}
//-----------------------------------------------------------------------------
std::unique_ptr<FiniteElement> FiniteElement::create() const
{
  assert(_ufc_element);
  std::shared_ptr<ufc_finite_element> ufc_element(_ufc_element->create(), free);
  return std::make_unique<FiniteElement>(ufc_element);
}
//-----------------------------------------------------------------------------
std::shared_ptr<FiniteElement> FiniteElement::extract_sub_element(
    const std::vector<std::size_t>& component) const
{
  // Recursively extract sub element
  std::shared_ptr<FiniteElement> sub_finite_element
      = extract_sub_element(*this, component);
  // spdlog::debug("Extracted finite element for sub system: %s",
  //               sub_finite_element->signature().c_str());

  return sub_finite_element;
}
//-----------------------------------------------------------------------------
std::shared_ptr<FiniteElement>
FiniteElement::extract_sub_element(const FiniteElement& finite_element,
                                   const std::vector<std::size_t>& component)
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

  // Create sub system
  std::shared_ptr<FiniteElement> sub_element
      = finite_element.create_sub_element(component[0]);
  assert(sub_element);

  // Return sub system if sub sub system should not be extracted
  if (component.size() == 1)
    return sub_element;

  // Otherwise, recursively extract the sub sub system
  const std::vector<std::size_t> sub_component(component.begin() + 1,
                                               component.end());

  return extract_sub_element(*sub_element, sub_component);
}
//-----------------------------------------------------------------------------
