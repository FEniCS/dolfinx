// Copyright (C) 2008-2011 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "FiniteElement.h"
#include <functional>
#include <memory>
#include <ufc.h>
// #include <spdlog/spdlog.h>

using namespace dolfin;
using namespace dolfin::fem;

//-----------------------------------------------------------------------------
FiniteElement::FiniteElement(const ufc_finite_element& element)
    : _signature(element.signature), _family(element.family),
      _tdim(element.topological_dimension), _space_dim(element.space_dimension),
      _value_size(element.value_size),
      _reference_value_size(element.reference_value_size),
      _degree(element.degree), _hash(std::hash<std::string>{}(_signature)),
      _evaluate_reference_basis(element.evaluate_reference_basis),
      _evaluate_reference_basis_derivatives(
          element.evaluate_reference_basis_derivatives),
      _transform_reference_basis_derivatives(
          element.transform_reference_basis_derivatives),
      _transform_values(element.transform_values)
{
  // Store dof coordinates on reference element
  _refX.resize(this->space_dimension(), this->topological_dimension());
  assert(element.tabulate_reference_dof_coordinates);
  int ret = element.tabulate_reference_dof_coordinates(_refX.data());
  if (ret == -1)
  {
    throw std::runtime_error(
        "Generated code returned error in tabulate_reference_dof_coordinates");
  }

  const ufc_shape _shape = element.cell_shape;
  switch (_shape)
  {
  case interval:
    _cell_shape = CellType::interval;
    break;
  case triangle:
    _cell_shape = CellType::triangle;
    break;
  case quadrilateral:
    _cell_shape = CellType::quadrilateral;
    break;
  case tetrahedron:
    _cell_shape = CellType::tetrahedron;
    break;
  case hexahedron:
    _cell_shape = CellType::hexahedron;
    break;
  default:
    throw std::runtime_error("Unknown UFC cell type");
  }
  assert(ReferenceCellTopology::dim(_cell_shape) == _tdim);

  // Fill value dimension
  for (int i = 0; i < element.value_rank; ++i)
    _value_dimension.push_back(element.value_dimension(i));

  // Create all sub-elements
  for (int i = 0; i < element.num_sub_elements; ++i)
  {
    ufc_finite_element* ufc_sub_element = element.create_sub_element(i);
    _sub_elements.push_back(std::make_shared<FiniteElement>(*ufc_sub_element));
    std::free(ufc_sub_element);
  }
}
//-----------------------------------------------------------------------------
FiniteElement::FiniteElement(
    std::string signature, std::string family, int topological_dimension,
    int space_dimension, const std::vector<int>& value_dimension,
    int value_size, int reference_value_size, int degree,
    std::function<int(double*, int, int, const double*)>
        evaluate_reference_basis_derivatives,
    std::function<int(double*, int, int, const double*, const double*,
                      const double*, const double*, const double*, int)>
        transform_reference_basis_derivatives,
    std::function<int(ufc_scalar_t*, const ufc_scalar_t*, const double*, int,
                      const ufc_coordinate_mapping*)>
        transform_values)
    : _signature(signature), _family(family), _tdim(topological_dimension),
      _space_dim(space_dimension), _value_size(value_size),
      _reference_value_size(reference_value_size), _degree(degree),
      _value_dimension(value_dimension),
      _evaluate_reference_basis_derivatives(
          evaluate_reference_basis_derivatives),
      _transform_reference_basis_derivatives(
          transform_reference_basis_derivatives),
      _transform_values(transform_values)
{
}
//-----------------------------------------------------------------------------
std::string FiniteElement::signature() const { return _signature; }
//-----------------------------------------------------------------------------
CellType FiniteElement::cell_shape() const { return _cell_shape; }
//-----------------------------------------------------------------------------
std::size_t FiniteElement::topological_dimension() const { return _tdim; }
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
std::size_t FiniteElement::value_rank() const
{
  return _value_dimension.size();
}
//-----------------------------------------------------------------------------
std::size_t FiniteElement::value_dimension(std::size_t i) const
{
  if (i >= _value_dimension.size())
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
  std::size_t num_points = X.rows();
  assert(_evaluate_reference_basis_derivatives);
  int ret = _evaluate_reference_basis_derivatives(reference_values.data(), 0,
                                                  num_points, X.data());
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
  std::size_t num_points = X.rows();
  assert(_transform_reference_basis_derivatives);
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
  std::size_t num_points = X.rows();
  assert(_transform_reference_basis_derivatives);
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
std::size_t FiniteElement::num_sub_elements() const
{
  return _sub_elements.size();
}
//-----------------------------------------------------------------------------
std::size_t FiniteElement::hash() const { return _hash; }
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

  // Get sub system
  std::shared_ptr<FiniteElement> sub_element
      = finite_element._sub_elements[component[0]];
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
