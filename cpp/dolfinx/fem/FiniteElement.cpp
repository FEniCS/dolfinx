// Copyright (C) 2008-2020 Anders Logg and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "FiniteElement.h"
#include <basix.h>
#include <basix/finite-element.h>
#include <dolfinx/common/log.h>
#include <functional>
#include <ufc.h>

using namespace dolfinx;
using namespace dolfinx::fem;

namespace
{
// Recursively extract sub finite element
std::shared_ptr<const FiniteElement>
_extract_sub_element(const FiniteElement& finite_element,
                     const std::vector<int>& component)
{
  // Check that a sub system has been specified
  if (component.empty())
  {
    throw std::runtime_error("Cannot extract subsystem of finite element. No "
                             "system was specified");
  }

  // Check if there are any sub systems
  if (finite_element.num_sub_elements() == 0)
  {
    throw std::runtime_error("Cannot extract subsystem of finite element. "
                             "There are no subsystems.");
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
      = finite_element.sub_elements()[component[0]];
  assert(sub_element);

  // Return sub system if sub sub system should not be extracted
  if (component.size() == 1)
    return sub_element;

  // Otherwise, recursively extract the sub sub system
  const std::vector<int> sub_component(component.begin() + 1, component.end());

  return _extract_sub_element(*sub_element, sub_component);
}
//-----------------------------------------------------------------------------

} // namespace

//-----------------------------------------------------------------------------
FiniteElement::FiniteElement(const ufc_finite_element& ufc_element)
    : _signature(ufc_element.signature), _family(ufc_element.family),
      _tdim(ufc_element.topological_dimension),
      _space_dim(ufc_element.space_dimension),
      _value_size(ufc_element.value_size),
      _reference_value_size(ufc_element.reference_value_size),
      _hash(std::hash<std::string>{}(_signature)), _bs(ufc_element.block_size),
      _interpolation_is_ident(ufc_element.interpolation_is_identity),
      _needs_permutation_data(ufc_element.needs_transformation_data)
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
    // Basix does not support mixed elements, so the subelements should be
    // handled separately
    // This will cause an error, if actually used
    _basix_element_handle = -1;
  }
  else
  {
    _element = std::make_shared<basix::FiniteElement>(basix::create_element(
        family.c_str(), cell_shape.c_str(), ufc_element.degree));

    _basix_element_handle = basix::register_element(
        family.c_str(), cell_shape.c_str(), ufc_element.degree);
  }

  // Fill value dimension
  for (int i = 0; i < ufc_element.value_rank; ++i)
    _value_dimension.push_back(ufc_element.value_shape[i]);

  // Create all sub-elements
  for (int i = 0; i < ufc_element.num_sub_elements; ++i)
  {
    ufc_finite_element* ufc_sub_element = ufc_element.sub_elements[i];
    _sub_elements.push_back(std::make_shared<FiniteElement>(*ufc_sub_element));
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
    const xt::xtensor<double, 2>& X) const
{
  xt::xtensor<double, 4> basis = _element->tabulate(0, X);
  assert(basis.shape(1) == X.shape(0));
  for (std::size_t p = 0; p < basis.shape(1); ++p)
  {
    for (std::size_t d = 0; d < basis.shape(2); ++d)
    {
      for (std::size_t v = 0; v < basis.shape(3); ++v)
      {
        reference_values[p * basis.shape(2) * basis.shape(3)
                         + d * basis.shape(3) + v]
            = basis(0, p, d, v);
      }
    }
  }
}
//-----------------------------------------------------------------------------
// void FiniteElement::evaluate_reference_basis_derivatives(
//     std::vector<double>& /*values*/, int /*order*/,
//     const xt::xtensor<double, 2>& /*X*/) const
// {
//   // NOTE: This function is untested. Add tests and re-active
//   throw std::runtime_error(
//       "FiniteElement::evaluate_reference_basis_derivatives required updating");
// }
//-----------------------------------------------------------------------------
void FiniteElement::transform_reference_basis(
    std::vector<double>& values, const std::vector<double>& reference_values,
    const xt::xtensor<double, 2>& X, const std::vector<double>& J,
    const xtl::span<const double>& detJ, const std::vector<double>& K) const
{
  const int num_points = X.shape(0);
  const int scalar_dim = _space_dim / _bs;
  const int value_size = _value_size / _bs;
  const int Jsize = J.size() / num_points;
  const int Jcols = X.shape(1);
  const int Jrows = Jsize / Jcols;

  basix::map_push_forward_real(
      _basix_element_handle, values.data(), reference_values.data(), J.data(),
      detJ.data(), K.data(), Jrows, value_size, scalar_dim, num_points);
}
//-----------------------------------------------------------------------------
int FiniteElement::num_sub_elements() const noexcept
{
  return _sub_elements.size();
}
//-----------------------------------------------------------------------------
const std::vector<std::shared_ptr<const FiniteElement>>&
FiniteElement::sub_elements() const noexcept
{
  return _sub_elements;
}
//-----------------------------------------------------------------------------

std::size_t FiniteElement::hash() const noexcept { return _hash; }
//-----------------------------------------------------------------------------
std::shared_ptr<const FiniteElement>
FiniteElement::extract_sub_element(const std::vector<int>& component) const
{
  // Recursively extract sub element
  std::shared_ptr<const FiniteElement> sub_finite_element
      = _extract_sub_element(*this, component);
  DLOG(INFO) << "Extracted finite element for sub-system: "
             << sub_finite_element->signature().c_str();
  return sub_finite_element;
}
//-----------------------------------------------------------------------------
bool FiniteElement::interpolation_ident() const noexcept
{
  return _interpolation_is_ident;
}
//-----------------------------------------------------------------------------
const xt::xtensor<double, 2>& FiniteElement::interpolation_points() const
{
  if (!_element)
  {
    throw std::runtime_error("Cannot get interpolation points - no basix "
                             "element handle. Maybe this is a mixed element?");
  }

  return _element->points();
}
//-----------------------------------------------------------------------------
bool FiniteElement::needs_permutation_data() const noexcept
{
  return _needs_permutation_data;
}
//-----------------------------------------------------------------------------
void FiniteElement::map_pull_back(double* physical_data,
                                  const double* reference_data, const double* J,
                                  const double* detJ, const double* K,
                                  const int physical_dim,
                                  const int physical_value_size,
                                  const int nresults, const int npoints) const
{
  basix::map_pull_back_real(_basix_element_handle, physical_data,
                            reference_data, J, detJ, K, physical_dim,
                            physical_value_size, nresults, npoints);
}
//-----------------------------------------------------------------------------
void FiniteElement::map_pull_back(std::complex<double>* physical_data,
                                  const std::complex<double>* reference_data,
                                  const double* J, const double* detJ,
                                  const double* K, const int physical_dim,
                                  const int physical_value_size,
                                  const int nresults, const int npoints) const
{
  basix::map_pull_back_complex(_basix_element_handle, physical_data,
                               reference_data, J, detJ, K, physical_dim,
                               physical_value_size, nresults, npoints);
}
//-----------------------------------------------------------------------------
void FiniteElement::apply_dof_transformation(double* data,
                                             std::uint32_t cell_permutation,
                                             int block_size) const
{
  basix::apply_dof_transformation_real(_basix_element_handle, data, block_size,
                                       cell_permutation);
}
//-----------------------------------------------------------------------------
void FiniteElement::apply_dof_transformation(std::complex<double>* data,
                                             std::uint32_t cell_permutation,
                                             int block_size) const
{
  basix::apply_dof_transformation_complex(_basix_element_handle, data,
                                          block_size, cell_permutation);
}
//-----------------------------------------------------------------------------
void FiniteElement::apply_inverse_transpose_dof_transformation(
    double* data, std::uint32_t cell_permutation, int block_size) const
{
  basix::apply_inverse_transpose_dof_transformation_real(
      _basix_element_handle, data, block_size, cell_permutation);
}
//-----------------------------------------------------------------------------
void FiniteElement::apply_inverse_transpose_dof_transformation(
    std::complex<double>* data, std::uint32_t cell_permutation,
    int block_size) const
{
  basix::apply_inverse_transpose_dof_transformation_complex(
      _basix_element_handle, data, block_size, cell_permutation);
}
//-----------------------------------------------------------------------------
