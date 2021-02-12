// Copyright (C) 2008-2020 Anders Logg and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "FiniteElement.h"
#include <basix.h>
#include <dolfinx/common/log.h>
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
    std::vector<int> value_shape(basix::value_rank(_basix_element_handle));
    basix::value_shape(_basix_element_handle, value_shape.data());
    int basix_value_size = 1;
    for (int w : value_shape)
      basix_value_size *= w;

    _interpolation_matrix = std::vector<double>(
        basix::dim(_basix_element_handle)
        * basix::interpolation_num_points(_basix_element_handle)
        * basix_value_size);
    basix::interpolation_matrix(_basix_element_handle,
                                _interpolation_matrix.data());
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
    const common::array2d<double>& X) const
{
  const int scalar_reference_value_size = _reference_value_size / _bs;
  common::array2d<double> basix_data(X.shape[0],
                                     basix::dim(_basix_element_handle)
                                         * scalar_reference_value_size);
  basix::tabulate(_basix_element_handle, basix_data.data(), 0, X.data(),
                  X.shape[0]);

  assert(basix_data.shape[1] % scalar_reference_value_size == 0);
  const int scalar_dofs = basix_data.shape[1] / scalar_reference_value_size;

  assert(reference_values.size()
         == X.shape[0] * scalar_dofs * scalar_reference_value_size);

  for (std::size_t p = 0; p < X.shape[0]; ++p)
  {
    for (int d = 0; d < scalar_dofs; ++d)
    {
      for (int v = 0; v < scalar_reference_value_size; ++v)
      {
        reference_values[(p * scalar_dofs + d) * scalar_reference_value_size
                         + v]
            = basix_data(p, d + scalar_dofs * v);
      }
    }
  }
}
//-----------------------------------------------------------------------------
void FiniteElement::evaluate_reference_basis_derivatives(
    std::vector<double>& values, int order,
    const common::array2d<double>& X) const
{
  // TODO: fix this for order > 1
  if (order != 1)
  {
    throw std::runtime_error(
        "FiniteElement::evaluate_reference_basis_derivatives only supports "
        "order 1 at the moment.");
  }

  // nd = tdim + 1;
  // FIXME
  const int nd = 4;
  common::array2d<double> basix_data(nd * X.shape[0],
                                     basix::dim(_basix_element_handle));
  basix::tabulate(_basix_element_handle, basix_data.data(), 1, X.data(),
                  X.shape[0]);
  for (std::size_t p = 0; p < X.shape[0]; ++p)
  {
    for (std::size_t d = 0; d < basix_data.shape[1] / _reference_value_size;
         ++d)
    {
      for (int v = 0; v < _reference_value_size; ++v)
      {
        for (std::size_t deriv = 0; deriv < nd; ++deriv)
        {
          values[(p * basix_data.shape[1] + d * _reference_value_size + v)
                     * (basix_data.size() - 1)
                 + deriv]
              = basix_data(p, d * _reference_value_size + v);
        }
      }
    }
  }
}
//-----------------------------------------------------------------------------
void FiniteElement::transform_reference_basis(
    std::vector<double>& values, const std::vector<double>& reference_values,
    const common::array2d<double>& X, const std::vector<double>& J,
    const tcb::span<const double>& detJ, const std::vector<double>& K) const
{
  const int num_points = X.shape[0];
  const int scalar_dim = _space_dim / _bs;
  const int value_size = _value_size / _bs;
  const int size_per_point = scalar_dim * value_size;
  const int Jsize = J.size() / num_points;
  const int Jcols = X.shape[1];
  const int Jrows = Jsize / Jcols;

  if ((int)(values.size()) != size_per_point * num_points)
    throw std::runtime_error("OH NO!");

  for (int pt = 0; pt < num_points; ++pt)
  {
    basix::map_push_forward(
        _basix_element_handle, values.data() + pt * size_per_point,
        reference_values.data() + pt * size_per_point, J.data() + Jsize * pt,
        detJ[pt], K.data() + Jsize * pt, Jrows, value_size, scalar_dim);
  }
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
common::array2d<double> FiniteElement::interpolation_points() const
{
  if (_basix_element_handle == -1)
  {
    throw std::runtime_error("Cannot get interpolation points - no basix "
                             "element handle. Maybe this is a mixed element?");
  }
  const int gdim
      = basix::cell_geometry_dimension(basix::cell_type(_basix_element_handle));
  common::array2d<double> points(
      basix::interpolation_num_points(_basix_element_handle), gdim);
  basix::interpolation_points(_basix_element_handle, points.data());
  return points;
}
//-----------------------------------------------------------------------------
bool FiniteElement::needs_permutation_data() const noexcept
{
  return _needs_permutation_data;
}
//-----------------------------------------------------------------------------
