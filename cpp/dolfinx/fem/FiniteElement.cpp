// Copyright (C) 2008-2020 Anders Logg and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "FiniteElement.h"
#include <basix/finite-element.h>
#include <basix/interpolation.h>
#include <dolfinx/common/log.h>
#include <functional>
#include <ufc.h>

using namespace dolfinx;
using namespace dolfinx::fem;

namespace
{
// Check if an element is a basix element (or a blocked element
// containing a Basix element)
bool is_basix_element(const ufc_finite_element& element)
{
  if (element.element_type == ufc_basix_element)
    return true;
  else if (element.element_type == ufc_blocked_element)
  {
    // TODO: what should happen if the element is a blocked element
    // containing a blocked element containing a Basix element?
    return element.sub_elements[0]->element_type == ufc_basix_element;
  }
  else
    return false;
}

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
      _hash(std::hash<std::string>{}(_signature)), _bs(ufc_element.block_size)
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
  case prism:
    _cell_shape = mesh::CellType::prism;
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
      = {{vertex, "point"},         {interval, "interval"},
         {triangle, "triangle"},    {tetrahedron, "tetrahedron"},
         {prism, "prism"},          {quadrilateral, "quadrilateral"},
         {hexahedron, "hexahedron"}};
  const std::string cell_shape = ufc_to_cell.at(ufc_element.cell_shape);

  // Fill value dimension
  for (int i = 0; i < ufc_element.value_rank; ++i)
    _value_dimension.push_back(ufc_element.value_shape[i]);

  _needs_dof_transformations = false;
  _needs_dof_permutations = false;
  // Create all sub-elements
  for (int i = 0; i < ufc_element.num_sub_elements; ++i)
  {
    ufc_finite_element* ufc_sub_element = ufc_element.sub_elements[i];
    _sub_elements.push_back(std::make_shared<FiniteElement>(*ufc_sub_element));
    if (_sub_elements[i]->needs_dof_permutations()
        and !_needs_dof_transformations)
    {
      _needs_dof_permutations = true;
    }
    if (_sub_elements[i]->needs_dof_transformations())
    {
      _needs_dof_permutations = false;
      _needs_dof_transformations = true;
    }
  }

  if (is_basix_element(ufc_element))
  {
    if (ufc_element.lagrange_variant != -1)
    {
      _element = std::make_unique<basix::FiniteElement>(basix::create_element(
          static_cast<basix::element::family>(ufc_element.basix_family),
          static_cast<basix::cell::type>(ufc_element.basix_cell),
          ufc_element.degree,
          static_cast<basix::element::lagrange_variant>(
              ufc_element.lagrange_variant),
          ufc_element.discontinuous));
    }
    else
    {
      _element = std::make_unique<basix::FiniteElement>(basix::create_element(
          static_cast<basix::element::family>(ufc_element.basix_family),
          static_cast<basix::cell::type>(ufc_element.basix_cell),
          ufc_element.degree, ufc_element.discontinuous));
    }

    _needs_dof_transformations
        = !_element->dof_transformations_are_identity()
          and !_element->dof_transformations_are_permutations();

    _needs_dof_permutations
        = !_element->dof_transformations_are_identity()
          and _element->dof_transformations_are_permutations();
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
void FiniteElement::tabulate(xt::xtensor<double, 4>& reference_values,
                             const xt::xtensor<double, 2>& X, int order) const
{
  assert(_element);
  reference_values = _element->tabulate(order, X);
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
  assert(_element);
  return _element->map_type == basix::maps::type::identity;
}
//-----------------------------------------------------------------------------
const xt::xtensor<double, 2>& FiniteElement::interpolation_points() const
{
  if (!_element)
  {
    throw std::runtime_error(
        "Cannot get interpolation points - no Basix element available. Maybe "
        "this is a mixed element?");
  }

  return _element->points();
}
//-----------------------------------------------------------------------------
xt::xtensor<double, 2>
FiniteElement::create_interpolation_operator(const FiniteElement& from) const
{
  if (_element->mapping_type() != from._element->mapping_type())
  {
    throw std::runtime_error(
        "Interpolation for elements with different maps is not yet supported.");
  }

  if (_bs == 1 or from._bs == 1)
  {
    // If one of the elements have bs=1, Basix can figure out the size of the
    // matrix
    return basix::compute_interpolation_operator(*from._element, *_element);
  }
  else if (_bs > 1 and from._bs == _bs)
  {
    // If bs != 1 for at least one element, then bs0 == bs1 for this
    // case
    xt::xtensor<double, 2> i_m
        = basix::compute_interpolation_operator(*from._element, *_element);
    std::array<std::size_t, 2> shape = {i_m.shape(0) * _bs, i_m.shape(1) * _bs};
    xt::xtensor<double, 2> out = xt::zeros<double>(shape);

    // NOTE: Alternatively this operation could be implemented during
    // matvec with the original matrix
    for (std::size_t i = 0; i < i_m.shape(0); ++i)
      for (std::size_t j = 0; j < i_m.shape(1); ++j)
        for (int k = 0; k < _bs; ++k)
          out(i * _bs + k, j * _bs + k) = i_m(i, j);

    return out;
  }
  else
  {
    throw std::runtime_error(
        "Interpolation for element combination is not supported.");
  }
}
//-----------------------------------------------------------------------------
bool FiniteElement::needs_dof_transformations() const noexcept
{
  return _needs_dof_transformations;
}
//-----------------------------------------------------------------------------
bool FiniteElement::needs_dof_permutations() const noexcept
{
  return _needs_dof_permutations;
}
//-----------------------------------------------------------------------------
void FiniteElement::permute_dofs(const xtl::span<std::int32_t>& doflist,
                                 std::uint32_t cell_permutation) const
{
  _element->permute_dofs(doflist, cell_permutation);
}
//-----------------------------------------------------------------------------
void FiniteElement::unpermute_dofs(const xtl::span<std::int32_t>& doflist,
                                   std::uint32_t cell_permutation) const
{
  _element->unpermute_dofs(doflist, cell_permutation);
}
//-----------------------------------------------------------------------------
std::function<void(const xtl::span<std::int32_t>&, std::uint32_t)>
FiniteElement::get_dof_permutation_function(bool inverse,
                                            bool scalar_element) const
{
  if (!needs_dof_permutations())
  {
    if (!needs_dof_transformations())
    {
      // If this element shouldn't be permuted, return a function that
      // does nothing
      return [](const xtl::span<std::int32_t>&, std::uint32_t) {};
    }
    else
    {
      // If this element shouldn't be permuted but needs transformations, return
      // a function that throws an error
      return [](const xtl::span<std::int32_t>&, std::uint32_t)
      {
        throw std::runtime_error(
            "Permutations should not be applied for this element.");
      };
    }
  }

  if (_sub_elements.size() != 0)
  {
    if (_bs == 1)
    {
      // Mixed element
      std::vector<
          std::function<void(const xtl::span<std::int32_t>&, std::uint32_t)>>
          sub_element_functions;
      std::vector<int> dims;
      for (std::size_t i = 0; i < _sub_elements.size(); ++i)
      {
        sub_element_functions.push_back(
            _sub_elements[i]->get_dof_permutation_function(inverse));
        dims.push_back(_sub_elements[i]->space_dimension());
      }

      return
          [dims, sub_element_functions](const xtl::span<std::int32_t>& doflist,
                                        std::uint32_t cell_permutation)
      {
        std::size_t start = 0;
        for (std::size_t e = 0; e < sub_element_functions.size(); ++e)
        {
          sub_element_functions[e](doflist.subspan(start, dims[e]),
                                   cell_permutation);
          start += dims[e];
        }
      };
    }
    else if (!scalar_element)
    {
      throw std::runtime_error(
          "Permuting DOFs for vector elements not implemented.");
    }
  }

  if (inverse)
  {
    return [this](const xtl::span<std::int32_t>& doflist,
                  std::uint32_t cell_permutation)
    { unpermute_dofs(doflist, cell_permutation); };
  }
  else
  {
    return [this](const xtl::span<std::int32_t>& doflist,
                  std::uint32_t cell_permutation)
    { permute_dofs(doflist, cell_permutation); };
  }
}
//-----------------------------------------------------------------------------
