// Copyright (C) 2020-2021 Garth N. Wells and Matthew W. Scroggs
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "FiniteElement.h"
#include <algorithm>
#include <array>
#include <basix/finite-element.h>
#include <basix/interpolation.h>
#include <basix/polyset.h>
#include <dolfinx/common/log.h>
#include <functional>
#include <numeric>
#include <utility>
#include <vector>

using namespace dolfinx;
using namespace dolfinx::fem;

namespace
{
/// @brief Create a list of fem::FiniteElement from Basix elements and
/// other data.
/// @tparam T
/// @param elements
/// @return List of DOLFINx elements
template <std::floating_point T>
std::vector<std::shared_ptr<const FiniteElement<T>>>
_build_element_list(std::vector<BasixElementData<T>> elements)
{
  std::vector<std::shared_ptr<const FiniteElement<T>>> _e;
  std::ranges::transform(elements, std::back_inserter(_e),
                         [](auto data)
                         {
                           auto [e, bs, symm] = data;
                          //  int _bs = bs ? bs->front() : 1;
                           // TODO: Check bs and symm, bs should be empty?
                           return std::make_shared<fem::FiniteElement<T>>(e);
                         });
  return _e;
}

/// Recursively extract sub finite element
template <std::floating_point T>
std::shared_ptr<const FiniteElement<T>>
_extract_sub_element(const FiniteElement<T>& finite_element,
                     std::span<const int> component)
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
    throw std::runtime_error("Cannot extract subsystem of finite element. "
                             "Requested subsystem out of range.");
  }

  // Get sub system
  auto sub_element = finite_element.sub_elements()[component[0]];
  assert(sub_element);

  // Return sub system if sub sub system should not be extracted
  if (component.size() == 1)
    return sub_element;

  // Otherwise, recursively extract the sub sub system
  std::vector<int> sub_component(component.begin() + 1, component.end());

  return _extract_sub_element(*sub_element, sub_component);
}
} // namespace

//-----------------------------------------------------------------------------
template <std::floating_point T>
FiniteElement<T>::FiniteElement(
    const basix::FiniteElement<T>& element,
    std::optional<std::vector<std::size_t>> block_shape, bool symmetric)
    : _cell_type(mesh::cell_type_from_basix_type(element.cell_type())),
      // _space_dim(block_size * element.dim()),
      _reference_value_shape(element.value_shape()),
      // _bs(block_size),
      _element(std::make_unique<basix::FiniteElement<T>>(element)),
      _symmetric(symmetric),
      _needs_dof_permutations(
          !_element->dof_transformations_are_identity()
          and _element->dof_transformations_are_permutations()),
      _needs_dof_transformations(
          !_element->dof_transformations_are_identity()
          and !_element->dof_transformations_are_permutations()),
      _entity_dofs(element.entity_dofs()),
      _entity_closure_dofs(element.entity_closure_dofs())
{
  _bs = block_shape ? std::accumulate(block_shape->begin(), block_shape->end(),
                                      1, std::multiplies{})
                    : 1;

  // Create all sub-elements
  if (block_shape)
  {
    _sub_elements
        = std::vector<std::shared_ptr<const FiniteElement<geometry_type>>>(
            _bs, std::make_shared<FiniteElement<T>>(element));
    _reference_value_shape = *block_shape;
  }

  _space_dim = _bs * element.dim();

  std::string family;
  switch (_element->family())
  {
  case basix::element::family::P:
    family = "Lagrange";
    break;
  case basix::element::family::DPC:
    family = "Discontinuous Lagrange";
    break;
  default:
    family = "unknown";
    break;
  }

  _signature = "Basix element " + family + " " + std::to_string(_bs);
}
//-----------------------------------------------------------------------------
template <std::floating_point T>
FiniteElement<T>::FiniteElement(std::vector<BasixElementData<T>> elements)
    : FiniteElement(_build_element_list(elements))
{
}
//-----------------------------------------------------------------------------
template <std::floating_point T>
FiniteElement<T>::FiniteElement(
    const std::vector<std::shared_ptr<const FiniteElement<T>>>& elements)
    : _cell_type(elements.front()->cell_type()), _space_dim(0),
      _sub_elements(elements), _reference_value_shape(std::nullopt), _bs(1),
      _symmetric(false), _needs_dof_permutations(false),
      _needs_dof_transformations(false)
{
  if (elements.size() < 2)
  {
    throw std::runtime_error("FiniteElement constructor for mixed elements "
                             "called with a single element.");
  }

  _signature = "Mixed element (";

  const std::vector<std::vector<std::vector<int>>>& ed
      = elements.front()->entity_dofs();
  _entity_dofs.resize(ed.size());
  _entity_closure_dofs.resize(ed.size());
  for (std::size_t i = 0; i < ed.size(); ++i)
  {
    _entity_dofs[i].resize(ed[i].size());
    _entity_closure_dofs[i].resize(ed[i].size());
  }

  int dof_offset = 0;
  for (auto& e : elements)
  {
    _signature += e->signature() + ", ";

    if (e->needs_dof_permutations())
      _needs_dof_permutations = true;
    if (e->needs_dof_transformations())
      _needs_dof_transformations = true;

    const std::size_t sub_bs = e->block_size();
    for (std::size_t i = 0; i < _entity_dofs.size(); ++i)
    {
      for (std::size_t j = 0; j < _entity_dofs[i].size(); ++j)
      {
        std::vector<int> sub_ed = e->entity_dofs()[i][j];
        std::vector<int> sub_ecd = e->entity_closure_dofs()[i][j];
        for (auto k : sub_ed)
        {
          for (std::size_t b = 0; b < sub_bs; ++b)
            _entity_dofs[i][j].push_back(dof_offset + k * sub_bs + b);
        }
        for (auto k : sub_ecd)
        {
          for (std::size_t b = 0; b < sub_bs; ++b)
            _entity_closure_dofs[i][j].push_back(dof_offset + k * sub_bs + b);
        }
      }
    }

    dof_offset += e->space_dimension();
  }

  _space_dim = dof_offset;
  _signature += ")";
}
//-----------------------------------------------------------------------------
template <std::floating_point T>
FiniteElement<T>::FiniteElement(mesh::CellType cell_type,
                                std::span<const geometry_type> points,
                                std::array<std::size_t, 2> pshape,
                                std::size_t block_size, bool symmetric)
    : _cell_type(cell_type),
      _signature("Quadrature element " + std::to_string(pshape[0]) + " "
                 + std::to_string(block_size)),
      _space_dim(pshape[0] * block_size), _reference_value_shape({}),
      _bs(block_size), _symmetric(symmetric), _needs_dof_permutations(false),
      _needs_dof_transformations(false),
      _entity_dofs(mesh::cell_dim(cell_type) + 1),
      _entity_closure_dofs(mesh::cell_dim(cell_type) + 1),
      _points(std::vector<T>(points.begin(), points.end()), pshape)
{
  const int tdim = mesh::cell_dim(cell_type);
  for (int d = 0; d <= tdim; ++d)
  {
    int num_entities = mesh::cell_num_entities(cell_type, d);
    _entity_dofs[d].resize(num_entities);
    _entity_closure_dofs[d].resize(num_entities);
  }

  for (std::size_t i = 0; i < pshape[0]; ++i)
  {
    _entity_dofs[tdim][0].push_back(i);
    _entity_closure_dofs[tdim][0].push_back(i);
  }
}
//-----------------------------------------------------------------------------
template <std::floating_point T>
bool FiniteElement<T>::operator==(const FiniteElement& e) const
{
  if (!_element or !e._element)
  {
    throw std::runtime_error(
        "Missing a Basix element. Cannot check for equivalence");
  }

  return *_element == *e._element;
}
//-----------------------------------------------------------------------------
template <std::floating_point T>
bool FiniteElement<T>::operator!=(const FiniteElement& e) const
{
  return !(*this == e);
}
//-----------------------------------------------------------------------------
template <std::floating_point T>
mesh::CellType FiniteElement<T>::cell_type() const noexcept
{
  return _cell_type;
}
//-----------------------------------------------------------------------------
template <std::floating_point T>
std::string FiniteElement<T>::signature() const noexcept
{
  return _signature;
}
//-----------------------------------------------------------------------------
template <std::floating_point T>
int FiniteElement<T>::space_dimension() const noexcept
{
  return _space_dim;
}
//-----------------------------------------------------------------------------
template <std::floating_point T>
std::span<const std::size_t> FiniteElement<T>::reference_value_shape() const
{
  if (_reference_value_shape)
    return *_reference_value_shape;
  else
    throw std::runtime_error("Element does not have a reference_value_shape.");
}
//-----------------------------------------------------------------------------
template <std::floating_point T>
const std::vector<std::vector<std::vector<int>>>&
FiniteElement<T>::entity_dofs() const noexcept
{
  return _entity_dofs;
}
//-----------------------------------------------------------------------------
template <std::floating_point T>
const std::vector<std::vector<std::vector<int>>>&
FiniteElement<T>::entity_closure_dofs() const noexcept
{
  return _entity_closure_dofs;
}
//-----------------------------------------------------------------------------
template <std::floating_point T>
bool FiniteElement<T>::symmetric() const
{
  return _symmetric;
}
//-----------------------------------------------------------------------------
template <std::floating_point T>
int FiniteElement<T>::reference_value_size() const
{
  if (_reference_value_shape)
  {
    return std::accumulate(_reference_value_shape->begin(),
                           _reference_value_shape->end(), 1, std::multiplies{});
  }
  else
    throw std::runtime_error("Element does not have a reference_value_shape.");
}
//-----------------------------------------------------------------------------
template <std::floating_point T>
int FiniteElement<T>::block_size() const noexcept
{
  return _bs;
}
//-----------------------------------------------------------------------------
template <std::floating_point T>
void FiniteElement<T>::tabulate(std::span<T> values, std::span<const T> X,
                                std::array<std::size_t, 2> shape,
                                int order) const
{
  assert(_element);
  _element->tabulate(order, X, shape, values);
}
//-----------------------------------------------------------------------------
template <std::floating_point T>
std::pair<std::vector<T>, std::array<std::size_t, 4>>
FiniteElement<T>::tabulate(std::span<const T> X,
                           std::array<std::size_t, 2> shape, int order) const
{
  assert(_element);
  return _element->tabulate(order, X, shape);
}
//-----------------------------------------------------------------------------
template <std::floating_point T>
int FiniteElement<T>::num_sub_elements() const noexcept
{
  return _sub_elements.size();
}
//-----------------------------------------------------------------------------
template <std::floating_point T>
bool FiniteElement<T>::is_mixed() const noexcept
{
  return !_reference_value_shape;
}
//-----------------------------------------------------------------------------
template <std::floating_point T>
const std::vector<std::shared_ptr<const FiniteElement<T>>>&
FiniteElement<T>::sub_elements() const noexcept
{
  return _sub_elements;
}
//-----------------------------------------------------------------------------
template <std::floating_point T>
std::shared_ptr<const FiniteElement<T>>
FiniteElement<T>::extract_sub_element(const std::vector<int>& component) const
{
  // Recursively extract sub element
  auto sub_finite_element = _extract_sub_element(*this, component);
  spdlog::debug("Extracted finite element for sub-system: {}",
                sub_finite_element->signature().c_str());
  return sub_finite_element;
}
//-----------------------------------------------------------------------------
template <std::floating_point T>
const basix::FiniteElement<T>& FiniteElement<T>::basix_element() const
{
  if (!_element)
  {
    throw std::runtime_error("No Basix element available. "
                             "Maybe this is a mixed element?");
  }

  return *_element;
}
//-----------------------------------------------------------------------------
template <std::floating_point T>
basix::maps::type FiniteElement<T>::map_type() const
{
  if (!_element)
  {
    throw std::runtime_error("Cannot element map type - no Basix element "
                             "available. Maybe this is a mixed element?");
  }

  return _element->map_type();
}
//-----------------------------------------------------------------------------
template <std::floating_point T>
bool FiniteElement<T>::map_ident() const noexcept
{
  if (!_element && _points.second[0] > 0)
    // Quadratute elements must use identity map
    return true;
  assert(_element);
  return _element->map_type() == basix::maps::type::identity;
}
//-----------------------------------------------------------------------------
template <std::floating_point T>
bool FiniteElement<T>::interpolation_ident() const noexcept
{
  if (!_element && _points.second[0] > 0)
    return true;
  assert(_element);
  return _element->interpolation_is_identity();
}
//-----------------------------------------------------------------------------
template <std::floating_point T>
std::pair<std::vector<T>, std::array<std::size_t, 2>>
FiniteElement<T>::interpolation_points() const
{
  if (_points.second[0] > 0)
    return _points;
  if (!_element)
  {
    throw std::runtime_error(
        "Cannot get interpolation points - no Basix element available. Maybe "
        "this is a mixed element?");
  }

  return _element->points();
}
//-----------------------------------------------------------------------------
template <std::floating_point T>
std::pair<std::vector<T>, std::array<std::size_t, 2>>
FiniteElement<T>::interpolation_operator() const
{
  if (!_element)
  {
    throw std::runtime_error("No underlying element for interpolation. "
                             "Cannot interpolate mixed elements directly.");
  }

  return _element->interpolation_matrix();
}
//-----------------------------------------------------------------------------
template <std::floating_point T>
std::pair<std::vector<T>, std::array<std::size_t, 2>>
FiniteElement<T>::create_interpolation_operator(const FiniteElement& from) const
{
  assert(_element);
  assert(from._element);
  if (_element->map_type() != from._element->map_type())
  {
    throw std::runtime_error("Interpolation between elements with different "
                             "maps is not supported.");
  }

  if (_bs == 1 or from._bs == 1)
  {
    // If one of the elements has bs=1, Basix can figure out the size
    // of the matrix
    return basix::compute_interpolation_operator<T>(*from._element, *_element);
  }
  else if (_bs > 1 and from._bs == _bs)
  {
    // If bs != 1 for at least one element, then bs0 == bs1 for this
    // case
    const auto [data, dshape]
        = basix::compute_interpolation_operator<T>(*from._element, *_element);
    std::array<std::size_t, 2> shape = {dshape[0] * _bs, dshape[1] * _bs};
    std::vector<T> out(shape[0] * shape[1]);

    // NOTE: Alternatively this operation could be implemented during
    // matvec with the original matrix.
    for (std::size_t i = 0; i < dshape[0]; ++i)
      for (std::size_t j = 0; j < dshape[1]; ++j)
        for (int k = 0; k < _bs; ++k)
          out[shape[1] * (i * _bs + k) + (j * _bs + k)]
              = data[dshape[1] * i + j];

    return {std::move(out), shape};
  }
  else
  {
    throw std::runtime_error(
        "Interpolation for element combination is not supported.");
  }
}
//-----------------------------------------------------------------------------
template <std::floating_point T>
bool FiniteElement<T>::needs_dof_transformations() const noexcept
{
  return _needs_dof_transformations;
}
//-----------------------------------------------------------------------------
template <std::floating_point T>
bool FiniteElement<T>::needs_dof_permutations() const noexcept
{
  return _needs_dof_permutations;
}
//-----------------------------------------------------------------------------
template <std::floating_point T>
void FiniteElement<T>::permute(std::span<std::int32_t> doflist,
                               std::uint32_t cell_permutation) const
{
  _element->permute(doflist, cell_permutation);
}
//-----------------------------------------------------------------------------
template <std::floating_point T>
void FiniteElement<T>::permute_inv(std::span<std::int32_t> doflist,
                                   std::uint32_t cell_permutation) const
{
  _element->permute_inv(doflist, cell_permutation);
}
//-----------------------------------------------------------------------------
/// @cond
template <std::floating_point T>
std::function<void(std::span<std::int32_t>, std::uint32_t)>
FiniteElement<T>::dof_permutation_fn(bool inverse, bool scalar_element) const
/// @endcond
{
  if (!needs_dof_permutations())
    return [](std::span<std::int32_t>, std::uint32_t) {};

  if (!_sub_elements.empty())
  {
    if (_bs == 1)
    {
      // Mixed element
      std::vector<std::function<void(std::span<std::int32_t>, std::uint32_t)>>
          sub_element_functions;
      std::vector<int> dims;
      for (std::size_t i = 0; i < _sub_elements.size(); ++i)
      {
        sub_element_functions.push_back(
            _sub_elements[i]->dof_permutation_fn(inverse));
        dims.push_back(_sub_elements[i]->space_dimension());
      }

      return [dims, sub_element_functions](std::span<std::int32_t> doflist,
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
      // Blocked element
      std::function<void(std::span<std::int32_t>, std::uint32_t)>
          sub_element_function = _sub_elements[0]->dof_permutation_fn(inverse);
      int dim = _sub_elements[0]->space_dimension();
      int bs = _bs;
      return
          [sub_element_function, bs, subdofs = std::vector<std::int32_t>(dim)](
              std::span<std::int32_t> doflist,
              std::uint32_t cell_permutation) mutable
      {
        for (int k = 0; k < bs; ++k)
        {
          for (std::size_t i = 0; i < subdofs.size(); ++i)
            subdofs[i] = doflist[bs * i + k];
          sub_element_function(subdofs, cell_permutation);
          for (std::size_t i = 0; i < subdofs.size(); ++i)
            doflist[bs * i + k] = subdofs[i];
        }
      };
    }
  }

  if (inverse)
  {
    return
        [this](std::span<std::int32_t> doflist, std::uint32_t cell_permutation)
    { permute_inv(doflist, cell_permutation); };
  }
  else
  {
    return
        [this](std::span<std::int32_t> doflist, std::uint32_t cell_permutation)
    { permute(doflist, cell_permutation); };
  }
}
//-----------------------------------------------------------------------------
template class fem::FiniteElement<float>;
template class fem::FiniteElement<double>;
//-----------------------------------------------------------------------------
