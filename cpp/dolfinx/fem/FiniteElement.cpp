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
#include <dolfinx/common/mdspan.h>
#include <functional>
#include <numeric>
#include <ufcx.h>
#include <utility>
#include <vector>

using namespace dolfinx;
using namespace dolfinx::fem;

namespace
{
//-----------------------------------------------------------------------------

/// Check if an element is a Basix element (or a blocked element
/// containing a Basix element)
bool is_basix_element(const ufcx_finite_element& element)
{
  if (element.element_type == ufcx_basix_element)
    return true;
  else if (element.block_size != 1)
  {
    // TODO: what should happen if the element is a blocked element
    // containing a blocked element containing a Basix element?
    return element.sub_elements[0]->element_type == ufcx_basix_element;
  }
  else
    return false;
}
//-----------------------------------------------------------------------------

/// Check if an element is a quadrature element (or a blocked element
/// containing a quadrature element)
bool is_quadrature_element(const ufcx_finite_element& element)
{
  if (element.element_type == ufcx_quadrature_element)
    return true;
  else if (element.block_size != 1)
  {
    // TODO: what should happen if the element is a blocked element
    // containing a blocked element containing a quadrature element?
    return element.sub_elements[0]->element_type == ufcx_quadrature_element;
  }
  else
    return false;
}
//-----------------------------------------------------------------------------

/// Check if an element is a custom Basix element (or a blocked element
/// containing a custom Basix element)
bool is_basix_custom_element(const ufcx_finite_element& element)
{
  if (element.element_type == ufcx_basix_custom_element)
    return true;
  else if (element.block_size != 1)
  {
    // TODO: what should happen if the element is a blocked element
    // containing a blocked element containing a Basix element?
    return element.sub_elements[0]->element_type == ufcx_basix_custom_element;
  }
  else
    return false;
}
//-----------------------------------------------------------------------------

/// Recursively extract sub finite element
template <std::floating_point T>
std::shared_ptr<const FiniteElement<T>>
_extract_sub_element(const FiniteElement<T>& finite_element,
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
//-----------------------------------------------------------------------------

} // namespace

//-----------------------------------------------------------------------------
template <std::floating_point T>
FiniteElement<T>::FiniteElement(const ufcx_finite_element& e)
    : _signature(e.signature), _space_dim(e.space_dimension),
      _reference_value_shape(e.reference_value_shape,
                             e.reference_value_shape + e.reference_value_rank),
      _bs(e.block_size), _is_mixed(e.element_type == ufcx_mixed_element)
{
  const ufcx_shape _shape = e.cell_shape;
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
  assert(mesh::cell_dim(_cell_shape) == e.topological_dimension);

  static const std::map<ufcx_shape, std::string> ufcx_to_cell
      = {{vertex, "point"},         {interval, "interval"},
         {triangle, "triangle"},    {tetrahedron, "tetrahedron"},
         {prism, "prism"},          {quadrilateral, "quadrilateral"},
         {hexahedron, "hexahedron"}};
  const std::string cell_shape = ufcx_to_cell.at(e.cell_shape);

  _needs_dof_transformations = false;
  _needs_dof_permutations = false;
  // Create all sub-elements
  for (int i = 0; i < e.num_sub_elements; ++i)
  {
    ufcx_finite_element* ufcx_sub_element = e.sub_elements[i];
    _sub_elements.push_back(
        std::make_shared<FiniteElement<T>>(*ufcx_sub_element));
    if (_sub_elements[i]->needs_dof_permutations())
      _needs_dof_permutations = true;
    if (_sub_elements[i]->needs_dof_transformations())
      _needs_dof_transformations = true;
  }

  if (is_basix_custom_element(e))
  {
    // Recreate the custom Basix element using information written into
    // the generated code
    ufcx_basix_custom_finite_element* ce = e.custom_element;
    const basix::cell::type cell_type
        = static_cast<basix::cell::type>(ce->cell_type);

    const std::vector<std::size_t> value_shape(
        ce->value_shape, ce->value_shape + ce->value_shape_length);
    const std::size_t value_size = std::reduce(
        value_shape.begin(), value_shape.end(), 1, std::multiplies{});

    const int nderivs = ce->interpolation_nderivs;
    const std::size_t nderivs_dim = basix::polyset::nderivs(cell_type, nderivs);

    using array2_t = std::pair<std::vector<T>, std::array<std::size_t, 2>>;
    using array4_t = std::pair<std::vector<T>, std::array<std::size_t, 4>>;
    std::array<std::vector<array2_t>, 4> x;
    std::array<std::vector<array4_t>, 4> M;
    { // scope
      int pt_n = 0;
      int p_e = 0;
      int m_e = 0;
      const std::size_t dim = static_cast<std::size_t>(
          basix::cell::topological_dimension(cell_type));
      for (std::size_t d = 0; d <= dim; ++d)
      {
        const int num_entities = basix::cell::num_sub_entities(cell_type, d);
        for (int entity = 0; entity < num_entities; ++entity)
        {
          std::size_t npts = ce->npts[pt_n + entity];
          std::size_t ndofs = ce->ndofs[pt_n + entity];

          std::array pshape = {npts, dim};
          auto& pts
              = x[d].emplace_back(std::vector<T>(pshape[0] * pshape[1]), pshape)
                    .first;
          std::copy_n(ce->x + p_e, pts.size(), pts.begin());
          p_e += pts.size();

          std::array mshape = {ndofs, value_size, npts, nderivs_dim};
          std::size_t msize
              = std::reduce(mshape.begin(), mshape.end(), 1, std::multiplies{});
          auto& mat = M[d].emplace_back(std::vector<T>(msize), mshape).first;
          std::copy_n(ce->M + m_e, mat.size(), mat.begin());
          m_e += mat.size();
        }

        pt_n += num_entities;
      }
    }

    using cmdspan2_t = dolfinx::common::mdspan::mdspan<
        const T, dolfinx::common::mdspan::dextents<std::size_t, 2>>;
    using cmdspan4_t = dolfinx::common::mdspan::mdspan<
        const T, dolfinx::common::mdspan::dextents<std::size_t, 4>>;

    std::array<std::vector<cmdspan2_t>, 4> _x;
    for (std::size_t i = 0; i < x.size(); ++i)
      for (auto& [buffer, shape] : x[i])
        _x[i].push_back(cmdspan2_t(buffer.data(), shape));

    std::array<std::vector<cmdspan4_t>, 4> _M;
    for (std::size_t i = 0; i < M.size(); ++i)
      for (auto& [buffer, shape] : M[i])
        _M[i].push_back(cmdspan4_t(buffer.data(), shape));

    std::vector<T> wcoeffs_b(ce->wcoeffs_rows * ce->wcoeffs_cols);
    cmdspan2_t wcoeffs(wcoeffs_b.data(), ce->wcoeffs_rows, ce->wcoeffs_cols);
    std::copy_n(ce->wcoeffs, wcoeffs_b.size(), wcoeffs_b.begin());

    _element = std::make_unique<basix::FiniteElement<T>>(
        basix::create_custom_element(
            cell_type, value_shape, wcoeffs, _x, _M, nderivs,
            static_cast<basix::maps::type>(ce->map_type),
            static_cast<basix::sobolev::space>(ce->sobolev_space),
            ce->discontinuous, ce->embedded_subdegree, ce->embedded_superdegree,
            static_cast<basix::polyset::type>(ce->polyset_type)));
    _needs_dof_transformations
        = !_element->dof_transformations_are_identity()
          and !_element->dof_transformations_are_permutations();

    _needs_dof_permutations
        = !_element->dof_transformations_are_identity()
          and _element->dof_transformations_are_permutations();
  }
  else if (is_basix_element(e))
  {
    _element
        = std::make_unique<basix::FiniteElement<T>>(basix::create_element<T>(
            static_cast<basix::element::family>(e.basix_family),
            static_cast<basix::cell::type>(e.basix_cell), e.degree,
            static_cast<basix::element::lagrange_variant>(e.lagrange_variant),
            static_cast<basix::element::dpc_variant>(e.dpc_variant),
            e.discontinuous));
    _needs_dof_transformations
        = !_element->dof_transformations_are_identity()
          and !_element->dof_transformations_are_permutations();
    _needs_dof_permutations
        = !_element->dof_transformations_are_identity()
          and _element->dof_transformations_are_permutations();
  }

  if (is_quadrature_element(e))
  {
    assert(e.custom_quadrature);
    ufcx_quadrature_rule* qr = e.custom_quadrature;
    std::size_t npts = qr->npts;
    std::size_t tdim = qr->topological_dimension;
    std::array qshape = {npts, tdim};
    _points = std::make_pair(std::vector<T>(qshape[0] * qshape[1]), qshape);
    std::copy_n(qr->points, _points.first.size(), _points.first.begin());
  }
}
//-----------------------------------------------------------------------------
template <std::floating_point T>
FiniteElement<T>::FiniteElement(const basix::FiniteElement<T>& element,
                                const std::size_t block_size)
    : _reference_value_shape(element.value_shape()), _bs(block_size),
      _is_mixed(false)
{
  _space_dim = _bs * element.dim();

  // Create all sub-elements
  if (_bs > 1)
  {
    for (int i = 0; i < _bs; ++i)
    {
      _sub_elements.push_back(std::make_shared<FiniteElement<T>>(element, 1));
    }
    _reference_value_shape = {block_size};
  }

  _element = std::make_unique<basix::FiniteElement<T>>(element);
  assert(_element);
  _needs_dof_transformations
      = !_element->dof_transformations_are_identity()
        and !_element->dof_transformations_are_permutations();
  _needs_dof_permutations
      = !_element->dof_transformations_are_identity()
        and _element->dof_transformations_are_permutations();
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
std::string FiniteElement<T>::signature() const noexcept
{
  return _signature;
}
//-----------------------------------------------------------------------------
template <std::floating_point T>
mesh::CellType FiniteElement<T>::cell_shape() const noexcept
{
  return _cell_shape;
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
  return _reference_value_shape;
}
//-----------------------------------------------------------------------------
template <std::floating_point T>
int FiniteElement<T>::reference_value_size() const
{
  return std::accumulate(_reference_value_shape.begin(),
                         _reference_value_shape.end(), 1, std::multiplies{});
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
  return _is_mixed;
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
  DLOG(INFO) << "Extracted finite element for sub-system: "
             << sub_finite_element->signature().c_str();
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
void FiniteElement<T>::permute_dofs(std::span<std::int32_t> doflist,
                                    std::uint32_t cell_permutation) const
{
  _element->permute_dofs(doflist, cell_permutation);
}
//-----------------------------------------------------------------------------
template <std::floating_point T>
void FiniteElement<T>::unpermute_dofs(std::span<std::int32_t> doflist,
                                      std::uint32_t cell_permutation) const
{
  _element->unpermute_dofs(doflist, cell_permutation);
}
//-----------------------------------------------------------------------------
/// @cond
template <std::floating_point T>
std::function<void(std::span<std::int32_t>, std::uint32_t)>
FiniteElement<T>::get_dof_permutation_function(bool inverse,
                                               bool scalar_element) const
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
            _sub_elements[i]->get_dof_permutation_function(inverse));
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
      // Vector element
      std::function<void(std::span<std::int32_t>, std::uint32_t)>
          sub_element_function
          = _sub_elements[0]->get_dof_permutation_function(inverse);
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
    { unpermute_dofs(doflist, cell_permutation); };
  }
  else
  {
    return
        [this](std::span<std::int32_t> doflist, std::uint32_t cell_permutation)
    { permute_dofs(doflist, cell_permutation); };
  }
}
//-----------------------------------------------------------------------------
template class fem::FiniteElement<float>;
template class fem::FiniteElement<double>;
//-----------------------------------------------------------------------------
