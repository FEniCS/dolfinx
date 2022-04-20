// Copyright (C) 2020-2021 Garth N. Wells and Matthew W. Scroggs
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "FiniteElement.h"
#include <algorithm>
#include <basix/finite-element.h>
#include <basix/interpolation.h>
#include <dolfinx/common/log.h>
#include <functional>
#include <ufcx.h>

using namespace dolfinx;
using namespace dolfinx::fem;

namespace
{
//-----------------------------------------------------------------------------
// Check if an element is a Basix element (or a blocked element
// containing a Basix element)
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
// Check if an element is a custom Basix element (or a blocked element
// containing a custom Basix element)
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
FiniteElement::FiniteElement(const ufcx_finite_element& e)
    : _signature(e.signature), _family(e.family), _space_dim(e.space_dimension),
      _value_shape(e.value_shape, e.value_shape + e.value_rank),
      _bs(e.block_size)
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
    _sub_elements.push_back(std::make_shared<FiniteElement>(*ufcx_sub_element));
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

  if (is_basix_custom_element(e))
  {
    // Recreate the custom Basix element using information written into the
    // generated code
    ufcx_basix_custom_finite_element* ce = e.custom_element;
    const basix::cell::type cell_type
        = static_cast<basix::cell::type>(ce->cell_type);

    std::vector<std::size_t> value_shape(ce->value_shape_length);
    int value_size = 1;
    for (int i = 0; i < ce->value_shape_length; ++i)
    {
      value_shape[i] = ce->value_shape[i];
      value_size *= ce->value_shape[i];
    }

    xt::xtensor<double, 2> wcoeffs(
        {static_cast<std::size_t>(ce->wcoeffs_rows),
         static_cast<std::size_t>(ce->wcoeffs_cols)});
    { // scope
      int e = 0;
      for (int i = 0; i < ce->wcoeffs_rows; ++i)
        for (int j = 0; j < ce->wcoeffs_cols; ++j)
          wcoeffs(i, j) = ce->wcoeffs[e++];
    }

    std::array<std::vector<xt::xtensor<double, 2>>, 4> x;
    std::array<std::vector<xt::xtensor<double, 3>>, 4> M;
    { // scope
      int pt_n = 0;
      int p_e = 0;
      int m_e = 0;
      const int dim = basix::cell::topological_dimension(cell_type);
      for (int d = 0; d <= dim; ++d)
      {
        const int num_entities = basix::cell::num_sub_entities(cell_type, d);
        for (int entity = 0; entity < num_entities; ++entity)
        {
          const int npts = ce->npts[pt_n++];
          xt::xtensor<double, 2> pts(
              {static_cast<std::size_t>(npts), static_cast<std::size_t>(dim)});
          xt::xtensor<double, 3> mat({static_cast<std::size_t>(npts),
                                      static_cast<std::size_t>(value_size),
                                      static_cast<std::size_t>(npts)});
          for (int i = 0; i < npts; ++i)
            for (int j = 0; j < dim; ++j)
              pts(i, j) = ce->x[p_e++];
          for (int i = 0; i < npts; ++i)
            for (int j = 0; j < value_size; ++j)
              for (int k = 0; k < npts; ++k)
                mat(i, j, k) = ce->M[m_e++];
          x[d].push_back(pts);
          M[d].push_back(mat);
        }
      }
    }

    _element
        = std::make_unique<basix::FiniteElement>(basix::create_custom_element(
            cell_type, ce->degree, value_shape, wcoeffs, x, M,
            static_cast<basix::maps::type>(ce->map_type), ce->discontinuous,
            ce->highest_complete_degree));
  }
  else if (is_basix_element(e))
  {
    if (e.lagrange_variant != -1 and e.dpc_variant != -1)
    {
      _element = std::make_unique<basix::FiniteElement>(basix::create_element(
          static_cast<basix::element::family>(e.basix_family),
          static_cast<basix::cell::type>(e.basix_cell), e.degree,
          static_cast<basix::element::lagrange_variant>(e.lagrange_variant),
          static_cast<basix::element::dpc_variant>(e.dpc_variant),
          e.discontinuous));
    }
    else if (e.lagrange_variant != -1)
    {
      _element = std::make_unique<basix::FiniteElement>(basix::create_element(
          static_cast<basix::element::family>(e.basix_family),
          static_cast<basix::cell::type>(e.basix_cell), e.degree,
          static_cast<basix::element::lagrange_variant>(e.lagrange_variant),
          e.discontinuous));
    }
    else if (e.dpc_variant != -1)
    {
      _element = std::make_unique<basix::FiniteElement>(basix::create_element(
          static_cast<basix::element::family>(e.basix_family),
          static_cast<basix::cell::type>(e.basix_cell), e.degree,
          static_cast<basix::element::dpc_variant>(e.dpc_variant),
          e.discontinuous));
    }
    else
    {
      _element = std::make_unique<basix::FiniteElement>(basix::create_element(
          static_cast<basix::element::family>(e.basix_family),
          static_cast<basix::cell::type>(e.basix_cell), e.degree,
          e.discontinuous));
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
FiniteElement::FiniteElement(const basix::FiniteElement& element, int bs)
    : _space_dim(bs * element.dim()), _value_shape(element.value_shape()),
      _bs(bs)
{
  if (_value_shape.empty() and bs > 1)
    _value_shape = {1};
  std::transform(_value_shape.cbegin(), _value_shape.cend(),
                 _value_shape.begin(), [bs](auto s) { return bs * s; });

  if (bs > 1)
  {
    // Create all sub-elements
    for (int i = 0; i < bs; ++i)
      _sub_elements.push_back(std::make_shared<FiniteElement>(element, 1));
  }

  _element = std::make_unique<basix::FiniteElement>(element);
  assert(_element);
  _needs_dof_transformations
      = !_element->dof_transformations_are_identity()
        and !_element->dof_transformations_are_permutations();

  _needs_dof_permutations
      = !_element->dof_transformations_are_identity()
        and _element->dof_transformations_are_permutations();
  switch (_element->family())
  {
  case basix::element::family::P:
    _family = "Lagrange";
    break;
  case basix::element::family::DPC:
    _family = "Discontinuous Lagrange";
    break;
  default:
    _family = "unknown";
    break;
  }

  _signature = "Basix element " + _family + " " + std::to_string(bs);
}
//-----------------------------------------------------------------------------
bool FiniteElement::operator==(const FiniteElement& e) const
{
  if (!_element or !e._element)
  {
    throw std::runtime_error(
        "Missing a Basix element. Cannot check for equivalence");
  }
  return *_element == *e._element;
}
//-----------------------------------------------------------------------------
bool FiniteElement::operator!=(const FiniteElement& e) const
{
  return !(*this == e);
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
int FiniteElement::value_size() const
{
  return std::accumulate(_value_shape.begin(), _value_shape.end(), 1,
                         std::multiplies<int>());
}
//-----------------------------------------------------------------------------
int FiniteElement::reference_value_size() const
{
  return std::accumulate(_value_shape.begin(), _value_shape.end(), 1,
                         std::multiplies<int>());
}
//-----------------------------------------------------------------------------
int FiniteElement::block_size() const noexcept { return _bs; }
//-----------------------------------------------------------------------------
xtl::span<const int> FiniteElement::value_shape() const noexcept
{
  return _value_shape;
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
bool FiniteElement::is_mixed() const noexcept
{
  return !_sub_elements.empty() and _bs == 1;
}
//-----------------------------------------------------------------------------
const std::vector<std::shared_ptr<const FiniteElement>>&
FiniteElement::sub_elements() const noexcept
{
  return _sub_elements;
}
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
basix::maps::type FiniteElement::map_type() const
{
  if (!_element)
  {
    throw std::runtime_error("Cannot element map type - no Basix element "
                             "available. Maybe this is a mixed element?");
  }

  return _element->map_type();
}
//-----------------------------------------------------------------------------
bool FiniteElement::map_ident() const noexcept
{
  assert(_element);
  return _element->map_type() == basix::maps::type::identity;
}
//-----------------------------------------------------------------------------
bool FiniteElement::interpolation_ident() const noexcept
{
  assert(_element);
  return _element->interpolation_is_identity();
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
const xt::xtensor<double, 2>& FiniteElement::interpolation_operator() const
{
  if (!_element)
  {
    throw std::runtime_error("No underlying element for interpolation. "
                             "Cannot interpolate mixed elements directly.");
  }

  return _element->interpolation_matrix();
}
//-----------------------------------------------------------------------------
xt::xtensor<double, 2>
FiniteElement::create_interpolation_operator(const FiniteElement& from) const
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
      // If this element shouldn't be permuted but needs
      // transformations, return a function that throws an error
      return [](const xtl::span<std::int32_t>&, std::uint32_t)
      {
        throw std::runtime_error(
            "Permutations should not be applied for this element.");
      };
    }
  }

  if (!_sub_elements.empty())
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
      // Vector element
      std::function<void(const xtl::span<std::int32_t>&, std::uint32_t)>
          sub_element_function
          = _sub_elements[0]->get_dof_permutation_function(inverse);
      int dim = _sub_elements[0]->space_dimension();
      int bs = _bs;
      return
          [sub_element_function, bs, subdofs = std::vector<std::int32_t>(dim)](
              const xtl::span<std::int32_t>& doflist,
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
