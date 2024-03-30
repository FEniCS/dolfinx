#pragma once

#include <basix/mdspan.hpp>

namespace dolfinx::common::mdspan
{

template <class ElementType, class Extents,
          class LayoutPolicy = std::layout_right,
          class AccessorPolicy = std::default_accessor<ElementType>>
using mdspan = typename MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
    ElementType, Extents, LayoutPolicy, AccessorPolicy>;

template <class IndexType, std::size_t... Extents>
using extents =
    typename MDSPAN_IMPL_STANDARD_NAMESPACE::extents<IndexType, Extents...>;

template <class IndexType, std::size_t Rank>
using dextents =
    typename MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<IndexType, Rank>;

constexpr auto full_extent = MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent;

#if defined(__cpp_lib_span)
using MDSPAN_IMPL_STANDARD_NAMESPACE::dynamic_extent;
#elif
constexpr auto dynamic_extent = MDSPAN_IMPL_STANDARD_NAMESPACE::dynamic_extent
#endif

template <class ElementType, class Extents, class LayoutPolicy,
          class AccessorPolicy, class... SliceSpecifiers>
auto submdspan(
    const mdspan<ElementType, Extents, LayoutPolicy, AccessorPolicy>& src,
    SliceSpecifiers... slices)
{
  return MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan<
      ElementType, Extents, LayoutPolicy, AccessorPolicy, SliceSpecifiers...>(
      src, std::forward<SliceSpecifiers>(slices)...);
}

// this does not allow for auto template type deduction, thus the previous one
// is used. template <class ElementType, class Extents, class LayoutPolicy,
// class AccessorPolicy, class... SliceSpecifiers> constexpr auto submdspan =
// MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan<ElementType, Extents, LayoutPolicy,
// AccessorPolicy, SliceSpecifiers...>;

} // namespace dolfinx::common::mdspan
