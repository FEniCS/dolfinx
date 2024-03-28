#pragma once

#include <basix/mdspan.hpp>

namespace dolfinx
{

template<class ElementType, class Extents, class LayoutPolicy=std::layout_right, class AccessorPolicy=std::default_accessor<ElementType>>
using mdspan = typename MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<ElementType, Extents, LayoutPolicy, AccessorPolicy>;

template<class IndexType, std::size_t... Extents>
using extents = typename MDSPAN_IMPL_STANDARD_NAMESPACE::extents<IndexType, Extents...>;

template<class IndexType, std::size_t Rank>
using dextents = typename MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<IndexType, Rank>;

}
