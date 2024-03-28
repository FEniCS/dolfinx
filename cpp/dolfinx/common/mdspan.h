#pragma once

#include <basix/mdspan.hpp>

namespace dolfinx
{

template<class ElementType, class Extents, class LayoutPolicy=std::layout_right, class AccessorPolicy=std::default_accessor<ElementType>>
using mdspan = typename MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<ElementType, Extents, LayoutPolicy, AccessorPolicy>;

}
