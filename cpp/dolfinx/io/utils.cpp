#include "utils.h"
#include <dolfinx/mesh/Topology.h>

using namespace dolfinx;

//-----------------------------------------------------------------------------
/// @cond
template std::pair<std::vector<std::int32_t>, std::vector<double>>
io::distribute_entity_data(
    const mesh::Topology&, std::span<const std::int64_t>, std::int64_t,
    const fem::ElementDofLayout&,
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        const std::int32_t,
        MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>,
    int,
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        const std::int64_t,
        MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>,
    std::span<const double>);
template std::pair<std::vector<std::int32_t>, std::vector<float>>
io::distribute_entity_data(
    const mesh::Topology&, std::span<const std::int64_t>, std::int64_t,
    const fem::ElementDofLayout&,
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        const std::int32_t,
        MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>,
    int,
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        const std::int64_t,
        MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>,
    std::span<const float>);
template std::pair<std::vector<std::int32_t>, std::vector<std::int32_t>>
io::distribute_entity_data(
    const mesh::Topology&, std::span<const std::int64_t>, std::int64_t,
    const fem::ElementDofLayout&,
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        const std::int32_t,
        MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>,
    int,
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        const std::int64_t,
        MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>,
    std::span<const std::int32_t>);
template std::pair<std::vector<std::int32_t>, std::vector<std::int64_t>>
io::distribute_entity_data(
    const mesh::Topology&, std::span<const std::int64_t>, std::int64_t,
    const fem::ElementDofLayout&,
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        const std::int32_t,
        MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>,
    int,
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        const std::int64_t,
        MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>,
    std::span<const std::int64_t>);
template std::pair<std::vector<std::int32_t>, std::vector<std::complex<double>>>
io::distribute_entity_data(
    const mesh::Topology&, std::span<const std::int64_t>, std::int64_t,
    const fem::ElementDofLayout&,
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        const std::int32_t,
        MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>,
    int,
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        const std::int64_t,
        MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>,
    std::span<const std::complex<double>>);
template std::pair<std::vector<std::int32_t>, std::vector<std::complex<float>>>
io::distribute_entity_data(
    const mesh::Topology&, std::span<const std::int64_t>, std::int64_t,
    const fem::ElementDofLayout&,
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        const std::int32_t,
        MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>,
    int,
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        const std::int64_t,
        MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>,
    std::span<const std::complex<float>>);
template std::pair<std::vector<std::int32_t>, std::vector<std::complex<float>>>
io::distribute_entity_data(
    const mesh::Topology&, std::span<const std::int64_t>, std::int64_t,
    const fem::ElementDofLayout&,
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        const std::int32_t,
        MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>,
    int,
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        const std::int64_t,
        MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>,
    std::span<const std::complex<float>>);
/// @endcond
