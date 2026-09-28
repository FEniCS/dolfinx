// Copyright (C) 2026 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

// Tests that drive the assembly implementation kernels directly rather
// than through fem::assemble_*. The kernels are local loops over
// caller-supplied arrays, so the data is synthetic and identical on
// every rank: no mesh, no communication.
//
// Covered are the cases a form-level test cannot reach conveniently:
// block sizes other than one, an integration domain whose cell list
// differs from the argument's, and an active dof transformation.

#include <array>
#include <basix/mdspan.hpp>
#include <catch2/catch_test_macros.hpp>
#include <concepts>
#include <cstdint>
#include <dolfinx/common/types.h>
// fem/assembler.h, not the assemble_*_impl.h headers directly: those are
// not self-contained (fem/Function.h and fem/assembler.h include one
// another), and assembler.h includes them in a working order.
#include <dolfinx/fem/assembler.h>
#include <functional>
#include <map>
#include <ranges>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

using namespace dolfinx;

namespace
{
using T = double;

/// Dofmap view with three entries per cell carried in the type.
using dofmap3_t = md::mdspan<const std::int32_t,
                             md::extents<std::size_t, md::dynamic_extent, 3>>;

/// Two P1 triangles embedded in 3D, sharing the edge (1, 2).
constexpr std::array<std::int32_t, 6> x_dofs = {0, 1, 2, 1, 3, 2};
constexpr std::array<T, 12> x_coords = {0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0};

/// Test/trial dofmap. Cell 1 is not the identity, so a scatter using
/// the integration cell is caught.
constexpr std::array<std::int32_t, 6> dofs = {0, 1, 2, 3, 2, 1};
constexpr std::size_t num_cells = 2;
constexpr std::size_t num_dofs_cell = 3;
constexpr std::size_t num_dofs = 4;

/// Per-cell permutation words, distinct so a transformation handed the
/// wrong cell index gives the wrong answer.
constexpr std::array<std::uint32_t, 2> cell_info = {2, 5};

auto geometry_pack()
{
  return fem::GeometryPack{
      dofmap3_t(x_dofs.data(), num_cells),
      md::mdspan<const T, md::extents<std::size_t, md::dynamic_extent, 3>>(
          x_coords.data(), x_coords.size() / 3, 3)};
}

/// Element vector entry for local dof `i`, component `k`, on a cell with
/// gathered coordinates `cdofs`. Coordinate-dependent, so a wrong
/// geometry gather is caught.
constexpr T be_value(const T* cdofs, std::size_t i, int k)
{
  return 100 * cdofs[3 * i] + 10 * cdofs[3 * i + 1] + (k + 1);
}

/// Kernel writing `be_value` into a `(num_dofs_cell, bs)` element vector.
template <int bs>
void vector_kernel(T* be, const T*, const T*, const T* cdofs, const int*,
                   const std::uint8_t*, void*)
{
  for (std::size_t i = 0; i < num_dofs_cell; ++i)
    for (int k = 0; k < bs; ++k)
      be[bs * i + k] = be_value(cdofs, i, k);
}

/// Reference result for assembling `vector_kernel` over `cells`,
/// scattering through `cells0`. Written independently of the assembler.
template <int bs>
std::vector<T> reference_vector(std::span<const std::int32_t> cells,
                                std::span<const std::int32_t> cells0,
                                std::span<const std::uint32_t> info)
{
  std::vector<T> b(bs * num_dofs, 0);
  for (std::size_t e = 0; e < cells.size(); ++e)
  {
    // Gather the cell coordinates from the integration domain cell.
    std::array<T, 3 * num_dofs_cell> cdofs;
    for (std::size_t i = 0; i < num_dofs_cell; ++i)
    {
      const std::int32_t node = x_dofs[num_dofs_cell * cells[e] + i];
      for (std::size_t c = 0; c < 3; ++c)
        cdofs[3 * i + c] = x_coords[3 * node + c];
    }

    // Scatter through the argument's cell.
    const std::int32_t cell0 = cells0[e];
    for (std::size_t i = 0; i < num_dofs_cell; ++i)
    {
      const std::int32_t dof = dofs[num_dofs_cell * cell0 + i];
      for (int k = 0; k < bs; ++k)
      {
        const T scale = info.empty() ? 1 : static_cast<T>(info[cell0]);
        b[bs * dof + k] += scale * be_value(cdofs.data(), i, k);
      }
    }
  }

  return b;
}

/// Assemble `vector_kernel` with the given block size, cell lists and
/// transformation, and return the result.
template <int bs>
std::vector<T>
assemble(const fem::IndexList auto& cells, const fem::IndexList auto& cells0,
         const auto& transform, std::span<const std::uint32_t> info)
{
  std::vector<T> b(bs * num_dofs, 0);
  std::array<T, bs * num_dofs_cell> be_b;
  std::array<T, 3 * num_dofs_cell> cdofs_b;
  fem::impl::assemble_cells_vector(
      b, geometry_pack(), cells,
      fem::FormArgument{fem::DofMapPack{dofmap3_t(dofs.data(), num_cells),
                                        std::integral_constant<int, bs>{},
                                        cells0},
                        transform, info},
      vector_kernel<bs>, {}, {}, std::span<T>(be_b), std::span<T>(cdofs_b));
  return b;
}

/// Scales the element tensor by the cell's permutation word, making the
/// cell index the assembler passes observable in the result.
void scale_by_cell_info(std::span<T> be, std::span<const std::uint32_t> info,
                        std::int32_t cell, int)
{
  for (T& v : be)
    v *= static_cast<T>(info[cell]);
}
} // namespace

// The kernels fold the element tensor offsets only if the block size
// and per-cell dofmap length survive the unpack as compile-time
// constants. Reading either through a reference -- as a structured
// binding of a tuple-like type does -- loses that silently, so pin it
// here where the compiler can check.
namespace
{
using pack_t = fem::DofMapPack<dofmap3_t, std::integral_constant<int, 3>,
                               std::span<const std::int32_t>>;
static_assert(std::same_as<decltype(std::declval<const pack_t&>().bs),
                           std::integral_constant<int, 3>>);
static_assert(decltype(std::declval<const pack_t&>().map)::static_extent(1)
              == 3);
} // namespace

TEST_CASE("Assembly kernel block size dispatch", "[assembly_kernels]")
{
  // 1 and 3 are handed to the callable as compile-time constants; any
  // other block size takes the run-time path.
  auto bs_of = [](int bs)
  {
    int value = -1;
    bool is_constant = false;
    fem::impl::dispatch_bs(bs,
                           [&value, &is_constant](auto b)
                           {
                             value = b;
                             is_constant = !std::same_as<decltype(b), int>;
                           });
    return std::pair{value, is_constant};
  };

  CHECK(bs_of(1) == std::pair{1, true});
  CHECK(bs_of(2) == std::pair{2, false});
  CHECK(bs_of(3) == std::pair{3, true});
  CHECK(bs_of(7) == std::pair{7, false});

  // Only matching block sizes are specialised.
  auto bs_pair_of = [](int bs0, int bs1)
  {
    std::array<int, 2> value = {-1, -1};
    bool is_constant = false;
    fem::impl::dispatch_bs(bs0, bs1,
                           [&value, &is_constant](auto b0, auto b1)
                           {
                             value = {b0, b1};
                             is_constant = !std::same_as<decltype(b0), int>;
                           });
    return std::pair{value, is_constant};
  };

  CHECK(bs_pair_of(1, 1) == std::pair{std::array{1, 1}, true});
  CHECK(bs_pair_of(2, 2) == std::pair{std::array{2, 2}, false});
  CHECK(bs_pair_of(3, 3) == std::pair{std::array{3, 3}, true});
  CHECK(bs_pair_of(3, 1) == std::pair{std::array{3, 1}, false});
  CHECK(bs_pair_of(1, 3) == std::pair{std::array{1, 3}, false});
}

TEST_CASE("Assemble cells into a vector (block sizes)", "[assembly_kernels]")
{
  constexpr std::array<std::int32_t, 2> cells = {0, 1};
  std::span<const std::int32_t> c(cells);
  auto none = std::span<const std::uint32_t>();
  auto unset = std::function<void(std::span<T>, std::span<const std::uint32_t>,
                                  std::int32_t, int)>();

  // Block size 1 and 3 take the compile-time path in the drivers, 2 the
  // run-time one; all three must give the same answer.
  CHECK(assemble<1>(c, c, unset, none) == reference_vector<1>(c, c, none));
  CHECK(assemble<2>(c, c, unset, none) == reference_vector<2>(c, c, none));
  CHECK(assemble<3>(c, c, unset, none) == reference_vector<3>(c, c, none));
}

TEST_CASE("Assemble cells into a vector (argument cell list differs)",
          "[assembly_kernels]")
{
  // The integration and test function domains number cells differently,
  // so geometry is gathered through `cells` and the result scattered
  // through `cells0`.
  constexpr std::array<std::int32_t, 2> cells = {0, 1};
  constexpr std::array<std::int32_t, 2> cells0 = {1, 0};
  std::span<const std::int32_t> c(cells), c0(cells0);
  auto none = std::span<const std::uint32_t>();
  auto unset = std::function<void(std::span<T>, std::span<const std::uint32_t>,
                                  std::int32_t, int)>();

  CHECK(assemble<2>(c, c0, unset, none) == reference_vector<2>(c, c0, none));

  // The swapped list must not give the same answer as the matched one,
  // or the check above would pass for an assembler that ignored cells0.
  CHECK(assemble<2>(c, c0, unset, none) != assemble<2>(c, c, unset, none));

  // The argument list need not have the same range type or contiguity as the
  // integration-domain list.
  auto generated_cells0
      = std::views::iota(std::int32_t(0), std::int32_t(cells0.size()));
  CHECK(assemble<2>(c, generated_cells0, unset, none)
        == reference_vector<2>(c, c, none));
}

TEST_CASE("Assemble cells into a vector (dof transformation)",
          "[assembly_kernels]")
{
  constexpr std::array<std::int32_t, 2> cells = {0, 1};
  constexpr std::array<std::int32_t, 2> cells0 = {1, 0};
  std::span<const std::int32_t> c(cells), c0(cells0);
  std::span<const std::uint32_t> info(cell_info);

  // An active transformation is applied once per cell, with the
  // argument's cell index and the argument's permutation data.
  CHECK(assemble<2>(c, c0, scale_by_cell_info, info)
        == reference_vector<2>(c, c0, info));

  // A transformation handed the integration cell instead of the
  // argument cell would produce this, which must differ.
  CHECK(assemble<2>(c, c0, scale_by_cell_info, info)
        != reference_vector<2>(c, c, info));

  // A null std::function is "no transformation", not a call through a
  // null target.
  auto unset = std::function<void(std::span<T>, std::span<const std::uint32_t>,
                                  std::int32_t, int)>();
  auto none = std::span<const std::uint32_t>();
  CHECK(assemble<2>(c, c0, unset, info) == reference_vector<2>(c, c0, none));
}

TEST_CASE("Assemble cells into a matrix (block sizes and bcs)",
          "[assembly_kernels]")
{
  constexpr std::array<std::int32_t, 2> cells = {0, 1};
  std::span<const std::int32_t> c(cells);
  auto none = std::span<const std::uint32_t>();
  auto unset = std::function<void(std::span<T>, std::span<const std::uint32_t>,
                                  std::int32_t, int)>();

  constexpr int bs = 2;
  constexpr std::size_t ndim = bs * num_dofs_cell;

  // Element matrix with a distinct value in every entry.
  auto kernel = [](T* Ae, const T*, const T*, const T* cdofs, const int*,
                   const std::uint8_t*, void*)
  {
    for (std::size_t r = 0; r < ndim; ++r)
      for (std::size_t c1 = 0; c1 < ndim; ++c1)
        Ae[ndim * r + c1] = 100 * cdofs[0] + 10 * r + c1 + 1;
  };

  // Collect the inserted blocks rather than build a matrix, so the rows,
  // columns and values handed to `mat_set` are visible.
  auto collect = [](std::map<std::pair<std::int32_t, std::int32_t>, T>& out)
  {
    return [&out](std::span<const std::int32_t> r,
                  std::span<const std::int32_t> c1, std::span<const T> vals)
    {
      const std::size_t nc = bs * c1.size();
      for (std::size_t i = 0; i < bs * r.size(); ++i)
      {
        for (std::size_t j = 0; j < nc; ++j)
        {
          const std::int32_t row = bs * r[i / bs] + static_cast<int>(i % bs);
          const std::int32_t col = bs * c1[j / bs] + static_cast<int>(j % bs);
          out[{row, col}] += vals[nc * i + j];
        }
      }
    };
  };

  auto arg = [&c, &unset, &none]()
  {
    return fem::FormArgument{fem::DofMapPack{dofmap3_t(dofs.data(), num_cells),
                                             std::integral_constant<int, bs>{},
                                             c},
                             unset, none};
  };

  std::array<T, ndim * ndim> Ab;
  std::array<T, 3 * num_dofs_cell> cdofs_b;

  std::map<std::pair<std::int32_t, std::int32_t>, T> full;
  fem::impl::assemble_cells_matrix<false>(
      collect(full), geometry_pack(), c, arg(), arg(), {}, {}, kernel, {}, {},
      std::span<T>(Ab), std::span<T>(cdofs_b));

  // Every (test, trial) dof pair sharing a cell is touched, and no
  // entry is zero.
  CHECK(!full.empty());
  for (const auto& [rc, v] : full)
    CHECK(v != T(0));

  // Marking a row for a Dirichlet condition zeros that row of the
  // element matrix, and leaves the others alone.
  std::vector<std::int8_t> bc0(bs * num_dofs, 0);
  bc0[0] = 1;
  std::map<std::pair<std::int32_t, std::int32_t>, T> zeroed;
  fem::impl::assemble_cells_matrix<false>(
      collect(zeroed), geometry_pack(), c, arg(), arg(), bc0, {}, kernel, {},
      {}, std::span<T>(Ab), std::span<T>(cdofs_b));

  for (const auto& [rc, v] : zeroed)
  {
    if (rc.first == 0)
      CHECK(v == T(0));
    else
      CHECK(v == full.at(rc));
  }
}

namespace
{
/// Entity list: (cell, local entity index) pairs, one per cell, with
/// distinct local indices so the kernel can check what it is handed.
constexpr std::array<std::int32_t, 4> entity_pairs = {0, 2, 1, 1};

/// The same entities as seen by the argument, whose mesh numbers the
/// cells the other way round.
constexpr std::array<std::int32_t, 4> entity_pairs0 = {1, 2, 0, 1};
using entities_t = md::mdspan<const std::int32_t,
                              md::extents<std::size_t, md::dynamic_extent, 2>>;

/// Entity permutations, indexed (cell, local entity).
constexpr std::array<std::uint8_t, 6> entity_perms = {0, 1, 2, 3, 4, 5};
using perms_t = md::mdspan<const std::uint8_t, md::dextents<std::size_t, 2>>;

/// Element vector kernel for an entity integral. The local entity index
/// and permutation are folded into the value so that passing the wrong
/// one is caught.
template <int bs>
void entity_kernel(T* be, const T*, const T*, const T* cdofs,
                   const int* local_entity, const std::uint8_t* perm, void*)
{
  for (std::size_t i = 0; i < num_dofs_cell; ++i)
    for (int k = 0; k < bs; ++k)
      be[bs * i + k] = be_value(cdofs, i, k) + 1000 * (*local_entity)
                       + 10000 * static_cast<T>(*perm);
}

/// Facet list: one interior facet between cells 0 and 1, as
/// ((cell, local facet), (cell, local facet)).
constexpr std::array<std::int32_t, 4> facet_entry = {0, 1, 1, 2};
using facets_t = md::mdspan<const std::int32_t,
                            md::extents<std::size_t, md::dynamic_extent, 2, 2>>;
using fcoeffs_t
    = md::mdspan<const T, md::extents<std::size_t, md::dynamic_extent, 2,
                                      md::dynamic_extent>>;

/// Element vector kernel for an interior facet integral. Writes a
/// `(2 * num_dofs_cell, bs)` element vector whose two halves are
/// distinguishable, from the coordinates of both attached cells.
template <int bs>
void facet_kernel(T* be, const T*, const T*, const T* cdofs,
                  const int* local_facet, const std::uint8_t*, void*)
{
  for (std::size_t side = 0; side < 2; ++side)
  {
    const T* c = cdofs + side * 3 * num_dofs_cell;
    for (std::size_t i = 0; i < num_dofs_cell; ++i)
      for (int k = 0; k < bs; ++k)
      {
        be[bs * (side * num_dofs_cell + i) + k]
            = be_value(c, i, k) + 1000 * local_facet[side];
      }
  }
}

auto unset_transform()
{
  return std::function<void(std::span<T>, std::span<const std::uint32_t>,
                            std::int32_t, int)>();
}
} // namespace

TEST_CASE("Assemble entities into a vector", "[assembly_kernels]")
{
  constexpr int bs = 2;
  entities_t entities(entity_pairs.data(), num_cells, 2);
  entities_t entities0(entity_pairs0.data(), num_cells, 2);
  perms_t perms(entity_perms.data(), num_cells, 3);

  std::vector<T> b(bs * num_dofs, 0);
  std::array<T, bs * num_dofs_cell> be_b;
  std::array<T, 3 * num_dofs_cell> cdofs_b;
  auto unset = unset_transform();
  fem::impl::assemble_entities_vector(
      b, geometry_pack(), entities,
      fem::FormArgument{fem::DofMapPack{dofmap3_t(dofs.data(), num_cells),
                                        std::integral_constant<int, bs>{},
                                        entities0},
                        unset,
                        {}},
      entity_kernel<bs>, {}, {}, perms, std::span<T>(be_b),
      std::span<T>(cdofs_b));

  // Reference: gather through the entity's cell, scatter through the
  // argument's cell, with the entity's local index and permutation.
  std::vector<T> expected(bs * num_dofs, 0);
  for (std::size_t e = 0; e < num_cells; ++e)
  {
    const std::int32_t cell = entity_pairs[2 * e];
    const std::int32_t cell0 = entity_pairs0[2 * e];
    const std::int32_t local = entity_pairs[2 * e + 1];
    std::array<T, 3 * num_dofs_cell> cdofs;
    for (std::size_t i = 0; i < num_dofs_cell; ++i)
    {
      const std::int32_t node = x_dofs[num_dofs_cell * cell + i];
      for (std::size_t c = 0; c < 3; ++c)
        cdofs[3 * i + c] = x_coords[3 * node + c];
    }
    const T perm = entity_perms[3 * cell + local];
    for (std::size_t i = 0; i < num_dofs_cell; ++i)
      for (int k = 0; k < bs; ++k)
      {
        expected[bs * dofs[num_dofs_cell * cell0 + i] + k]
            += be_value(cdofs.data(), i, k) + 1000 * local + 10000 * perm;
      }
  }

  CHECK(b == expected);
}

TEST_CASE("Assemble entities into a scalar", "[assembly_kernels]")
{
  entities_t entities(entity_pairs.data(), num_cells, 2);
  perms_t perms(entity_perms.data(), num_cells, 3);
  std::array<T, 3 * num_dofs_cell> cdofs_b;

  // Accumulates the first coordinate of each entity's cell, so both the
  // gather and the per-entity accumulation are checked.
  auto kernel = [](T* value, const T*, const T*, const T* cdofs, const int*,
                   const std::uint8_t*, void*) { *value += cdofs[0] + 1; };

  const T value = fem::impl::assemble_entities_scalar<T>(
      geometry_pack(), entities, kernel, {}, {}, perms, std::span<T>(cdofs_b));

  T expected = 0;
  for (std::size_t e = 0; e < num_cells; ++e)
  {
    const std::int32_t node = x_dofs[num_dofs_cell * entity_pairs[2 * e]];
    expected += x_coords[3 * node] + 1;
  }

  CHECK(value == expected);
}

TEST_CASE("Assemble interior facets into a vector", "[assembly_kernels]")
{
  constexpr int bs = 2;
  facets_t facets(facet_entry.data(), 1, 2, 2);
  std::array<T, bs * 2 * num_dofs_cell> be_b;
  std::array<T, 2 * 3 * num_dofs_cell> cdofs_b;
  auto unset = unset_transform();

  // The element vector the kernel writes, computed independently.
  auto element_vector = [&]()
  {
    std::array<T, bs * 2 * num_dofs_cell> be{};
    for (std::size_t side = 0; side < 2; ++side)
    {
      const std::int32_t cell = facet_entry[2 * side];
      const std::int32_t local = facet_entry[2 * side + 1];
      std::array<T, 3 * num_dofs_cell> cdofs;
      for (std::size_t i = 0; i < num_dofs_cell; ++i)
      {
        const std::int32_t node = x_dofs[num_dofs_cell * cell + i];
        for (std::size_t c = 0; c < 3; ++c)
          cdofs[3 * i + c] = x_coords[3 * node + c];
      }
      for (std::size_t i = 0; i < num_dofs_cell; ++i)
        for (int k = 0; k < bs; ++k)
        {
          be[bs * (side * num_dofs_cell + i) + k]
              = be_value(cdofs.data(), i, k) + 1000 * local;
        }
    }
    return be;
  }();

  auto assemble_facets = [&](std::span<const std::int32_t> facets0_data)
  {
    std::vector<T> b(bs * num_dofs, 0);
    facets_t facets0(facets0_data.data(), 1, 2, 2);
    fem::impl::assemble_interior_facets_vector(
        b, geometry_pack(), facets,
        fem::FormArgument{fem::DofMapPack{dofmap3_t(dofs.data(), num_cells),
                                          std::integral_constant<int, bs>{},
                                          facets0},
                          unset,
                          {}},
        facet_kernel<bs>, {}, fcoeffs_t(nullptr, 1, 2, 0), perms_t(),
        std::span<T>(be_b), std::span<T>(cdofs_b));
    return b;
  };

  SECTION("both sides in the argument domain")
  {
    const std::vector<T> b = assemble_facets(facet_entry);

    std::vector<T> expected(bs * num_dofs, 0);
    for (std::size_t side = 0; side < 2; ++side)
    {
      const std::int32_t cell = facet_entry[2 * side];
      for (std::size_t i = 0; i < num_dofs_cell; ++i)
        for (int k = 0; k < bs; ++k)
        {
          expected[bs * dofs[num_dofs_cell * cell + i] + k]
              += element_vector[bs * (side * num_dofs_cell + i) + k];
        }
    }

    CHECK(b == expected);
  }

  SECTION("one side absent from the argument domain")
  {
    // A cell marked -1 does not exist in the test function domain, so
    // only the other side's block is scattered.
    constexpr std::array<std::int32_t, 4> facets0 = {0, 1, -1, 2};
    const std::vector<T> b = assemble_facets(facets0);

    std::vector<T> expected(bs * num_dofs, 0);
    for (std::size_t i = 0; i < num_dofs_cell; ++i)
      for (int k = 0; k < bs; ++k)
        expected[bs * dofs[i] + k] += element_vector[bs * i + k];

    CHECK(b == expected);

    // The absent side must contribute nothing, so this differs from the
    // both-sides result.
    CHECK(b != assemble_facets(facet_entry));
  }
}

TEST_CASE("Assemble interior facets into a scalar", "[assembly_kernels]")
{
  facets_t facets(facet_entry.data(), 1, 2, 2);
  std::array<T, 2 * 3 * num_dofs_cell> cdofs_b;

  // Uses a coordinate from each side, so a gather that filled only one
  // half of the buffer is caught.
  auto kernel = [](T* value, const T*, const T*, const T* cdofs, const int*,
                   const std::uint8_t*, void*)
  { *value += cdofs[0] + cdofs[3 * num_dofs_cell]; };

  const T value = fem::impl::assemble_interior_facets_scalar<T>(
      geometry_pack(), facets, kernel, {}, fcoeffs_t(nullptr, 1, 2, 0),
      perms_t(), std::span<T>(cdofs_b));

  const std::int32_t node0 = x_dofs[num_dofs_cell * facet_entry[0]];
  const std::int32_t node1 = x_dofs[num_dofs_cell * facet_entry[2]];
  CHECK(value == x_coords[3 * node0] + x_coords[3 * node1]);
}

TEST_CASE("Assemble cells into a matrix (lifting mode)", "[assembly_kernels]")
{
  constexpr std::array<std::int32_t, 2> cells = {0, 1};
  std::span<const std::int32_t> c(cells);
  auto unset = unset_transform();

  constexpr int bs = 1;
  constexpr std::size_t ndim = bs * num_dofs_cell;

  auto kernel
      = [](T* Ae, const T*, const T*, const T*, const int*, const std::uint8_t*,
           void*) { std::fill_n(Ae, ndim * ndim, T(1)); };

  auto arg = [&c, &unset]()
  {
    return fem::FormArgument{fem::DofMapPack{dofmap3_t(dofs.data(), num_cells),
                                             std::integral_constant<int, bs>{},
                                             c},
                             unset,
                             {}};
  };

  std::array<T, ndim * ndim> Ab;
  std::array<T, 3 * num_dofs_cell> cdofs_b;

  auto run = [&](std::span<const std::int8_t> bc1)
  {
    std::vector<std::int32_t> rows;
    auto collect = [&rows](std::span<const std::int32_t> r,
                           std::span<const std::int32_t>, std::span<const T>)
    { rows.insert(rows.end(), r.begin(), r.end()); };
    fem::impl::assemble_cells_matrix<true>(
        collect, geometry_pack(), c, arg(), arg(), {}, bc1, kernel, {}, {},
        std::span<T>(Ab), std::span<T>(cdofs_b));
    return rows;
  };

  // Lifting mode indexes the column markers for every dof, so they
  // must cover the column space; an empty span is not valid input.
  // With nothing marked, no cell can contribute and all are skipped
  // before the kernel runs.
  std::vector<std::int8_t> bc_none(num_dofs, 0);
  CHECK(run(bc_none).empty());

  // Dof 0 belongs to cell 0 only, so only that cell is visited.
  std::vector<std::int8_t> bc_dof0(num_dofs, 0);
  bc_dof0[0] = 1;
  CHECK(run(bc_dof0).size() == num_dofs_cell);

  // Dof 2 belongs to both cells.
  std::vector<std::int8_t> bc_dof2(num_dofs, 0);
  bc_dof2[2] = 1;
  CHECK(run(bc_dof2).size() == 2 * num_dofs_cell);
}

TEST_CASE("Assemble interior facets into a matrix (lifting mode)",
          "[assembly_kernels]")
{
  constexpr int bs = 1;
  constexpr std::size_t nfacets = 2;

  // Two synthetic interior facets, both between cells 0 and 1.
  constexpr std::array<std::int32_t, 8> facet_pairs = {0, 0, 1, 0, 0, 1, 1, 1};

  // In the trial function domain the first side of the second facet
  // does not exist. Its half of the joint dofmap buffer is then left
  // holding the first facet's dofs.
  constexpr std::array<std::int32_t, 8> trial_pairs = {0, 0, 1, 0, -1, 1, 1, 1};

  facets_t facets(facet_pairs.data(), nfacets, 2, 2);
  facets_t trial_facets(trial_pairs.data(), nfacets, 2, 2);

  int calls = 0;
  auto kernel = [&calls](T* Ae, const T*, const T*, const T*, const int*,
                         const std::uint8_t*, void*)
  {
    ++calls;
    std::fill_n(Ae, 2 * num_dofs_cell * 2 * num_dofs_cell, T(1));
  };

  auto unset = unset_transform();
  auto dmap = dofmap3_t(dofs.data(), num_cells);
  fem::FormArgument row{
      fem::DofMapPack{dmap, std::integral_constant<int, bs>{}, facets},
      unset,
      {}};
  fem::FormArgument col{
      fem::DofMapPack{dmap, std::integral_constant<int, bs>{}, trial_facets},
      unset,
      {}};

  // Dof 0 appears in cell 0's dofmap but not cell 1's, so on the second
  // facet the only live trial side carries no marked column.
  std::vector<std::int8_t> bc1(num_dofs, 0);
  bc1[0] = 1;

  std::array<T, (2 * num_dofs_cell) * (2 * num_dofs_cell)> Ab;
  std::array<T, 2 * 3 * num_dofs_cell> cdofs_b;
  std::array<std::int32_t, 4 * num_dofs_cell> dofs_b;
  std::array<T, num_dofs_cell * num_dofs_cell> Ae_block_b;

  auto nothing = [](std::span<const std::int32_t>,
                    std::span<const std::int32_t>, std::span<const T>) {};
  fem::impl::assemble_interior_facets_matrix<true>(
      nothing, geometry_pack(), facets, row, col, {}, bc1, kernel, {},
      fcoeffs_t(nullptr, nfacets, 2, 0), perms_t(), std::span<T>(Ab),
      std::span<T>(cdofs_b), std::span<std::int32_t>(dofs_b),
      std::span<T>(Ae_block_b));

  // Only the first facet can contribute a lifting term. Testing the
  // joint dofmap instead of the two per-cell ones would see the first
  // facet's dofs left in the absent side's half and run the kernel
  // twice.
  CHECK(calls == 1);
}
