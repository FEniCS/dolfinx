// Copyright (C) 2019-2026 Garth N. Wells, Chris Richardson, Joseph P. Dean and
// Jørgen S. Dokken
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "FunctionSpace.h"
#include "traits.h"
#include <algorithm>
#include <basix/mdspan.hpp>
#include <concepts>
#include <cstdint>
#include <dolfinx/common/types.h>
#include <dolfinx/mesh/EntityMap.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/Topology.h>
#include <dolfinx/mesh/cell_types.h>
#include <format>
#include <functional>
#include <map>
#include <memory>
#include <ranges>
#include <set>
#include <span>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

namespace dolfinx::fem
{
template <dolfinx::scalar T>
class Constant;
template <dolfinx::scalar T, std::floating_point U>
class Function;

/// @brief Type of integral
enum class IntegralType : std::int8_t
{
  cell = 0,           ///< Cell
  exterior_facet = 1, ///< Exterior facet
  interior_facet = 2, ///< Interior facet
  vertex = 3,         ///< Vertex
  ridge = 4           ///< Ridge
};

/// @brief Topological dimension of the mesh entities an integral of the
/// given type is over.
///
/// @param[in] type Integral type.
/// @param[in] tdim Topological dimension of the integration domain.
/// @return Entity dimension, equal to `tdim` for a cell integral.
constexpr int integral_entity_dim(IntegralType type, int tdim)
{
  switch (type)
  {
  case IntegralType::exterior_facet:
  case IntegralType::interior_facet:
    return tdim - 1;
  case IntegralType::ridge:
    return tdim - 2;
  case IntegralType::vertex:
    return 0;
  case IntegralType::cell:
    return tdim;
  }

  throw std::invalid_argument("Unknown integral type.");
}

namespace impl
{

/// @brief Permutations of the cell-local entities that an integral of
/// the given type is over.
///
/// @param[in,out] topology Mesh topology of the integration domain.
/// @param[in] type Integral type.
/// @param[in] cell_type Cell type of the integration domain.
/// @return Permutation of each cell-local entity, shape
/// `(num_cells, entities_per_cell)`.
inline md::mdspan<const std::uint8_t, md::dextents<std::size_t, 2>>
entity_permutations(mesh::Topology& topology, IntegralType type,
                    mesh::CellType cell_type)
{
  return mesh::entity_permutations(
      topology, integral_entity_dim(type, topology.dim()), cell_type);
}

/// @brief Check that integration entities can be mapped to cells of the
/// mesh of an argument or coefficient.
///
/// An integration entity maps to a *cell* of that mesh, so unless the
/// meshes have equal dimension the entity dimension must be that
/// mesh's.
///
/// @param[in] tdim Topological dimension of the integration domain.
/// @param[in] edim Topological dimension of the integration entities.
/// @param[in] dim0 Topological dimension of the argument/coefficient
/// mesh.
/// @throws std::invalid_argument if the entities cannot be mapped.
inline void check_entity_mapping_dim(int tdim, int edim, int dim0)
{
  if (tdim > dim0 and edim != dim0)
  {
    throw std::invalid_argument(std::format(
        "Cannot map integration entities of dimension {} to cells of a "
        "mesh of dimension {}. An argument or coefficient on another mesh "
        "must live on the entities being integrated over.",
        edim, dim0));
  }
}
} // namespace impl

/// @brief Represents integral data, containing the kernel, and a list
/// of entities to integrate over and the indices of the coefficient
/// functions (relative to the Form) active for this integral.
template <dolfinx::scalar T, std::floating_point U = scalar_value_t<T>>
struct integral_data
{
  /// @brief Create a structure to hold integral data.
  /// @param[in] kernel Integration kernel function.
  /// @param[in] entities Indices of entities to integrate over.
  /// @param[in] coeffs Indices of the coefficients that are present
  /// (active) in `kernel`.
  template <typename K, typename V, typename W>
    requires std::is_convertible_v<
                 std::remove_cvref_t<K>,
                 std::function<void(T*, const T*, const T*, const U*,
                                    const int*, const uint8_t*, void*)>>
                 and std::is_convertible_v<std::remove_cvref_t<V>,
                                           std::vector<std::int32_t>>
                 and std::is_convertible_v<std::remove_cvref_t<W>,
                                           std::vector<int>>
  integral_data(K&& kernel, V&& entities, W&& coeffs)
      : kernel(std::forward<K>(kernel)), entities(std::forward<V>(entities)),
        coeffs(std::forward<W>(coeffs))
  {
  }

  /// @brief The integration kernel.
  std::function<void(T*, const T*, const T*, const U*, const int*,
                     const uint8_t*, void*)>
      kernel;

  /// @brief The entities to integrate over for this integral. These are
  /// the entities in 'full' mesh.
  std::vector<std::int32_t> entities;

  /// @brief Indices of coefficients (from the form) that are in this
  /// integral.
  std::vector<int> coeffs;
};

/// @brief A representation of finite element variational forms.
///
/// A note on the order of trial and test spaces: FEniCS numbers
/// argument spaces starting with the leading dimension of the
/// corresponding tensor (matrix). In other words, the test space is
/// numbered 0 and the trial space is numbered 1. However, in order to
/// have a notation that agrees with most existing finite element
/// literature, in particular
///
///  \f[   a = a(u, v)        \f]
///
/// the spaces are numbered from right to left
///
///  \f[   a: V_1 \times V_0 \rightarrow \mathbb{R}  \f]
///
/// This is reflected in the ordering of the spaces that should be
/// supplied to generated subclasses. In particular, when a bilinear
/// form is initialized, it should be initialized as `a(V_1, V_0) =
/// ...`, where `V_1` is the trial space and `V_0` is the test space.
/// However, when a form is initialized by a list of argument spaces
/// (the variable `function_spaces` in the constructors below), the list
/// of spaces should start with space number 0 (the test space) and then
/// space number 1 (the trial space).
///
/// @tparam T Scalar type in the form.
/// @tparam U Float (real) type used for the finite element and
/// geometry.
template <dolfinx::scalar T, std::floating_point U = dolfinx::scalar_value_t<T>>
class Form
{
public:
  /// Scalar type
  using scalar_type = T;

  /// Geometry type
  using geometry_type = U;

  /// @brief Create a finite element form.
  ///
  /// @note User applications will normally call a factory function
  /// rather using this interface directly.
  ///
  /// @param[in] V Function spaces for the form arguments, e.g. test and
  /// trial function spaces.
  /// @param[in] integrals Integrals in the form, where
  /// `integrals[{type, i, kernel_index}]` gives the `integral_data` of
  /// type `type` at position `i` in the flattened, sorted-by-subdomain-id
  /// list of integrals, for kernel `kernel_index`. The subdomain ids can
  /// contain duplicate entries referring to different kernels over the
  /// same subdomain.
  /// @param[in] coefficients Coefficients in the form.
  /// @param[in] constants Constants in the form.
  /// @param[in] mesh Mesh of the domain to integrate over (the
  /// 'integration domain').
  /// @param[in] needs_facet_permutations Set to `true` if any of the
  /// integration kernels require cell permutation data.
  /// @param[in] entity_maps A list of `EntityMap`s. For every mesh other
  /// than `mesh` on which a trial function, test function, or
  /// coefficient is defined, `entity_maps` must contain an `EntityMap`
  /// relating that mesh and `mesh`.
  ///
  /// @note For the single domain case, pass an empty `entity_maps`.
  template <typename X>
    requires std::is_convertible_v<
                 std::remove_cvref_t<X>,
                 std::map<std::tuple<IntegralType, int, int>,
                          integral_data<scalar_type, geometry_type>>>
  Form(
      const std::vector<std::shared_ptr<const FunctionSpace<geometry_type>>>& V,
      X&& integrals, std::shared_ptr<const mesh::Mesh<geometry_type>> mesh,
      const std::vector<
          std::shared_ptr<const Function<scalar_type, geometry_type>>>&
          coefficients,
      const std::vector<std::shared_ptr<const Constant<scalar_type>>>&
          constants,
      bool needs_facet_permutations,
      const std::vector<std::reference_wrapper<const mesh::EntityMap>>&
          entity_maps)
      : _function_spaces(V), _integrals(std::forward<X>(integrals)),
        _mesh(std::move(mesh)), _coefficients(coefficients),
        _constants(constants),
        _needs_facet_permutations(needs_facet_permutations)
  {
    if (!_mesh)
      throw std::invalid_argument("Form Mesh is null.");

    // `_mesh` is fixed for the remainder of construction, so its
    // topology and dimension are fetched once and reused below rather
    // than being re-fetched for every integral/coefficient.
    const mesh::Topology& topology = *_mesh->topology();
    const int tdim = topology.dim();

    // Map the integration entities of one integral to the
    // argument/coefficient domain, checking first that the mapping is
    // expressible: an integration entity maps to a *cell* of that mesh,
    // so unless the meshes have equal dimension the integral's entity
    // dimension must be that mesh's.
    auto map_entities
        = [tdim,
           &topology](IntegralType type, std::span<const std::int32_t> entities,
                      const mesh::Topology& topology0,
                      const mesh::EntityMap& emap) -> std::vector<std::int32_t>
    {
      if (type == IntegralType::cell)
      {
        return mesh::extract_cells_from_entities(
            topology0, topology, md::mdspan(entities.data(), entities.size()),
            std::cref(emap));
      }

      if (type == IntegralType::vertex)
      {
        throw std::invalid_argument(
            "Vertex integrals are not supported for a form with an argument "
            "or coefficient on another mesh. Supported types are cell, "
            "exterior facet, interior facet and ridge.");
      }

      impl::check_entity_mapping_dim(tdim, integral_entity_dim(type, tdim),
                                     topology0.dim());

      // Map the (cell, local_entity) pairs, flattened (interior facets
      // hold two pairs per entity), to cells of the argument/coefficient
      // mesh
      std::vector<std::int32_t> cells = mesh::extract_cells_from_entities(
          topology0, topology,
          md::mdspan<const std::int32_t,
                     md::extents<std::size_t, md::dynamic_extent, 2>>(
              entities.data(), entities.size() / 2, 2),
          std::cref(emap));

      // Replace the cell of each pair. Only the cell column is
      // meaningful: for codim > 0 the entity is itself the cell, so it
      // has no local index. The local index column is kept for the
      // layout used in packing/assembly.
      std::vector<std::int32_t> e(entities.begin(), entities.end());
      for (std::size_t i = 0; i < cells.size(); ++i)
        e[2 * i] = cells[i];
      return e;
    };

    _edata.reserve(_function_spaces.size());
    for (auto& space : _function_spaces)
    {
      // Working map: [integral type, integral_idx, kernel_idx]->entities
      std::map<std::tuple<IntegralType, int, int>,
               std::variant<std::vector<std::int32_t>,
                            std::span<const std::int32_t>>>
          vdata;

      if (auto mesh0 = space->mesh(); mesh0 == _mesh)
      {
        for (auto& [key, integral] : _integrals)
          vdata.insert({key, std::span(integral.entities)});
      }
      else
      {
        // Find correct entity map
        const mesh::Topology& topology0 = *mesh0->topology();
        const mesh::EntityMap& emap
            = mesh::find_entity_map(entity_maps, topology, topology0);
        for (auto& [key, itg] : _integrals)
        {
          auto [type, idx, kernel_idx] = key;
          vdata.insert(
              {key, map_entities(type, itg.entities, topology0, emap)});
        }
      }

      _edata.push_back(std::move(vdata));
    }

    for (auto& [key, integral] : _integrals)
    {
      auto [type, idx, kernel_idx] = key;
      for (int c : integral.coeffs)
      {
        if (auto mesh0 = coefficients.at(c)->function_space()->mesh();
            mesh0 == _mesh)
        {
          _cdata.insert({{type, idx, c}, std::span(integral.entities)});
        }
        else
        {
          // Find correct entity map
          const mesh::Topology& topology0 = *mesh0->topology();
          const mesh::EntityMap& emap
              = mesh::find_entity_map(entity_maps, topology, topology0);
          _cdata.insert(
              {{type, idx, c},
               map_entities(type, integral.entities, topology0, emap)});
        }
      }
    }
  }

  // Copy constructor (deleted). _edata and _cdata cache std::spans
  // aliasing the entity vectors owned by _integrals; a shallow copy
  // would leave the copy's spans pointing into the original's data.
  Form(const Form& form) = delete;

  /// Move constructor
  /// @note Valid because ::_integrals is a `std::map`, whose elements
  /// keep a stable address across a move, so the `std::span`s cached in
  /// ::_edata and ::_cdata remain valid after the move.
#ifdef _MSC_VER
  /// @note Explicit `noexcept`, MSVC only: MSVC's `std::map` move
  /// constructor isn't marked `noexcept`, so Form's move constructor
  /// would otherwise be deduced possibly-throwing. A map's move never
  /// actually throws - it only transfers internal state - so the
  /// noexcept override is safe.
  Form(Form&& form) noexcept = default;
#else
  Form(Form&& form) = default;
#endif

  /// Destructor
  ~Form() = default;

  // Copy assignment (deleted). Same aliasing reason as the copy
  // constructor.
  Form& operator=(const Form& form) = delete;

  /// Move assignment
  /// @note Valid for the same reason as the move constructor: move
  /// assigning a `std::map` transfers its nodes, so the entity vectors
  /// aliased by the `std::span`s cached in ::_edata and ::_cdata keep
  /// their addresses.
  Form& operator=(Form&& form) = default;

  /// @brief Rank of the form.
  ///
  /// bilinear form = 2, linear form = 1, functional = 0, etc.
  ///
  /// @return The rank of the form.
  int rank() const { return _function_spaces.size(); }

  /// @brief Common mesh for the form (the 'integration domain').
  /// @return The integration domain mesh.
  std::shared_ptr<const mesh::Mesh<geometry_type>> mesh() const
  {
    return _mesh;
  }

  /// @brief Function spaces for all arguments.
  /// @return Function spaces.
  const std::vector<std::shared_ptr<const FunctionSpace<geometry_type>>>&
  function_spaces() const
  {
    return _function_spaces;
  }

  /// @brief Get the kernel function for an integral.
  ///
  /// @param[in] type Integral type.
  /// @param[in] idx Integral index in the flattened list of integral
  /// kernels (see ::domain).
  /// @param[in] kernel_idx Index of the kernel (we may have multiple
  /// kernels for a given idx in mixed-topology meshes).
  /// @return Function to call for `tabulate_tensor`.
  std::function<void(scalar_type*, const scalar_type*, const scalar_type*,
                     const geometry_type*, const int*, const uint8_t*, void*)>
  kernel(IntegralType type, int idx, int kernel_idx) const
  {
    auto it = _integrals.find({type, idx, kernel_idx});
    if (it == _integrals.end())
      throw std::out_of_range("Requested integral kernel not found.");
    return it->second.kernel;
  }

  /// @brief Get types of integrals in the form.
  /// @return Integrals types.
  std::set<IntegralType> integral_types() const
  {
    std::set<IntegralType> types;
    for (auto& [key, integral] : _integrals)
      types.insert(std::get<0>(key));
    return types;
  }

  /// @brief Indices of coefficients that are active for a given
  /// integral (kernel).
  ///
  /// A form is split into multiple integrals (kernels) and each
  /// integral might contain only a subset of all coefficients in the
  /// form. This function returns an indicator array for a given
  /// integral kernel that signifies which coefficients are present.
  ///
  /// @param[in] type Integral type.
  /// @param[in] idx Integral index in the flattened list of integral
  /// kernels (see ::domain).
  /// @return Indices of the coefficients that are active (present) in
  /// the given integral.
  std::vector<int> active_coeffs(IntegralType type, int idx) const
  {
    auto it = std::ranges::find_if(_integrals,
                                   [type, idx](auto& x)
                                   {
                                     auto [t, idx_, kernel_idx] = x.first;
                                     return t == type and idx_ == idx;
                                   });
    if (it == _integrals.end())
      throw std::out_of_range("Could not find active coefficient list.");
    return it->second.coeffs;
  }

  /// @brief Get number of integrals (kernels) for a given integral type and
  /// kernel index.
  ///
  /// For a form containing two integrals `integral_a` and `integral_b`
  /// with subdomain ids `(1, 4)` and `(3, 4, 5)` respectively, the integrals
  /// are stored as a flattened list, sorted by subdomain id:
  /// ```cpp
  /// auto form_integrals = {integral_a, integral_b, integral_a,
  ///                        integral_b, integral_b};
  /// auto form_integral_ids = {1, 3, 4, 4, 5};
  /// ```
  /// @param[in] type Integral type.
  /// @param[in] kernel_idx Index of the kernel (we may have multiple
  /// kernels for a integral type in mixed-topology meshes).
  /// @return Number of integrals (kernels) of the given type and kernel
  /// index.
  int num_integrals(IntegralType type, int kernel_idx) const
  {
    return std::ranges::count_if(_integrals,
                                 [type, kernel_idx](auto& x)
                                 {
                                   auto [t, id, k_idx] = x.first;
                                   return t == type and k_idx == kernel_idx;
                                 });
  }

  /// @brief Mesh entity indices to integrate over for a given integral
  /// (kernel).
  ///
  /// These are the entities in the mesh returned by ::mesh that are
  /// integrated over by a given integral (kernel).
  ///
  /// - For IntegralType::cell, returns a list of cell indices.
  /// - For IntegralType::exterior_facet, returns a list with shape
  /// `(num_facets, 2)` (row-major storage), where for row `i`, `[i, 0]`
  /// is the cell index and `[i, 1]` is the local facet index relative
  /// to the cell.
  /// - For IntegralType::interior_facet, returns a list with shape
  /// `(num_facets, 4)` (row-major storage), where for row `i`, `[i, 0]`
  /// is the index of one attached cell, `[i, 1]` is the local facet
  /// index relative to that cell, `[i, 2]` is the index of the other
  /// attached cell, and `[i, 3]` is the local facet index relative to
  /// that cell.
  ///
  /// @param[in] type Integral type.
  /// @param[in] idx Integral index in the flattened list of integral
  /// kernels. For a form containing two integrals `integral_a` and
  /// `integral_b` with subdomain ids `(1, 4)` and `(3, 4, 5)`
  /// respectively, the integrals are stored as a flattened list,
  /// sorted by subdomain id:
  /// ```cpp
  /// auto form_integrals = {integral_a, integral_b, integral_a,
  ///                        integral_b, integral_b};
  /// auto form_integral_ids = {1, 3, 4, 4, 5};
  /// ```
  /// @param[in] kernel_idx Index of the kernel within the domain (we
  /// may have multiple kernels for a given id in mixed-topology
  /// meshes).
  /// @return Entity indices, with respect to the mesh::Mesh returned by
  /// ::mesh, to integrate over.
  std::span<const std::int32_t> domain(IntegralType type, int idx,
                                       int kernel_idx) const
  {
    auto it = _integrals.find({type, idx, kernel_idx});
    if (it == _integrals.end())
      throw std::out_of_range("Requested domain not found.");
    return it->second.entities;
  }

  /// @brief Argument function mesh integration entity indices.
  ///
  /// Integration can be performed over cells/facets involving functions
  /// that are defined on different meshes but which share common cells,
  /// i.e. meshes can be 'views' into a common mesh. Meshes can share
  /// some cells but a common cell will have a different index in each
  /// mesh::Mesh. Consider:
  /// ```cpp
  /// auto mesh = this->mesh();
  /// auto entities = this->domain(type, idx, kernel_idx);
  /// auto entities0 = this->domain_arg(type, rank, idx, kernel_idx);
  /// ```
  ///
  /// Assembly is performed over `entities`, where `entities[i]` is an
  /// entity index (e.g., cell index) in `mesh`. `entities0` holds the
  /// corresponding entity indices but in the mesh associated with the
  /// argument function (test/trial function) space. `entities[i]` and
  /// `entities0[i]` point to the same mesh entity, but with respect to
  /// different mesh views. In some cases, such as when integrating over
  /// the interface between two domains that do not overlap, an entity
  /// may exist in one domain but not another. In this case, the entity
  /// is marked with -1.
  ///
  /// @param[in] type Integral type.
  /// @param[in] rank Argument index, e.g. `0` for the test function space, `1`
  /// for the trial function space.
  /// @param[in] idx Integral identifier.
  /// @param[in] kernel_idx Index of the kernel (we may have multiple
  /// kernels for a given id in mixed-topology meshes).
  /// @return Entity indices in the argument function space mesh that is
  /// integrated over.
  /// - For cell integrals it has shape `(num_cells,)`.
  /// - For exterior/interior facet integrals, it has shape `(num_facets, 2)`
  /// (row-major storage), where `[i, 0]` is the index of a cell and
  /// `[i, 1]` is the local index of the facet relative to the cell.
  std::span<const std::int32_t> domain_arg(IntegralType type, int rank, int idx,
                                           int kernel_idx) const
  {
    auto it = _edata.at(rank).find({type, idx, kernel_idx});
    if (it == _edata.at(rank).end())
      throw std::out_of_range("Requested domain for argument not found.");

    return std::visit([](const auto& v) -> std::span<const std::int32_t>
                      { return v; }, it->second);
  }

  /// @brief Coefficient function mesh integration entity indices.
  ///
  /// This method is equivalent to ::domain_arg, but returns mesh entity
  /// indices for coefficient Function%s.
  ///
  /// @param[in] type Integral type.
  /// @param[in] idx Integral identifier.
  /// @param[in] c Coefficient index.
  /// @return Entity indices in the coefficient function space mesh that
  /// is integrated over.
  /// - For cell integrals it has shape `(num_cells,)`.
  /// - For exterior/interior facet integrals, it has shape `(num_facets, 2)`
  /// (row-major storage), where `[i, 0]` is the index of a cell and
  /// `[i, 1]` is the local index of the facet relative to the cell.
  std::span<const std::int32_t> domain_coeff(IntegralType type, int idx,
                                             int c) const
  {
    auto it = _cdata.find({type, idx, c});
    if (it == _cdata.end())
      throw std::out_of_range("No domain for requested integral.");
    return std::visit([](const auto& v) -> std::span<const std::int32_t>
                      { return v; }, it->second);
  }

  /// @brief Access coefficients.
  /// @return Coefficients in the form.
  const std::vector<
      std::shared_ptr<const Function<scalar_type, geometry_type>>>&
  coefficients() const
  {
    return _coefficients;
  }

  /// @brief Get bool indicating whether permutation data needs to be
  /// passed into these integrals.
  /// @return True if cell permutation data is required
  bool needs_facet_permutations() const { return _needs_facet_permutations; }

  /// @brief Offset for each coefficient expansion array on a cell.
  ///
  /// Used to pack data for multiple coefficients in a flat array. The
  /// last entry is the size required to store all coefficients.
  ///
  /// @return Coefficient offsets.
  std::vector<int> coefficient_offsets() const
  {
    std::vector<int> n{0};
    n.reserve(_coefficients.size() + 1);
    for (auto& c : _coefficients)
    {
      if (!c)
        throw std::runtime_error("Not all form coefficients have been set.");
      n.push_back(n.back() + c->function_space()->element()->space_dimension());
    }
    return n;
  }

  /// @brief Access constants.
  /// @return Constants in the form.
  const std::vector<std::shared_ptr<const Constant<scalar_type>>>&
  constants() const
  {
    return _constants;
  }

private:
  // Function spaces (one for each argument)
  std::vector<std::shared_ptr<const FunctionSpace<geometry_type>>>
      _function_spaces;

  // Integrals (integral type, idx, kernel_idx)
  std::map<std::tuple<IntegralType, int, int>,
           integral_data<scalar_type, geometry_type>>
      _integrals;

  // The mesh
  std::shared_ptr<const mesh::Mesh<geometry_type>> _mesh;

  // Form coefficients
  std::vector<std::shared_ptr<const Function<scalar_type, geometry_type>>>
      _coefficients;

  // Constants associated with the Form
  std::vector<std::shared_ptr<const Constant<scalar_type>>> _constants;

  // True if permutation data needs to be passed into these integrals
  bool _needs_facet_permutations;

  // Mapped domain index data for argument functions.
  //
  // Consider:
  //
  // entities  = this->domain(type, idx, kernel_idx);
  // entities0 = _edata[0][{type, idx, kernel_idx}];
  //
  // Then `entities[i]` is a mesh entity index (e.g., cell index) in
  // `_mesh`, and  `entities0[i]` is the index of the same entity but in
  // the mesh associated with the argument 0 (test function) space.
  std::vector<std::map<
      std::tuple<IntegralType, int, int>,
      std::variant<std::vector<std::int32_t>, std::span<const std::int32_t>>>>
      _edata;

  // Mapped domain index data for coefficient functions.
  //
  // Consider:
  //
  // entities  = this->domain(type, idx, kernel_idx);
  // entities0 = _cdata[{type, idx, c}];
  //
  // where `c` is the coefficient index.
  //
  // Then `entities[i]` is a mesh entity index (e.g., cell index) in
  // `_mesh`, and  `entities0[i]` is the index of the same entity but in
  // the mesh associated with the coefficient Function.
  std::map<
      std::tuple<IntegralType, int, int>,
      std::variant<std::vector<std::int32_t>, std::span<const std::int32_t>>>
      _cdata;
};
} // namespace dolfinx::fem
