// Copyright (C) 2019-2025 Garth N. Wells, Chris Richardson, Joseph P. Dean and
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
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/types.h>
#include <dolfinx/mesh/EntityMap.h>
#include <dolfinx/mesh/Mesh.h>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <ranges>
#include <span>
#include <tuple>
#include <utility>
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
  vertex = 3          ///< Vertex
};

/// @brief Represents integral data, containing the kernel, and a list
/// of entities to integrate over and the indicies of the coefficient
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
/// @tparam Kern Element kernel.
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
  /// `integrals[IntegralType, domain ID, kernel index]` returns the
  /// integral (`integral_data`) of type `IntegralType` over domain `ID`
  /// with kernel index `kernel index`.
  /// @param[in] coefficients Coefficients in the form.
  /// @param[in] constants Constants in the form.
  /// @param[in] mesh Mesh of the domain to integrate over (the
  /// 'integration domain').
  /// @param[in] needs_facet_permutations Set to `true` is any of the
  /// integration kernels require cell permutation data.
  /// @param[in] entity_maps If any trial functions, test functions, or
  /// coefficients in the form are not defined on `mesh` (the
  /// 'integration domain'),`entity_maps` must be supplied. For each key
  /// (a mesh, which is different to `mesh`) an array map must be
  /// provided which relates the entities in `mesh` to the entities in
  /// the key mesh e.g. for a key/value pair `(mesh0, emap)` in
  /// `entity_maps`, `emap[i]` is the entity in `mesh0` corresponding to
  /// entity `i` in `mesh`.
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
        _mesh(mesh), _coefficients(coefficients), _constants(constants),
        _needs_facet_permutations(needs_facet_permutations)
  {
    if (!_mesh)
      throw std::runtime_error("Form Mesh is null.");

    // A helper function to find the correct entity map for a given mesh
    auto get_entity_map
        = [mesh, &entity_maps](auto& mesh0) -> const mesh::EntityMap&
    {
      auto it = std::ranges::find_if(
          entity_maps,
          [mesh, mesh0](const mesh::EntityMap& em)
          {
            return ((em.topology() == mesh0->topology()
                     and em.sub_topology() == mesh->topology()))
                   or ((em.sub_topology() == mesh0->topology()
                        and em.topology() == mesh->topology()));
          });

      if (it == entity_maps.end())
      {
        throw std::runtime_error(
            "Incompatible mesh. argument entity_maps must be provided.");
      }
      return *it;
    };

    // A helper function to compute the (cell, local_facet) pairs in the
    // argument/coefficient domain from the (cell, local_facet) pairs in
    // `this->mesh()`.
    auto compute_facet_domains
        = [&](const auto& int_ents_mesh, int codim, const auto& c_to_f,
              const auto& emap, bool inverse)
    {
      // TODO: This function would be much neater using
      // `std::views::stride(2)` from C++ 23

      // Get a list of entities to map to the argument/coefficient
      // domain
      std::vector<std::int32_t> entities;
      entities.reserve(int_ents_mesh.size() / 2);
      if (codim == 0)
      {
        // In the codim 0 case, we need to map from cells in
        // `this->mesh()` to cells in the argument/coefficient mesh, so
        // here we extract the cells.
        for (std::size_t i = 0; i < int_ents_mesh.size(); i += 2)
          entities.push_back(int_ents_mesh[i]);
      }
      else if (codim == 1)
      {
        // In the codim 1 case, we need to map facets in `this->mesh()`
        // to cells in the argument/coefficient mesh, so here we extract
        // the facet index using the cell-to-facet connectivity.
        for (std::size_t i = 0; i < int_ents_mesh.size(); i += 2)
        {
          entities.push_back(
              c_to_f->links(int_ents_mesh[i])[int_ents_mesh[i + 1]]);
        }
      }
      else
        throw std::runtime_error("Codimension > 1 not supported.");

      // Map from entity indices in `this->mesh()` to the corresponding
      // cell indices in the argument/coefficient mesh
      std::vector<std::int32_t> cells_mesh0
          = emap.sub_topology_to_topology(entities, inverse);

      // Create a list of (cell, local_facet_index) pairs in the
      // argument/coefficient domain. Since `create_submesh`preserves
      // the local facet index (with respect to the cell), we can use
      // the local facet indices from the input integration entities
      std::vector<std::int32_t> e = int_ents_mesh;
      for (std::size_t i = 0; i < cells_mesh0.size(); ++i)
        e[2 * i] = cells_mesh0[i];

      return e;
    };

    for (auto& space : _function_spaces)
    {
      // Working map: [integral type, domain ID, kernel_idx]->entities
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
        const mesh::EntityMap& emap = get_entity_map(mesh0);

        // Determine direction of the map. We need to map from
        // `this->mesh()` to `mesh0`, so if `emap->sub_topology()` isn't
        // the source topology, we need the inverse map
        bool inverse = emap.sub_topology() == mesh0->topology();
        for (auto& [key, itg] : _integrals)
        {
          auto [type, id, kernel_idx] = key;
          std::vector<std::int32_t> e;
          if (type == IntegralType::cell)
            e = emap.sub_topology_to_topology(itg.entities, inverse);
          else if (type == IntegralType::exterior_facet
                   or type == IntegralType::interior_facet)
          {
            const mesh::Topology topology = *_mesh->topology();
            int tdim = topology.dim();
            assert(mesh0);
            int codim = tdim - mesh0->topology()->dim();
            assert(codim >= 0);
            auto c_to_f = topology.connectivity(tdim, tdim - 1);
            assert(c_to_f);

            e = compute_facet_domains(itg.entities, codim, c_to_f, emap,
                                      inverse);
          }
          else
            throw std::runtime_error("Integral type not supported.");

          vdata.insert({key, std::move(e)});
        }
      }

      _edata.push_back(vdata);
    }

    for (auto& [key, integral] : _integrals)
    {
      auto [type, id, kernel_idx] = key;
      for (int c : integral.coeffs)
      {
        if (auto mesh0 = coefficients.at(c)->function_space()->mesh();
            mesh0 == _mesh)
        {
          _cdata.insert({{type, id, c}, std::span(integral.entities)});
        }
        else
        {
          // Find correct entity map and determine direction of the map
          const mesh::EntityMap& emap = get_entity_map(mesh0);
          bool inverse = emap.sub_topology() == mesh0->topology();

          std::vector<std::int32_t> e;
          if (type == IntegralType::cell)
            e = emap.sub_topology_to_topology(integral.entities, inverse);
          else if (type == IntegralType::exterior_facet
                   or type == IntegralType::interior_facet)
          {
            const mesh::Topology topology = *_mesh->topology();
            int tdim = topology.dim();
            assert(mesh0);
            int codim = tdim - mesh0->topology()->dim();
            auto c_to_f = topology.connectivity(tdim, tdim - 1);
            assert(c_to_f);

            e = compute_facet_domains(integral.entities, codim, c_to_f, emap,
                                      inverse);
          }
          else
            throw std::runtime_error("Integral type not supported.");

          _cdata.insert({{type, id, c}, std::move(e)});
        }
      }
    }
  }

  /// Copy constructor
  Form(const Form& form) = delete;

  /// Move constructor
  Form(Form&& form) = default;

  /// Destructor
  virtual ~Form() = default;

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
  /// @param[in] type Integral type.
  /// @param[in] id Integral subdomain ID.
  /// @param[in] kernel_idx Index of the kernel (we may have multiple
  /// kernels for a given ID in mixed-topology meshes).
  /// @return Function to call for `tabulate_tensor`.
  std::function<void(scalar_type*, const scalar_type*, const scalar_type*,
                     const geometry_type*, const int*, const uint8_t*, void*)>
  kernel(IntegralType type, int id, int kernel_idx) const
  {
    auto it = _integrals.find({type, id, kernel_idx});
    if (it == _integrals.end())
      throw std::runtime_error("Requested integral kernel not found.");
    return it->second.kernel;
  }

  /// @brief Get types of integrals in the form.
  /// @return Integrals types.
  std::set<IntegralType> integral_types() const
  {
    std::vector<IntegralType> set_data;
    std::ranges::transform(_integrals, std::back_inserter(set_data),
                           [](auto& x) { return std::get<0>(x.first); });
    return std::set<IntegralType>(set_data.begin(), set_data.end());
  }

  /// @brief Indices of coefficients that are active for a given
  /// integral (kernel).
  ///
  /// A form is split into multiple integrals (kernels) and each
  /// integral might container only a subset of all coefficients in the
  /// form. This function returns an indicator array for a given
  /// integral kernel that signifies which coefficients are present.
  ///
  /// @param[in] type Integral type.
  /// @param[in] id Domain index (identifier) of the integral.
  std::vector<int> active_coeffs(IntegralType type, int id) const
  {
    auto it = std::ranges::find_if(_integrals,
                                   [type, id](auto& x)
                                   {
                                     auto [t, id_, kernel_idx] = x.first;
                                     return t == type and id_ == id;
                                   });
    if (it == _integrals.end())
      throw std::runtime_error("Could not find active coefficient list.");
    return it->second.coeffs;
  }

  /// @brief Get the IDs for integrals (kernels) for given integral
  /// domain type.
  ///
  /// The IDs correspond to the domain IDs which the integrals are
  /// defined for in the form. `ID=-1` is the default integral over the
  /// whole domain.
  ///
  /// @param[in] type Integral type.
  /// @return List of IDs for given integral type.
  std::vector<int> integral_ids(IntegralType type) const
  {
    std::vector<int> ids;
    for (auto& [key, integral] : _integrals)
    {
      auto [t, id, kernel_idx] = key;
      if (t == type)
        ids.push_back(id);
    }

    // IDs may be repeated in mixed-topology meshes, so remove
    // duplicates
    std::sort(ids.begin(), ids.end());
    auto it = std::unique(ids.begin(), ids.end());
    ids.erase(it, ids.end());
    return ids;
  }

  /// @brief Mesh entity indices to integrate over for a given integral
  /// (kernel).
  ///
  /// These are the entities in the mesh returned by ::mesh that are
  /// integrated over by a given integral (kernel).
  ///
  /// - For IntegralType::cell, returns a list of cell indices.
  /// - For IntegralType::exterior_facet, returns a list with shape
  /// `(num_facets, 2)`, where `[cell_index, 0]` is the cell index and
  /// `[cell_index, 1]` is the local facet index relative to the cell.
  /// - For IntegralType::interior_facet the shape is `(num_facets, 4)`,
  /// where `[cell_index, 0]` is one attached cell and `[cell_index, 1]`
  /// is the is the local facet index relative to the cell, and
  /// `[cell_index, 2]` is the other one attached cell and `[cell_index, 1]`
  /// is the is the local facet index relative to this cell. Storage
  /// is row-major.
  ///
  /// @param[in] type Integral type.
  /// @param[in] id Integral domain identifier.
  /// @param[in] kernel_idx Index of the kernel with in the domain (we
  /// may have multiple kernels for a given ID in mixed-topology
  /// meshes).
  /// @return Entity indices in the mesh::Mesh returned by mesh() to
  /// integrate over.
  std::span<const std::int32_t> domain(IntegralType type, int id,
                                       int kernel_idx) const
  {
    auto it = _integrals.find({type, id, kernel_idx});
    if (it == _integrals.end())
      throw std::runtime_error("Requested domain not found.");
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
  /// auto entities = this->domain(type, id, kernel_idx);
  /// auto entities0 = this->domain_arg(type, rank, id, kernel_idx);
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
  /// @param type Integral type.
  /// @param rank Argument index, e.g. `0` for the test function space, `1`
  /// for the trial function space.
  /// @param id Integral domain identifier.
  /// @param kernel_idx Kernel index (cell type).
  /// @return Entity indices in the argument function space mesh that is
  /// integrated over.
  /// - For cell integrals it has shape `(num_cells,)`.
  /// - For exterior/interior facet integrals, it has shape `(num_facts, 2)`
  /// (row-major storage), where `[i, 0]` is the index of a cell and
  /// `[i, 1]` is the local index of the facet relative to the cell.
  std::span<const std::int32_t> domain_arg(IntegralType type, int rank, int id,
                                           int kernel_idx) const
  {
    auto it = _edata.at(rank).find({type, id, kernel_idx});
    if (it == _edata.at(rank).end())
      throw std::runtime_error("Requested domain for argument not found.");
    try
    {
      return std::get<std::span<const std::int32_t>>(it->second);
    }
    catch (std::bad_variant_access& e)
    {
      return std::get<std::vector<std::int32_t>>(it->second);
    }
  }

  /// @brief Coefficient function mesh integration entity indices.
  ///
  /// This method is equivalent to ::domain_arg, but returns mesh entity
  /// indices for coefficient \link Function Functions. \endlink
  ///
  /// @param type Integral type.
  /// @param id Integral identifier index.
  /// @param c Coefficient index.
  /// @return Entity indices in the coefficient function space mesh that
  /// is integrated over.
  /// - For cell integrals it has shape `(num_cells,)`.
  /// - For exterior/interior facet integrals, it has shape `(num_facts, 2)`
  /// (row-major storage), where `[i, 0]` is the index of a cell and
  /// `[i, 1]` is the local index of the facet relative to the cell.
  std::span<const std::int32_t> domain_coeff(IntegralType type, int id,
                                             int c) const
  {
    auto it = _cdata.find({type, id, c});
    if (it == _cdata.end())
      throw std::runtime_error("No domain for requested integral.");
    try
    {
      return std::get<std::span<const std::int32_t>>(it->second);
    }
    catch (std::bad_variant_access& e)
    {
      return std::get<std::vector<std::int32_t>>(it->second);
    }
  }

  /// @brief Access coefficients.
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
  std::vector<int> coefficient_offsets() const
  {
    std::vector<int> n{0};
    for (auto& c : _coefficients)
    {
      if (!c)
        throw std::runtime_error("Not all form coefficients have been set.");
      n.push_back(n.back() + c->function_space()->element()->space_dimension());
    }
    return n;
  }

  /// @brief Access constants.
  const std::vector<std::shared_ptr<const Constant<scalar_type>>>&
  constants() const
  {
    return _constants;
  }

private:
  // Function spaces (one for each argument)
  std::vector<std::shared_ptr<const FunctionSpace<geometry_type>>>
      _function_spaces;

  // Integrals (integral type, id, celltype)
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
  // entities  = this->domain(IntegralType, integral(id), kernel_idx];
  // entities0 = _edata[0][IntegralType, integral(id), coefficient_index];
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
  // entities  = this->domain(IntegralType, integral(id), kernel_idx];
  // entities0 = _cdata[IntegralType, integral(id), coefficient_index];
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
