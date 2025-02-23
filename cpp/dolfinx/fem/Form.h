// Copyright (C) 2019-2023 Garth N. Wells and Chris Richardson
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "FunctionSpace.h"
#include "traits.h"
#include <algorithm>
#include <array>
#include <concepts>
#include <cstdint>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/types.h>
#include <dolfinx/mesh/Mesh.h>
#include <functional>
#include <map>
#include <memory>
#include <span>
#include <string>
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

namespace impl
{
/// @brief Compute the list of entity indices in `mesh` for the ith
/// integral (kernel) of a given type (i.e. cell, exterior facet, or
/// interior facet).
inline std::vector<std::int32_t>
compute_domain(IntegralType type, const mesh::Topology& topology, int codim,
               std::span<const std::int32_t> entities,
               std::span<const std::int32_t> entity_map)
{
  assert(codim >= 0);
  const int tdim = topology.dim();

  std::vector<std::int32_t> mapped_entities;
  mapped_entities.reserve(entities.size());
  switch (type)
  {
  case IntegralType::cell:
  {
    std::ranges::transform(entities, std::back_inserter(mapped_entities),
                           [&entity_map](auto e) { return entity_map[e]; });
    break;
  }
  case IntegralType::exterior_facet:
  {
    if (codim == 0)
    {
      for (std::size_t i = 0; i < entities.size(); i += 2)
      {
        // Add cell and the local facet index
        mapped_entities.insert(mapped_entities.end(),
                               {entity_map[entities[i]], entities[i + 1]});
      }
    }
    else if (codim == 1)
    {
      // In this case, the entity maps take facets in (`_mesh`) to cells
      // in `mesh`, so we need to get the facet number from the (cell,
      // local_facet pair) first.
      auto c_to_f = topology.connectivity(tdim, tdim - 1);
      assert(c_to_f);
      for (std::size_t i = 0; i < entities.size(); i += 2)
      {
        // Get the facet index, and add cell and the local facet index
        std::int32_t facet = c_to_f->links(entities[i])[entities[i + 1]];
        mapped_entities.insert(mapped_entities.end(),
                               {entity_map[facet], entities[i + 1]});
      }
    }
    else
      throw std::runtime_error("Codimension > 1 not supported.");

    break;
  }
  case IntegralType::interior_facet:
  {
    // Get the codimension of the mesh
    if (codim == 0)
    {
      for (std::size_t i = 0; i < entities.size(); i += 2)
      {
        // Add cell and the local facet index
        mapped_entities.insert(mapped_entities.end(),
                               {entity_map[entities[i]], entities[i + 1]});
      }
    }
    else if (codim == 1)
    {
      // In this case, the entity maps take facets in (`_mesh`) to
      // cells in `mesh`, so we need to get the facet number from
      // the (cell, local_facet pair) first.
      auto c_to_f = topology.connectivity(tdim, tdim - 1);
      assert(c_to_f);
      for (std::size_t i = 0; i < entities.size(); i += 2)
      {
        // Get the facet index, and add cell and the local facet index
        std::int32_t facet = c_to_f->links(entities[i])[entities[i + 1]];
        mapped_entities.insert(mapped_entities.end(),
                               {entity_map[facet], entities[i + 1]});
      }
    }

    break;
  }
  default:
    throw std::runtime_error("Integral type not supported.");
  }

  return mapped_entities;
}

} // namespace impl

/// @brief Represents integral data, containing the integral ID, the
/// kernel, and a list of entities to integrate over.
template <dolfinx::scalar T, std::floating_point U = scalar_value_type_t<T>>
struct integral_data
{
  /// @brief Create a structure to hold integral data.
  /// @param[in] kernel Integration kernel.
  /// @param[in] entities Indices of entities to integrate over.
  /// @param[in] coeffs Indices of the coefficients that are present
  /// (active) in `kernel`.
  template <typename K, typename V, typename W>
    requires std::is_convertible_v<
                 std::remove_cvref_t<K>,
                 std::function<void(T*, const T*, const T*, const U*,
                                    const int*, const uint8_t*)>>
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
                     const uint8_t*)>
      kernel;

  /// @brief The entities to integrate over.
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
/// @tparam U Float (real) type used for the finite element and geometry.
/// @tparam Kern Element kernel.
template <dolfinx::scalar T,
          std::floating_point U = dolfinx::scalar_value_type_t<T>>
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
  /// @todo 'integration domain' is not properly defined. Fix it.
  ///
  /// @param[in] V Function spaces for the form arguments.
  /// @param[in] integrals The integrals in the form. For each integral
  /// type, there is a list of integral data.
  /// @param[in] coefficients Coefficients in the form.
  /// @param[in] constants Constants in the form.
  /// @param[in] needs_facet_permutations Set to `true` is any of the
  /// integration kernels require cell permutation data.
  /// @param[in] entity_maps If any trial functions, test functions, or
  /// coefficients in the form are not defined over the same mesh as the
  /// integration domain, `entity_maps` must be supplied. For each key
  /// (a mesh, different to the integration domain mesh) a map should be
  /// provided relating the entities in the integration domain mesh to
  /// the entities in the key mesh e.g. for a pair (msh, emap) in
  /// `entity_maps`, `emap[i]` is the entity in `msh` corresponding to
  /// entity `i` in the integration domain mesh.
  /// @param[in] mesh Mesh of the domain to integrate over. This is
  /// required when
  /// 1. There are no argument functions from which the mesh can be
  /// extracted, e.g. for functionals; or
  /// 2. The argument functions are defined on different meshes.
  ///
  /// @note For the single domain case, pass an empty `entity_maps`.
  ///
  /// @pre The integral data in integrals must be sorted by domain
  /// (domain id).
  template <typename X>
    requires std::is_convertible_v<
                 std::remove_cvref_t<X>,
                 std::map<std::tuple<IntegralType, int, int>,
                          integral_data<scalar_type, geometry_type>>>
  Form(
      const std::vector<std::shared_ptr<const FunctionSpace<geometry_type>>>& V,
      X&& integrals,
      const std::vector<
          std::shared_ptr<const Function<scalar_type, geometry_type>>>&
          coefficients,
      const std::vector<std::shared_ptr<const Constant<scalar_type>>>&
          constants,
      bool needs_facet_permutations,
      const std::map<std::shared_ptr<const mesh::Mesh<geometry_type>>,
                     std::span<const std::int32_t>>& entity_maps,
      std::shared_ptr<const mesh::Mesh<geometry_type>> mesh)
      : _function_spaces(V), _integrals(std::forward<X>(integrals)),
        _coefficients(coefficients), _constants(constants), _mesh(mesh),
        _needs_facet_permutations(needs_facet_permutations)
  {
    if (!_mesh)
      throw std::runtime_error("Form Mesh is null.");

    {
      // Integration domain mesh is passed, so check that it is (1)
      // common for spaces and coefficients (2) or an entity_map is
      // available
      for (auto& space : V)
      {
        if (auto mesh0 = space->mesh();
            mesh0 != _mesh and !entity_maps.contains(mesh0))
        {
          throw std::runtime_error(
              "Incompatible mesh. argument entity_maps must be provided.");
        }
      }
      for (auto& c : coefficients)
      {
        if (auto mesh0 = c->function_space()->mesh();
            mesh0 != _mesh and !entity_maps.contains(mesh0))
        {
          throw std::runtime_error(
              "Incompatible mesh. coefficient entity_maps must be provided.");
        }
      }
    }

    for (auto V : _function_spaces)
    {
      std::map<std::tuple<IntegralType, int, int>, std::vector<std::int32_t>>
          vdata;
      if (auto mesh0 = V->mesh(); mesh0 == _mesh)
      {
        for (auto& [key, integral] : _integrals)
          vdata.insert({key, integral.entities});
      }
      else
      {
        const mesh::Topology topology = *_mesh->topology();
        int tdim = topology.dim();
        assert(mesh0);
        int codim = tdim - mesh0->topology()->dim();
        auto it = entity_maps.find(mesh0);
        if (it == entity_maps.end())
          throw std::runtime_error("No entity map for the mesh.");
        std::span<const std::int32_t> entity_map = it->second;
        for (auto& [key, integral] : _integrals)
        {
          auto [type, i, kernel_idx] = key;
          std::vector<std::int32_t> e = impl::compute_domain(
              type, topology, codim, integral.entities, entity_map);
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
          _cdata.insert({{type, id, c}, integral.entities});
        }
        else
        {
          const mesh::Topology topology = *_mesh->topology();
          int tdim = topology.dim();
          assert(mesh0);
          int codim = tdim - mesh0->topology()->dim();
          auto it = entity_maps.find(mesh0);
          if (it == entity_maps.end())
            throw std::runtime_error("No entity map for the mesh.");
          std::span<const std::int32_t> entity_map = it->second;
          std::vector<std::int32_t> e = impl::compute_domain(
              type, topology, codim, integral.entities, entity_map);
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
  /// @return The rank of the form
  int rank() const { return _function_spaces.size(); }

  /// @brief Extract common mesh for the form.
  /// @return The mesh.
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

  /// @brief Get the kernel function for integral `i` on given domain
  /// type.
  /// @param[in] type Integral type.
  /// @param[in] i The subdomain ID.
  /// @param[in] kernel_idx Index of the kernel (we may have multiple
  /// kernels for a given ID in mixed-topology meshes).
  /// @return Function to call for `tabulate_tensor`.
  std::function<void(scalar_type*, const scalar_type*, const scalar_type*,
                     const geometry_type*, const int*, const uint8_t*)>
  kernel(IntegralType type, int i, int kernel_idx) const
  {
    auto it = _integrals.find({type, i, kernel_idx});
    if (it == _integrals.end())
      throw std::runtime_error("No kernel for requested domain index.");
    return it->second.kernel;
  }

  /// @brief Get types of integrals in the form.
  /// @return Integrals types.
  std::set<IntegralType> integral_types() const
  {
    std::set<IntegralType> set;
    std::ranges::for_each(_integrals, [&set](auto& x)
                          { set.insert(std::get<0>(x.first)); });
    return set;
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

  /// @brief Get the list of mesh entity indices for the ith integral
  /// (kernel) of a given type.
  ///
  /// For IntegralType::cell, returns a list of cell indices.
  ///
  /// For IntegralType::exterior_facet, returns a list of (cell_index,
  /// local_facet_index) pairs. Data is flattened with row-major layout,
  /// `shape=(num_facets, 2)`.
  ///
  /// For IntegralType::interior_facet, returns list of tuples of the
  /// form `(cell_index_0, local_facet_index_0, cell_index_1,
  /// local_facet_index_1)`. Data is flattened with row-major layout,
  /// `shape=(num_facets, 4)`.
  ///
  /// @param[in] type Integral type.
  /// @param[in] i Integral ID, i.e. (sub)domain index.
  /// @param[in] kernel_idx Index of the kernel (we may have multiple kernels
  /// for a given ID in mixed-topology meshes).
  /// @return List of active entities for the given integral (kernel).
  std::span<const std::int32_t> domain(IntegralType type, int i,
                                       int kernel_idx) const
  {
    auto it = _integrals.find({type, i, kernel_idx});
    if (it == _integrals.end())
      throw std::runtime_error("No kernel for requested domain index.");
    return it->second.entities;
  }

  /// @brief Argument domain.
  std::span<const std::int32_t> domain_arg(IntegralType type, int rank, int i,
                                           int kernel_idx) const
  {
    auto it = _edata.at(rank).find({type, i, kernel_idx});
    if (it == _edata.at(rank).end())
      throw std::runtime_error("No kernel for requested domain index.");
    return it->second;
  }

  /// @brief Argument domain.
  std::span<const std::int32_t> domain_coeff(IntegralType type, int i,
                                             int c) const
  {
    auto it = _cdata.find({type, i, c});
    if (it == _cdata.end())
    {
      throw std::runtime_error("No coefficient entity data for requested "
                               "domain index (type, id, coeff_idx). "
                               + std::to_string(static_cast<int>(type)) + " "
                               + std::to_string(i) + ", " + std::to_string(c));
    }
    return it->second;
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
    std::vector<int> n = {0};
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

  // Form coefficients
  std::vector<std::shared_ptr<const Function<scalar_type, geometry_type>>>
      _coefficients;

  // Constants associated with the Form
  std::vector<std::shared_ptr<const Constant<scalar_type>>> _constants;

  // The mesh
  std::shared_ptr<const mesh::Mesh<geometry_type>> _mesh;

  // True if permutation data needs to be passed into these integrals
  bool _needs_facet_permutations;

  // Domain data for argument functions
  //  [rank_i][IntegralType, integral(id), cell_type] -> entity map
  std::vector<
      std::map<std::tuple<IntegralType, int, int>, std::vector<std::int32_t>>>
      _edata;

  // Domain data for coefficients
  //  [IntegralType, integral(id), cell_type] -> entity map
  std::map<std::tuple<IntegralType, int, int>, std::vector<std::int32_t>>
      _cdata;
};
} // namespace dolfinx::fem
