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
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/types.h>
#include <dolfinx/mesh/Mesh.h>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <tuple>
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

/// @brief Represents integral data, containing the integral ID, the
/// kernel, and a list of entities to integrate over.
template <dolfinx::scalar T,
          FEkernel<T> Kern = std::function<void(
              T*, const T*, const T*, const T*, const int*, const uint8_t*)>>
struct integral_data
{
  /// @brief Kernel type
  using kern_t = Kern;

  /// @brief Create a structure to hold integral data.
  /// @tparam U `std::vector<std::int32_t>` holding entity indices.
  /// @param id Domain ID.
  /// @param kernel Integration kernel.
  /// @param entities Entities to integrate over.
  template <typename U>
  integral_data(int id, kern_t kernel, U&& entities)
      : id(id), kernel(kernel), entities(std::forward<U>(entities))
  {
  }

  /// @brief Create a structure to hold integral data.
  /// @param id Domain ID
  /// @param kernel Integration kernel.
  /// @param e Entities to integrate over.
  integral_data(int id, kern_t kernel, std::span<const std::int32_t> e)
      : id(id), kernel(kernel), entities(e.begin(), e.end())
  {
  }

  /// Integral ID
  int id;

  /// The integration kernel
  kern_t kernel;

  /// The entities to integrate over
  std::vector<std::int32_t> entities;
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
template <
    dolfinx::scalar T, std::floating_point U = dolfinx::scalar_value_type_t<T>,
    FEkernel<T> Kern
    = std::function<void(T*, const T*, const T*, const scalar_value_type_t<T>*,
                         const int*, const std::uint8_t*)>>
class Form
{
  using kern_t = Kern;

public:
  /// Scalar type
  using scalar_type = T;

  /// @brief Create a finite element form.
  ///
  /// @note User applications will normally call a builder function
  /// rather using this interface directly.
  ///
  /// @param[in] V Function spaces for the form arguments
  /// @param[in] integrals The integrals in the form. For each
  /// integral type, there is a list of integral data
  /// @param[in] coefficients
  /// @param[in] constants Constants in the Form
  /// @param[in] needs_facet_permutations Set to true is any of the
  /// integration kernels require cell permutation data
  /// @param[in] mesh Mesh of the domain. This is required when there
  /// are no argument functions from which the mesh can be extracted,
  /// e.g. for functionals.
  /// @param[in] entity_maps If any trial functions, test functions, or
  /// coefficients in the form are not defined over the same mesh as the
  /// integration domain, `entity_maps` must be supplied. For each key
  /// (a mesh, different to the integration domain mesh) a map should
  /// be provided relating the entities in the integration domain mesh
  /// to the entities in the key mesh.
  ///
  /// @pre The integral data in integrals must be sorted by domain
  template <typename X>
  Form(const std::vector<std::shared_ptr<const FunctionSpace<U>>>& V,
       X&& integrals,
       const std::vector<std::shared_ptr<const Function<scalar_type, U>>>&
           coefficients,
       const std::vector<std::shared_ptr<const Constant<scalar_type>>>&
           constants,
       bool needs_facet_permutations,
       std::shared_ptr<const mesh::Mesh<U>> mesh = nullptr,
       const std::map<std::shared_ptr<const mesh::Mesh<U>>,
                      std::span<const std::int32_t>>& entity_maps
       = {})
      : _function_spaces(V), _coefficients(coefficients), _constants(constants),
        _mesh(mesh), _needs_facet_permutations(needs_facet_permutations)
  {
    // Extract _mesh from FunctionSpace, and check they are the same
    if (!_mesh and !V.empty())
      _mesh = V[0]->mesh();
    for (auto& space : V)
      if (_mesh != space->mesh()
          and entity_maps.find(space->mesh()) == entity_maps.end())
        throw std::runtime_error(
            "Incompatible mesh. entity_maps must be provided.");
    if (!_mesh)
      throw std::runtime_error("No mesh could be associated with the Form.");

    // Store kernels, looping over integrals by domain type (dimension)
    for (auto&& [domain_type, data] : integrals)
    {
      if (!std::is_sorted(data.begin(), data.end(),
                          [](auto& a, auto& b) { return a.id < b.id; }))
      {
        throw std::runtime_error("Integral IDs not sorted");
      }

      std::vector<integral_data<T, kern_t>>& itg
          = _integrals[static_cast<std::size_t>(domain_type)];
      for (auto&& [id, kern, e] : data)
        itg.emplace_back(id, kern, std::move(e));
    }

    // Store entity maps
    for (auto [msh, map] : entity_maps)
      _entity_maps.insert({msh, std::vector(map.begin(), map.end())});
  }

  /// Copy constructor
  Form(const Form& form) = delete;

  /// Move constructor
  Form(Form&& form) = default;

  /// Destructor
  virtual ~Form() = default;

  /// Rank of the form (bilinear form = 2, linear form = 1, functional =
  /// 0, etc)
  /// @return The rank of the form
  int rank() const { return _function_spaces.size(); }

  /// Extract common mesh for the form
  /// @return The mesh
  std::shared_ptr<const mesh::Mesh<U>> mesh() const { return _mesh; }

  /// Return function spaces for all arguments
  /// @return Function spaces
  const std::vector<std::shared_ptr<const FunctionSpace<U>>>&
  function_spaces() const
  {
    return _function_spaces;
  }

  /// @brief Get the kernel function for integral i on given domain
  /// type.
  /// @param[in] type Integral type
  /// @param[in] i Domain identifier (index)
  /// @return Function to call for tabulate_tensor
  kern_t kernel(IntegralType type, int i) const
  {
    const auto& integrals = _integrals[static_cast<std::size_t>(type)];
    auto it = std::lower_bound(integrals.begin(), integrals.end(), i,
                               [](auto& itg_data, int i)
                               { return itg_data.id < i; });
    if (it != integrals.end() and it->id == i)
      return it->kernel;
    else
      throw std::runtime_error("No kernel for requested domain index.");
  }

  /// @brief Get types of integrals in the form.
  /// @return Integrals types.
  std::set<IntegralType> integral_types() const
  {
    std::set<IntegralType> set;
    for (std::size_t i = 0; i < _integrals.size(); ++i)
    {
      if (!_integrals[i].empty())
        set.insert(static_cast<IntegralType>(i));
    }

    return set;
  }

  /// @brief Number of integrals on given domain type.
  /// @param[in] type Integral type.
  /// @return Number of integrals.
  int num_integrals(IntegralType type) const
  {
    return _integrals[static_cast<std::size_t>(type)].size();
  }

  /// Get the IDs for integrals (kernels) for given integral type. The
  /// IDs correspond to the domain IDs which the integrals are defined
  /// for in the form. ID=-1 is the default integral over the whole
  /// domain.
  /// @param[in] type Integral type
  /// @return List of IDs for given integral type
  std::vector<int> integral_ids(IntegralType type) const
  {
    std::vector<int> ids;
    const auto& integrals = _integrals[static_cast<std::size_t>(type)];
    std::transform(integrals.begin(), integrals.end(), std::back_inserter(ids),
                   [](auto& integral) { return integral.id; });
    return ids;
  }

  /// @brief Get the list of cell indices for the ith integral (kernel)
  /// for the cell domain type.
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
  /// @param[in] type Integral domain type
  /// @param[in] i Integral ID, i.e. (sub)domain index
  /// @return List of active cell entities for the given integral (kernel)
  std::span<const std::int32_t> domain(IntegralType type, int i) const
  {
    const auto& integrals = _integrals[static_cast<std::size_t>(type)];
    auto it = std::lower_bound(integrals.begin(), integrals.end(), i,
                               [](auto& itg_data, int i)
                               { return itg_data.id < i; });
    if (it != integrals.end() and it->id == i)
      return it->entities;
    else
      throw std::runtime_error("No mesh entities for requested domain index.");
  }

  /// @brief TODO
  /// @param type TODO
  /// @param i TODO
  /// @param mesh TODO
  /// @return TODO
  std::vector<std::int32_t>
  domain(IntegralType type, int i,
         std::shared_ptr<const mesh::Mesh<U>> mesh) const
  {
    const auto& integrals = _integrals[static_cast<std::size_t>(type)];
    auto it = std::lower_bound(integrals.begin(), integrals.end(), i,
                               [](auto& itg_data, int i)
                               { return itg_data.id < i; });
    if (it != integrals.end() and it->id == i)
    {
      std::span<const std::int32_t> entities = it->entities;
      // TODO Handle out of range exception
      std::span<const std::int32_t> entity_map = _entity_maps.at(mesh);

      std::vector<std::int32_t> mapped_entities(entities.size(), -1);
      // TODO Use std::transform
      for (std::size_t i = 0; i < entities.size(); ++i)
        mapped_entities[i] = entity_map[entities[i]];
      // TODO Check all mapped correctly
      return mapped_entities;
    }
    else
      throw std::runtime_error("No mesh entities for requested domain index.");
  }

  /// Access coefficients
  const std::vector<std::shared_ptr<const Function<T, U>>>& coefficients() const
  {
    return _coefficients;
  }

  /// Get bool indicating whether permutation data needs to be passed
  /// into these integrals
  /// @return True if cell permutation data is required
  bool needs_facet_permutations() const { return _needs_facet_permutations; }

  /// Offset for each coefficient expansion array on a cell. Used to
  /// pack data for multiple coefficients in a flat array. The last
  /// entry is the size required to store all coefficients.
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

  /// Access constants
  const std::vector<std::shared_ptr<const Constant<T>>>& constants() const
  {
    return _constants;
  }

private:
  // Function spaces (one for each argument)
  std::vector<std::shared_ptr<const FunctionSpace<U>>> _function_spaces;

  // Integrals. Array index is
  // static_cast<std::size_t(IntegralType::foo)
  std::array<std::vector<integral_data<T, kern_t>>, 4> _integrals;

  // Form coefficients
  std::vector<std::shared_ptr<const Function<T, U>>> _coefficients;

  // Constants associated with the Form
  std::vector<std::shared_ptr<const Constant<T>>> _constants;

  // The mesh
  std::shared_ptr<const mesh::Mesh<U>> _mesh;

  // True if permutation data needs to be passed into these integrals
  bool _needs_facet_permutations;

  // Entity maps
  std::map<std::shared_ptr<const mesh::Mesh<U>>, std::vector<std::int32_t>>
      _entity_maps;
};
} // namespace dolfinx::fem
