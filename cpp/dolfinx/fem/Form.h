// Copyright (C) 2019-2020 Garth N. Wells and Chris Richardson
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "FunctionSpace.h"
#include <algorithm>
#include <array>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/MeshTags.h>
#include <functional>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

namespace dolfinx::fem
{

template <typename T>
class Constant;
template <typename T>
class Function;

/// @brief Type of integral
enum class IntegralType : std::int8_t
{
  cell = 0,           ///< Cell
  exterior_facet = 1, ///< Exterior facet
  interior_facet = 2, ///< Interior facet
  vertex = 3          ///< Vertex
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
template <typename T>
class Form
{
public:
  /// @brief Create a finite element form.
  ///
  /// @note User applications will normally call a fem::Form builder
  /// function rather using this interfcae directly.
  ///
  /// @param[in] function_spaces Function spaces for the form arguments
  /// @param[in] integrals The integrals in the form. The first key is
  /// the domain type. For each key there is a pair (list[domain id,
  /// integration kernel], domain markers).
  /// @param[in] coefficients
  /// @param[in] constants Constants in the Form
  /// @param[in] needs_facet_permutations Set to true is any of the
  /// integration kernels require cell permutation data
  /// @param[in] mesh The mesh of the domain. This is required when
  /// there are not argument functions from which the mesh can be
  /// extracted, e.g. for functionals
  Form(
      const std::vector<std::shared_ptr<const fem::FunctionSpace>>&
          function_spaces,
      const std::map<
          IntegralType,
          std::pair<
              std::vector<std::pair<
                  int, std::function<void(T*, const T*, const T*, const double*,
                                          const int*, const std::uint8_t*)>>>,
              const mesh::MeshTags<int>*>>& integrals,
      const std::vector<std::shared_ptr<const fem::Function<T>>>& coefficients,
      const std::vector<std::shared_ptr<const fem::Constant<T>>>& constants,
      bool needs_facet_permutations,
      const std::shared_ptr<const mesh::Mesh>& mesh = nullptr)
      : _function_spaces(function_spaces), _coefficients(coefficients),
        _constants(constants), _mesh(mesh),
        _needs_facet_permutations(needs_facet_permutations)
  {
    // Extract _mesh from fem::FunctionSpace, and check they are the same
    if (!_mesh and !function_spaces.empty())
      _mesh = function_spaces[0]->mesh();
    for (const auto& V : function_spaces)
    {
      if (_mesh != V->mesh())
        throw std::runtime_error("Incompatible mesh");
    }
    if (!_mesh)
      throw std::runtime_error("No mesh could be associated with the Form.");

    // Store kernels, looping over integrals by domain type (dimension)
    for (auto& integral_type : integrals)
    {
      const IntegralType type = integral_type.first;
      // Loop over integrals kernels and set domains
      switch (type)
      {
      case IntegralType::cell:
        for (auto& integral : integral_type.second.first)
          _cell_integrals.insert({integral.first, {integral.second, {}}});
        break;
      case IntegralType::exterior_facet:
        for (auto& integral : integral_type.second.first)
        {
          _exterior_facet_integrals.insert(
              {integral.first, {integral.second, {}}});
        }
        break;
      case IntegralType::interior_facet:
        for (auto& integral : integral_type.second.first)
        {
          _interior_facet_integrals.insert(
              {integral.first, {integral.second, {}}});
        }
        break;
      }

      if (integral_type.second.second)
      {
        assert(_mesh == integral_type.second.second->mesh());
        set_domains(type, *integral_type.second.second);
      }
    }

    // FIXME: do this neatly via a static function
    // Set markers for default integrals
    set_default_domains(*_mesh);
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
  std::shared_ptr<const mesh::Mesh> mesh() const { return _mesh; }

  /// Return function spaces for all arguments
  /// @return Function spaces
  const std::vector<std::shared_ptr<const fem::FunctionSpace>>&
  function_spaces() const
  {
    return _function_spaces;
  }

  /// Get the function for 'kernel' for integral i of given
  /// type
  /// @param[in] type Integral type
  /// @param[in] i Domain index
  /// @return Function to call for tabulate_tensor
  const std::function<void(T*, const T*, const T*, const double*, const int*,
                           const std::uint8_t*)>&
  kernel(IntegralType type, int i) const
  {
    switch (type)
    {
    case IntegralType::cell:
      return get_kernel_from_integrals(_cell_integrals, i);
    case IntegralType::exterior_facet:
      return get_kernel_from_integrals(_exterior_facet_integrals, i);
    case IntegralType::interior_facet:
      return get_kernel_from_integrals(_interior_facet_integrals, i);
    default:
      throw std::runtime_error(
          "Cannot access kernel. Integral type not supported.");
    }
  }

  /// Get types of integrals in the form
  /// @return Integrals types
  std::set<IntegralType> integral_types() const
  {
    std::set<IntegralType> set;
    if (!_cell_integrals.empty())
      set.insert(IntegralType::cell);
    if (!_exterior_facet_integrals.empty())
      set.insert(IntegralType::exterior_facet);
    if (!_interior_facet_integrals.empty())
      set.insert(IntegralType::interior_facet);

    return set;
  }

  /// Number of integrals of given type
  /// @param[in] type Integral type
  /// @return Number of integrals
  int num_integrals(IntegralType type) const
  {
    switch (type)
    {
    case IntegralType::cell:
      return _cell_integrals.size();
    case IntegralType::exterior_facet:
      return _exterior_facet_integrals.size();
    case IntegralType::interior_facet:
      return _interior_facet_integrals.size();
    default:
      throw std::runtime_error("Integral type not supported.");
    }
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
    switch (type)
    {
    case IntegralType::cell:
      std::transform(_cell_integrals.cbegin(), _cell_integrals.cend(),
                     std::back_inserter(ids),
                     [](auto& integral) { return integral.first; });
      break;
    case IntegralType::exterior_facet:
      std::transform(_exterior_facet_integrals.cbegin(),
                     _exterior_facet_integrals.cend(), std::back_inserter(ids),
                     [](auto& integral) { return integral.first; });
      break;
    case IntegralType::interior_facet:
      std::transform(_interior_facet_integrals.cbegin(),
                     _interior_facet_integrals.cend(), std::back_inserter(ids),
                     [](auto& integral) { return integral.first; });
      break;
    default:
      throw std::runtime_error(
          "Cannot return IDs. Integral type not supported.");
    }

    return ids;
  }

  /// Get the list of cell indices for the ith integral (kernel)
  /// for the cell domain type
  /// @param[in] i Integral ID, i.e. (sub)domain index
  /// @return List of active cell entities for the given integral (kernel)
  const std::vector<std::int32_t>& cell_domains(int i) const
  {
    auto it = _cell_integrals.find(i);
    if (it == _cell_integrals.end())
      throw std::runtime_error("No mesh entities for requested domain index.");
    return it->second.second;
  }

  /// Get the list of (cell_index, local_facet_index) pairs for the ith
  /// integral (kernel) for the exterior facet domain type
  /// @param[in] i Integral ID, i.e. (sub)domain index
  /// @return List of (cell_index, local_facet_index) pairs
  const std::vector<std::pair<std::int32_t, int>>&
  exterior_facet_domains(int i) const
  {
    auto it = _exterior_facet_integrals.find(i);
    if (it == _exterior_facet_integrals.end())
      throw std::runtime_error("No mesh entities for requested domain index.");
    return it->second.second;
  }

  /// Get the list of (cell_index_0, local_facet_index_0, cell_index_1,
  /// local_facet_index_1) tuples for the ith integral (kernel) for the
  /// interior facet domain type,
  /// @param[in] i Integral ID, i.e. (sub)domain index
  /// @return List of tuples of the form
  /// (cell_index_0, local_facet_index_0, cell_index_1, local_facet_index_1)
  const std::vector<std::tuple<std::int32_t, int, std::int32_t, int>>&
  interior_facet_domains(int i) const
  {
    auto it = _interior_facet_integrals.find(i);
    if (it == _interior_facet_integrals.end())
      throw std::runtime_error("No mesh entities for requested domain index.");
    return it->second.second;
  }

  /// Access coefficients
  const std::vector<std::shared_ptr<const fem::Function<T>>>&
  coefficients() const
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
    for (const auto& c : _coefficients)
    {
      if (!c)
        throw std::runtime_error("Not all form coefficients have been set.");
      n.push_back(n.back() + c->function_space()->element()->space_dimension());
    }
    return n;
  }

  /// Access constants
  const std::vector<std::shared_ptr<const fem::Constant<T>>>& constants() const
  {
    return _constants;
  }

  /// Scalar type (T)
  using scalar_type = T;

private:
  using kern = std::function<void(T*, const T*, const T*, const double*,
                                  const int*, const std::uint8_t*)>;

  // Helper function to get the kernel for integral i from a map
  // of integrals i.e. from _cell_integrals
  // @param[in] integrals Map of integrals
  // @param[in] i Domain index
  // @return Function to call for tabulate_tensor
  template <typename U>
  const std::function<void(T*, const T*, const T*, const double*, const int*,
                           const std::uint8_t*)>&
  get_kernel_from_integrals(const U& integrals, int i) const
  {
    auto it = integrals.find(i);
    if (it == integrals.end())
      throw std::runtime_error("No kernel for requested domain index.");
    return it->second.first;
  }

  // Helper function to get a std::vector of (cell, local_facet) pairs
  // corresponding to a given facet index.
  // @param[in] f Facet index
  // @param[in] f_to_c Facet to cell connectivity
  // @param[in] c_to_f Cell to facet connectivity
  // @return Vector of (cell, local_facet) pairs
  template <int num_cells>
  static std::array<std::pair<std::int32_t, int>, num_cells>
  get_cell_local_facet_pairs(
      std::int32_t f, const xtl::span<const std::int32_t>& cells,
      const dolfinx::graph::AdjacencyList<std::int32_t>& c_to_f)
  {
    // Loop over cells sharing facet
    assert(cells.size() == num_cells);
    std::array<std::pair<std::int32_t, int>, num_cells> cell_local_facet_pairs;
    for (int c = 0; c < num_cells; ++c)
    {
      // Get local index of facet with respect to the cell
      std::int32_t cell = cells[c];
      auto cell_facets = c_to_f.links(cell);
      auto facet_it = std::find(cell_facets.begin(), cell_facets.end(), f);
      assert(facet_it != cell_facets.end());
      int local_f = std::distance(cell_facets.begin(), facet_it);
      cell_local_facet_pairs[c] = {cell, local_f};
    }

    return cell_local_facet_pairs;
  }

  // Set cell domains
  template <typename iterator>
  void set_cell_domains(
      std::map<int, std::pair<kern, std::vector<std::int32_t>>>& integrals,
      const iterator& tagged_cells_begin, const iterator& tagged_cells_end,
      const std::vector<int>& tags)
  {
    // For cell integrals use all markers (but not on ghost entities)
    for (auto c = tagged_cells_begin; c != tagged_cells_end; ++c)
    {
      const std::size_t pos = std::distance(tagged_cells_begin, c);
      if (auto it = integrals.find(tags[pos]); it != integrals.end())
        it->second.second.push_back(*c);
    }
  }

  // Set exterior facet domains
  template <typename iterator>
  void set_exterior_facet_domains(
      const mesh::Topology& topology,
      std::map<int, std::pair<kern, std::vector<std::pair<std::int32_t, int>>>>&
          integrals,
      const iterator& tagged_facets_begin, const iterator& tagged_facets_end,
      const std::vector<int>& tags)
  {
    // When a mesh is not ghosted by cell, it is not straightforward
    // to distinguish between (i) exterior facets and (ii) interior
    // facets that are on a partition boundary. If there are no
    // ghost cells, build a set of owned facts that are ghosted on
    // another process to help determine if a facet is on an
    // exterior boundary.
    int tdim = topology.dim();
    assert(topology.index_map(tdim));
    assert(topology.index_map(tdim - 1));
    const std::vector<std::int32_t> fwd_shared_facets
        = topology.index_map(tdim)->overlapped()
              ? std::vector<std::int32_t>()
              : topology.index_map(tdim - 1)->shared_indices();

    auto f_to_c = topology.connectivity(tdim - 1, tdim);
    assert(f_to_c);
    auto c_to_f = topology.connectivity(tdim, tdim - 1);
    assert(c_to_f);
    for (auto f = tagged_facets_begin; f != tagged_facets_end; ++f)
    {
      // All "owned" facets connected to one cell, that are not
      // shared, should be external
      // TODO: Consider removing this check and integrating over all
      // tagged facets. This may be useful in a few cases.
      if (f_to_c->num_links(*f) == 1)
      {
        if (!std::binary_search(fwd_shared_facets.begin(),
                                fwd_shared_facets.end(), *f))
        {
          const std::size_t pos = std::distance(tagged_facets_begin, f);
          if (auto it = integrals.find(tags[pos]); it != integrals.end())
          {
            // There will only be one pair for an exterior facet integral
            std::pair<std::int32_t, int> pair = get_cell_local_facet_pairs<1>(
                *f, f_to_c->links(*f), *c_to_f)[0];
            it->second.second.push_back(pair);
          }
        }
      }
    }
  }

  // Set interior facet domains
  template <typename iterator>
  static void set_interior_facet_domains(
      const mesh::Topology& topology,
      std::map<int,
               std::pair<kern, std::vector<std::tuple<std::int32_t, int,
                                                      std::int32_t, int>>>>&
          integrals,
      const iterator& tagged_facets_begin, const iterator& tagged_facets_end,
      const std::vector<int>& tags)
  {
    int tdim = topology.dim();
    auto f_to_c = topology.connectivity(tdim - 1, tdim);
    assert(f_to_c);
    auto c_to_f = topology.connectivity(tdim, tdim - 1);
    assert(c_to_f);
    for (auto f = tagged_facets_begin; f != tagged_facets_end; ++f)
    {
      if (f_to_c->num_links(*f) == 2)
      {
        const std::size_t pos = std::distance(tagged_facets_begin, f);
        if (auto it = integrals.find(tags[pos]); it != integrals.end())
        {
          std::array<std::pair<std::int32_t, int>, 2> pairs
              = get_cell_local_facet_pairs<2>(*f, f_to_c->links(*f), *c_to_f);
          it->second.second.emplace_back(pairs[0].first, pairs[0].second,
                                         pairs[1].first, pairs[1].second);
        }
      }
    }
  }

  // Sets the entity indices to assemble over for kernels with a domain
  // ID
  // @param[in] type Integral type
  // @param[in] marker MeshTags with domain ID. Entities with marker 'i'
  // will be assembled over using the kernel with ID 'i'. The MeshTags
  // is not stored.
  void set_domains(IntegralType type, const mesh::MeshTags<int>& marker)
  {
    std::shared_ptr<const mesh::Mesh> mesh = marker.mesh();
    const mesh::Topology& topology = mesh->topology();
    const int tdim = topology.dim();
    int dim = type == IntegralType::cell ? tdim : tdim - 1;
    if (dim != marker.dim())
    {
      throw std::runtime_error("Invalid MeshTags dimension: "
                               + std::to_string(marker.dim()));
    }

    // Get mesh tag data
    const std::vector<int>& tags = marker.values();
    const std::vector<std::int32_t>& tagged_entities = marker.indices();
    assert(topology.index_map(dim));
    const auto entity_end
        = std::lower_bound(tagged_entities.begin(), tagged_entities.end(),
                           topology.index_map(dim)->size_local());
    switch (type)
    {
    case IntegralType::cell:
      set_cell_domains(_cell_integrals, tagged_entities.cbegin(), entity_end,
                       tags);
      break;
    default:
      mesh->topology_mutable().create_connectivity(dim, tdim);
      mesh->topology_mutable().create_connectivity(tdim, dim);
      switch (type)
      {
      case IntegralType::exterior_facet:
        set_exterior_facet_domains(topology, _exterior_facet_integrals,
                                   tagged_entities.cbegin(), entity_end, tags);
        break;
      case IntegralType::interior_facet:
        set_interior_facet_domains(topology, _interior_facet_integrals,
                                   tagged_entities.cbegin(), entity_end, tags);
        break;
      default:
        throw std::runtime_error(
            "Cannot set domains. Integral type not supported.");
      }
    }
  }

  /// If there exists a default integral of any type, set the list of
  /// entities for those integrals from the mesh topology. For cell
  /// integrals, this is all cells. For facet integrals, it is either
  /// all interior or all exterior facets.
  /// @param[in] mesh Mesh
  void set_default_domains(const mesh::Mesh& mesh)
  {
    const mesh::Topology& topology = mesh.topology();
    const int tdim = topology.dim();

    // Cells. If there is a default integral, define it on all owned
    // cells
    for (auto& [domain_id, kernel_cells] : _cell_integrals)
    {
      if (domain_id == -1)
      {
        std::vector<std::int32_t>& cells = kernel_cells.second;
        const int num_cells = topology.index_map(tdim)->size_local();
        cells.resize(num_cells);
        std::iota(cells.begin(), cells.end(), 0);
      }
    }

    // Exterior facets. If there is a default integral, define it only
    // on owned surface facets.

    if (!_exterior_facet_integrals.empty())
    {
      mesh.topology_mutable().create_connectivity(tdim - 1, tdim);
      mesh.topology_mutable().create_connectivity(tdim, tdim - 1);
    }
    const std::vector<std::int32_t> boundary_facets
        = _exterior_facet_integrals.empty()
              ? std::vector<std::int32_t>()
              : mesh::exterior_facet_indices(topology);
    for (auto& [domain_id, kernel_facets] : _exterior_facet_integrals)
    {
      if (domain_id == -1)
      {
        std::vector<std::pair<std::int32_t, int>>& facets
            = kernel_facets.second;
        facets.clear();

        auto f_to_c = topology.connectivity(tdim - 1, tdim);
        assert(f_to_c);
        auto c_to_f = topology.connectivity(tdim, tdim - 1);
        assert(c_to_f);
        for (std::int32_t f : boundary_facets)
        {
          // There will only be one pair for an exterior facet integral
          std::pair<std::int32_t, int> pair
              = get_cell_local_facet_pairs<1>(f, f_to_c->links(f), *c_to_f)[0];
          facets.push_back(pair);
        }
      }
    }

    // Interior facets. If there is a default integral, define it only on
    // owned interior facets.
    for (auto& [domain_id, kernel_facets] : _interior_facet_integrals)
    {
      if (domain_id == -1)
      {
        std::vector<std::tuple<std::int32_t, int, std::int32_t, int>>& facets
            = kernel_facets.second;
        facets.clear();

        mesh.topology_mutable().create_connectivity(tdim - 1, tdim);
        auto f_to_c = topology.connectivity(tdim - 1, tdim);
        assert(f_to_c);
        mesh.topology_mutable().create_connectivity(tdim, tdim - 1);
        auto c_to_f = mesh.topology().connectivity(tdim, tdim - 1);
        assert(c_to_f);

        // Get number of facets owned by this process
        assert(topology.index_map(tdim - 1));
        const int num_facets = topology.index_map(tdim - 1)->size_local();
        facets.reserve(num_facets);
        for (int f = 0; f < num_facets; ++f)
        {
          if (f_to_c->num_links(f) == 2)
          {
            std::array<std::pair<std::int32_t, int>, 2> pairs
                = get_cell_local_facet_pairs<2>(f, f_to_c->links(f), *c_to_f);
            facets.emplace_back(pairs[0].first, pairs[0].second, pairs[1].first,
                                pairs[1].second);
          }
        }
      }
    }
  }

  // Function spaces (one for each argument)
  std::vector<std::shared_ptr<const fem::FunctionSpace>> _function_spaces;

  // Form coefficients
  std::vector<std::shared_ptr<const fem::Function<T>>> _coefficients;

  // Constants associated with the Form
  std::vector<std::shared_ptr<const fem::Constant<T>>> _constants;

  // The mesh
  std::shared_ptr<const mesh::Mesh> _mesh;

  // Cell integrals
  std::map<int, std::pair<kern, std::vector<std::int32_t>>> _cell_integrals;

  // Exterior facet integrals
  std::map<int, std::pair<kern, std::vector<std::pair<std::int32_t, int>>>>
      _exterior_facet_integrals;

  // Interior facet integrals
  std::map<int, std::pair<kern, std::vector<std::tuple<std::int32_t, int,
                                                       std::int32_t, int>>>>
      _interior_facet_integrals;

  // True if permutation data needs to be passed into these integrals
  bool _needs_facet_permutations;
};
} // namespace dolfinx::fem
