// Copyright (C) 2019-2020 Garth N. Wells and Chris Richardson
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfinx/function/FunctionSpace.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/MeshTags.h>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace dolfinx
{

namespace function
{
template <typename T>
class Constant;
template <typename T>
class Function;
} // namespace function

namespace fem
{

/// Type of integral
enum class IntegralType : std::int8_t
{
  cell = 0,
  exterior_facet = 1,
  interior_facet = 2,
  vertex = 3
};

/// Class for variational forms
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
  /// Create form
  ///
  /// @param[in] function_spaces Function spaces for the form arguments
  /// @param[in] integrals The integrals in the form. The first key is
  /// the domain type. For each key there is a pair (list[domain id,
  /// integration kernel], domain markers).
  /// @param[in] coefficients
  /// @param[in] constants Constants in the Form
  /// @param[in] needs_permutation_data Set to true is any of the
  /// integration kernels require cell permutation data
  /// @param[in] mesh The mesh of the domain. This is required when
  /// there are not argument functions from which the mesh can be
  /// extracted, e.g. for functionals
  Form(const std::vector<std::shared_ptr<const function::FunctionSpace>>&
           function_spaces,
       const std::map<
           IntegralType,
           std::pair<
               std::vector<std::pair<
                   int, std::function<void(
                            T*, const T*, const T*, const double*, const int*,
                            const std::uint8_t*, const std::uint32_t)>>>,
               const mesh::MeshTags<int>*>>& integrals,
       const std::vector<std::shared_ptr<const function::Function<T>>>&
           coefficients,
       const std::vector<std::shared_ptr<const function::Constant<T>>>&
           constants,
       bool needs_permutation_data,
       const std::shared_ptr<const mesh::Mesh>& mesh = nullptr)
      : _function_spaces(function_spaces), _coefficients(coefficients),
        _constants(constants), _mesh(mesh),
        _needs_permutation_data(needs_permutation_data)
  {
    // Extract _mesh from function::FunctionSpace, and check they are the same
    if (!_mesh and !function_spaces.empty())
      _mesh = function_spaces[0]->mesh();
    for (const auto& V : function_spaces)
      if (_mesh != V->mesh())
        throw std::runtime_error("Incompatible mesh");
    if (!_mesh)
      throw std::runtime_error("No mesh could be associated with the Form.");

    // Store kernels, looping over integrals by domain type (dimension)
    for (auto& integral_type : integrals)
    {
      // Add key to map
      const IntegralType type = integral_type.first;
      auto it = _integrals.emplace(
          type, std::map<int, std::pair<kern, std::vector<std::int32_t>>>());

      // Loop over integrals kernels
      for (auto& integral : integral_type.second.first)
        it.first->second.insert({integral.first, {integral.second, {}}});

      // FIXME: do this neatly via a static function
      // Set domains for integral type
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
  const std::vector<std::shared_ptr<const function::FunctionSpace>>&
  function_spaces() const
  {
    return _function_spaces;
  }

  /// Get the function for 'kernel' for integral i of given
  /// type
  /// @param[in] type Integral type
  /// @param[in] i Integral number
  /// @return Function to call for tabulate_tensor
  const std::function<void(T*, const T*, const T*, const double*, const int*,
                           const std::uint8_t*, const std::uint32_t)>&
  kernel(IntegralType type, int i) const
  {
    auto it0 = _integrals.find(type);
    if (it0 == _integrals.end())
      throw std::runtime_error("No kernels for requested type.");
    auto it1 = it0->second.find(i);
    if (it1 == it0->second.end())
      throw std::runtime_error("No kernel for requested index.");

    return it1->second.first;
  }

  /// Get types of integrals in the form
  /// @return Integrals types
  std::set<IntegralType> integral_types() const
  {
    std::set<IntegralType> set;
    for (auto& type : _integrals)
      set.insert(type.first);
    return set;
  }

  /// Number of integrals of given type
  /// @param[in] type Integral type
  /// @return Number of integrals
  int num_integrals(IntegralType type) const
  {
    if (auto it = _integrals.find(type); it == _integrals.end())
      return 0;
    else
      return it->second.size();
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
    if (auto it = _integrals.find(type); it != _integrals.end())
    {
      for (auto& kernel : it->second)
        ids.push_back(kernel.first);
    }
    return ids;
  }

  /// Get the list of mesh entity indices for the ith integral (kernel)
  /// for the given domain type, i.e. for cell integrals a list of cell
  /// indices, for facet integrals a list of facet indices, etc.
  /// @param[in] type The integral type
  /// @param[in] i Integral (kernel) index
  /// @return List of active entities for the given integral (kernel)
  const std::vector<std::int32_t>& domains(IntegralType type, int i) const
  {
    auto it0 = _integrals.find(type);
    if (it0 == _integrals.end())
      throw std::runtime_error("No kernels for requested type.");
    auto it1 = it0->second.find(i);
    if (it1 == it0->second.end())
      throw std::runtime_error("No kernel for requested index.");
    return it1->second.second;
  }

  /// Access coefficients
  const std::vector<std::shared_ptr<const function::Function<T>>>
  coefficients() const
  {
    return _coefficients;
  }

  /// Get bool indicating whether permutation data needs to be passed
  /// into these integrals
  /// @return True if cell permutation data is required
  bool needs_permutation_data() const { return _needs_permutation_data; }

  /// Offset for each coefficient expansion array on a cell. Used to
  /// pack data for multiple coefficients in a flat array. The last
  /// entry is the size required to store all coefficients.
  std::vector<int> coefficient_offsets() const
  {
    std::vector<int> n{0};
    for (const auto& c : _coefficients)
    {
      if (!c)
        throw std::runtime_error("Not all form coefficients have been set.");
      n.push_back(n.back() + c->function_space()->element()->space_dimension());
    }
    return n;
  }

  /// Access constants
  const std::vector<std::shared_ptr<const function::Constant<T>>>&
  constants() const
  {
    return _constants;
  }

private:
  /// Sets the entity indices to assemble over for kernels with a domain ID.
  /// @param[in] type Integral type
  /// @param[in] marker MeshTags with domain ID. Entities with marker
  /// 'i' will be assembled over using the kernel with ID 'i'. The
  /// MeshTags is not stored.
  void set_domains(IntegralType type, const mesh::MeshTags<int>& marker)
  {
    auto it0 = _integrals.find(type);
    assert(it0 != _integrals.end());

    std::shared_ptr<const mesh::Mesh> mesh = marker.mesh();
    const mesh::Topology& topology = mesh->topology();
    const int tdim = topology.dim();
    int dim = tdim;
    if (type == IntegralType::exterior_facet
        or type == IntegralType::interior_facet)
    {
      dim = tdim - 1;
      mesh->topology_mutable().create_connectivity(dim, tdim);
    }
    else if (type == IntegralType::vertex)
      dim = 0;

    if (dim != marker.dim())
    {
      throw std::runtime_error("Invalid MeshTags dimension:"
                               + std::to_string(marker.dim()));
    }

    // Get all integrals for considered entity type
    std::map<int, std::pair<kern, std::vector<std::int32_t>>>& integrals
        = it0->second;

    // Get mesh tag data
    const std::vector<int>& values = marker.values();
    const std::vector<std::int32_t>& tagged_entities = marker.indices();
    assert(topology.index_map(dim));
    const auto entity_end
        = std::lower_bound(tagged_entities.begin(), tagged_entities.end(),
                           topology.index_map(dim)->size_local());

    if (dim == tdim - 1)
    {
      auto f_to_c = topology.connectivity(tdim - 1, tdim);
      assert(f_to_c);
      if (type == IntegralType::exterior_facet)
      {
        // Only need to consider shared facets when there are no ghost
        // cells
        assert(topology.index_map(tdim));
        std::set<std::int32_t> fwd_shared;
        if (topology.index_map(tdim)->num_ghosts() == 0)
        {
          fwd_shared.insert(
              topology.index_map(tdim - 1)->shared_indices().begin(),
              topology.index_map(tdim - 1)->shared_indices().end());
        }

        for (auto f = tagged_entities.begin(); f != entity_end; ++f)
        {
          // All "owned" facets connected to one cell, that are not
          // shared, should be external
          if (f_to_c->num_links(*f) == 1
              and fwd_shared.find(*f) == fwd_shared.end())
          {
            const std::size_t i = std::distance(tagged_entities.cbegin(), f);
            if (auto it = integrals.find(values[i]); it != integrals.end())
              it->second.second.push_back(*f);
          }
        }
      }
      else if (type == IntegralType::interior_facet)
      {
        for (auto f = tagged_entities.begin(); f != entity_end; ++f)
        {
          if (f_to_c->num_links(*f) == 2)
          {
            const std::size_t i = std::distance(tagged_entities.cbegin(), f);
            if (auto it = integrals.find(values[i]); it != integrals.end())
              it->second.second.push_back(*f);
          }
        }
      }
    }
    else
    {
      // For cell and vertex integrals use all markers (but not on ghost
      // entities)
      for (auto e = tagged_entities.begin(); e != entity_end; ++e)
      {
        const std::size_t i = std::distance(tagged_entities.cbegin(), e);
        if (auto it = integrals.find(values[i]); it != integrals.end())
          it->second.second.push_back(*e);
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

    // Cells. If there is a default integral, define it on all owned cells
    if (auto kernels = _integrals.find(IntegralType::cell);
        kernels != _integrals.end())
    {
      if (auto it = kernels->second.find(-1); it != kernels->second.end())
      {
        std::vector<std::int32_t>& active_entities = it->second.second;
        const int num_cells = topology.index_map(tdim)->size_local();
        active_entities.resize(num_cells);
        std::iota(active_entities.begin(), active_entities.end(), 0);
      }
    }

    // Exterior facets. If there is a default integral, define it only
    // on owned surface facets.
    if (auto kernels = _integrals.find(IntegralType::exterior_facet);
        kernels != _integrals.end())
    {
      if (auto it = kernels->second.find(-1); it != kernels->second.end())
      {
        std::vector<std::int32_t>& active_entities = it->second.second;
        active_entities.clear();

        // Get number of facets owned by this process
        mesh.topology_mutable().create_connectivity(tdim - 1, tdim);
        auto f_to_c = topology.connectivity(tdim - 1, tdim);
        assert(topology.index_map(tdim - 1));
        std::set<std::int32_t> fwd_shared_facets;

        // Only need to consider shared facets when there are no ghost cells
        if (topology.index_map(tdim)->num_ghosts() == 0)
        {
          fwd_shared_facets.insert(
              topology.index_map(tdim - 1)->shared_indices().begin(),
              topology.index_map(tdim - 1)->shared_indices().end());
        }

        const int num_facets = topology.index_map(tdim - 1)->size_local();
        for (int f = 0; f < num_facets; ++f)
        {
          if (f_to_c->num_links(f) == 1
              and fwd_shared_facets.find(f) == fwd_shared_facets.end())
          {
            active_entities.push_back(f);
          }
        }
      }
    }

    // Interior facets. If there is a default integral, define it only on
    // owned interior facets.
    if (auto kernels = _integrals.find(IntegralType::interior_facet);
        kernels != _integrals.end())
    {
      if (auto it = kernels->second.find(-1); it != kernels->second.end())
      {
        std::vector<std::int32_t>& active_entities = it->second.second;

        // Get number of facets owned by this process
        mesh.topology_mutable().create_connectivity(tdim - 1, tdim);
        assert(topology.index_map(tdim - 1));
        const int num_facets = topology.index_map(tdim - 1)->size_local();
        auto f_to_c = topology.connectivity(tdim - 1, tdim);
        active_entities.clear();
        active_entities.reserve(num_facets);
        for (int f = 0; f < num_facets; ++f)
        {
          if (f_to_c->num_links(f) == 2)
            active_entities.push_back(f);
        }
      }
    }
  }

  // Function spaces (one for each argument)
  std::vector<std::shared_ptr<const function::FunctionSpace>> _function_spaces;

  // Form coefficients
  std::vector<std::shared_ptr<const function::Function<T>>> _coefficients;

  // Constants associated with the Form
  std::vector<std::shared_ptr<const function::Constant<T>>> _constants;

  // The mesh
  std::shared_ptr<const mesh::Mesh> _mesh;

  using kern
      = std::function<void(T*, const T*, const T*, const double*, const int*,
                           const std::uint8_t*, const std::uint32_t)>;
  std::map<IntegralType,
           std::map<int, std::pair<kern, std::vector<std::int32_t>>>>
      _integrals;

  // True if permutation data needs to be passed into these integrals
  bool _needs_permutation_data;

}; // namespace fem

} // namespace fem
} // namespace dolfinx
