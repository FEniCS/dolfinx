// Copyright (C) 2018 Chris Richardson
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <array>
#include <dolfinx/mesh/MeshTags.h>
#include <functional>
#include <map>
#include <set>
#include <vector>

namespace dolfinx
{
namespace mesh
{
class Mesh;
} // namespace mesh

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

/// Integrals of a Form, including those defined over cells, interior
/// and exterior facets, and vertices.

template <typename T>
class FormIntegrals
{
public:
  /// Construct object from (index, tabulate function) pairs for
  /// different integral types
  /// @param[in] integrals For each integral type (domain index,
  ///   tabulate function) pairs for each integral. Domain index -1 means
  ///   for all entities.
  /// @param[in] needs_permutation_data Pass true if an integral
  ///   requires mesh entity permutation data
  FormIntegrals(
      const std::map<
          IntegralType,
          std::pair<
              std::vector<std::pair<
                  int, std::function<void(T*, const T*, const T*, const double*,
                                          const int*, const std::uint8_t*,
                                          const std::uint32_t)>>>,
              const mesh::MeshTags<int>*>>& integrals,
      bool needs_permutation_data)
      : _needs_permutation_data(needs_permutation_data)
  {
    // Loop over integrals by domain type (dimension)
    for (auto& integral_type : integrals)
    {
      // Add key to map
      auto it = _integrals.emplace(
          integral_type.first,
          std::map<int, std::pair<kern, std::vector<std::int32_t>>>());

      // Loop over integrals kernels
      for (auto& integral : integral_type.second.first)
        it.first->second.insert({integral.first, {integral.second, {}}});

      // Set domains
      if (integral_type.second.second)
        set_domains(integral_type.first, *integral_type.second.second);
    }
  }

  /// Copy constructor
  FormIntegrals(const FormIntegrals& integrals) = default;

  /// Move constructor
  FormIntegrals(FormIntegrals&& integrals) = default;

  /// Destructor
  ~FormIntegrals() = default;

  /// Get the function for 'tabulate_tensor' for integral i of given
  /// type
  /// @param[in] type Integral type
  /// @param[in] i Integral number
  /// @return Function to call for tabulate_tensor
  const std::function<void(T*, const T*, const T*, const double*, const int*,
                           const std::uint8_t*, const std::uint32_t)>&
  get_tabulate_tensor(IntegralType type, int i) const
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
  std::set<IntegralType> types() const
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

  /// Get the integer IDs of integrals of type t. The IDs correspond to
  /// the domains which the integrals are defined for in the form,
  /// except ID -1, which denotes the default integral.
  /// @param[in] type Integral type
  /// @return List of IDs for this integral
  std::vector<int> integral_ids(IntegralType type) const
  {
    std::vector<int> ids;
    // auto it = _integrals.find(type);
    if (auto it = _integrals.find(type); it != _integrals.end())
    {
      for (auto& kernel : it->second)
        ids.push_back(kernel.first);
    }
    return ids;
  }

  /// Get the list of active entities for the ith integral of type t.
  /// Note, these are not retrieved by ID, but stored in order. The IDs
  /// can be obtained with "FormIntegrals::integral_ids()". For cell
  /// integrals, a list of cells. For facet integrals, a list of facets
  /// etc.
  /// @param[in] type Integral type
  /// @param[in] i Integral number
  /// @return List of active entities for this integral
  const std::vector<std::int32_t>& integral_domains(IntegralType type,
                                                    int i) const
  {
    auto it0 = _integrals.find(type);
    if (it0 == _integrals.end())
      throw std::runtime_error("No kernels for requested type.");
    auto it1 = it0->second.find(i);
    if (it1 == it0->second.end())
      throw std::runtime_error("No kernel for requested index.");
    return it1->second.second;
  }

private:
  /// Set the valid domains for the integrals of a given type from a
  /// MeshTags "marker". Note the MeshTags is not stored, so if there
  /// any changes to the integration domain this must be called again.
  /// @param[in] type Integral type
  /// @param[in] marker MeshTags mapping entities to integrals
  void set_domains(IntegralType type, const mesh::MeshTags<int>& marker)
  {
    // std::vector<struct Integral>& integrals
    //     = _integrals.at(static_cast<int>(type));
    // if (integrals.size() == 0)
    //   return;

    std::shared_ptr<const mesh::Mesh> mesh = marker.mesh();
    const mesh::Topology& topology = mesh->topology();
    const int tdim = topology.dim();
    int dim = tdim;
    if (type == IntegralType::exterior_facet
        or type == IntegralType::interior_facet)
    {
      mesh->topology_mutable().create_connectivity(tdim - 1, tdim);
      dim = tdim - 1;
    }
    else if (type == IntegralType::vertex)
      dim = 0;

    if (dim != marker.dim())
    {
      throw std::runtime_error("Invalid MeshTags dimension:"
                               + std::to_string(marker.dim()));
    }

    // // Create a reverse map
    // std::map<int, int> id_to_integral;
    // for (std::size_t i = 0; i < integrals.size(); ++i)
    // {
    //   if (integrals[i].id != -1)
    //   {
    //     integrals[i].active_entities.clear();
    //     id_to_integral.insert({integrals[i].id, i});
    //   }
    // }

    auto it0 = _integrals.find(type);
    if (it0 == _integrals.end())
    {
      // TODO: Add warning
      return;
    }
    // std::vector<std::int32_t>& active_entities = it->second.second;
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
        // Only need to consider shared facets when there are no ghost cells
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

public:
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

  /// Get bool indicating whether permutation data needs to be passed
  /// into these integrals
  /// @return True if cell permutation data is required
  bool needs_permutation_data() const { return _needs_permutation_data; }

private:
  using kern
      = std::function<void(T*, const T*, const T*, const double*, const int*,
                           const std::uint8_t*, const std::uint32_t)>;
  std::map<IntegralType,
           std::map<int, std::pair<kern, std::vector<std::int32_t>>>>
      _integrals;

  // True if permutation data needs to be passed into these integrals
  bool _needs_permutation_data;
};
} // namespace fem
} // namespace dolfinx