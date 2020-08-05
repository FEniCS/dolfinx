// Copyright (C) 2018 Chris Richardson
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <array>
#include <dolfinx/mesh/MeshTags.h>
#include <functional>
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
          std::vector<std::pair<
              int, std::function<void(T*, const T*, const T*, const double*,
                                      const int*, const std::uint8_t*,
                                      const std::uint32_t)>>>>& integrals,
      bool needs_permutation_data)
      : _needs_permutation_data(needs_permutation_data)
  {
    for (auto& integral_type : integrals)
    {
      for (auto& integral : integral_type.second)
      {
        set_tabulate_tensor(integral_type.first, integral.first,
                            integral.second);
      }
    }
  };

  /// Get the function for 'tabulate_tensor' for integral i of given
  /// type
  /// @param[in] type Integral type
  /// @param[in] i Integral number
  /// @return Function to call for tabulate_tensor
  const std::function<void(T*, const T*, const T*, const double*, const int*,
                           const std::uint8_t*, const std::uint32_t)>&
  get_tabulate_tensor(IntegralType type, int i) const
  {
    return _integrals.at(static_cast<int>(type)).at(i).tabulate;
  }

  /// @todo Should this be removed
  ///
  /// Set the function for 'tabulate_tensor' for integral i of
  /// given type
  /// @param[in] type Integral type
  /// @param[in] i Integral number
  /// @param[in] fn tabulate function
  void set_tabulate_tensor(
      IntegralType type, int i,
      const std::function<void(T*, const T*, const T*, const double*,
                               const int*, const std::uint8_t*,
                               const std::uint32_t)>& fn)
  {
    std::vector<struct FormIntegrals::Integral>& integrals
        = _integrals.at(static_cast<int>(type));

    // Find insertion point
    int pos = 0;
    for (const auto& q : integrals)
    {
      if (q.id == i)
      {
        throw std::runtime_error("Integral with ID " + std::to_string(i)
                                 + " already exists");
      }
      else if (q.id > i)
        break;
      ++pos;
    }

    // Insert new Integral
    integrals.insert(integrals.begin() + pos, {fn, i, {}});
  }

  /// Get types of integrals in the form
  /// @return Integrals types
  std::set<IntegralType> types() const
  {
    static const std::array types{IntegralType::cell,
                                  IntegralType::exterior_facet,
                                  IntegralType::interior_facet};
    std::set<IntegralType> set;
    for (auto type : types)
      if (!_integrals.at(static_cast<int>(type)).empty())
        set.insert(type);
    return set;
  }

  /// Number of integrals of given type
  /// @param[in] type Integral type
  /// @return Number of integrals
  int num_integrals(IntegralType type) const
  {
    return _integrals.at(static_cast<int>(type)).size();
  }

  /// Get the integer IDs of integrals of type t. The IDs correspond to
  /// the domains which the integrals are defined for in the form,
  /// except ID -1, which denotes the default integral.
  /// @param[in] type Integral type
  /// @return List of IDs for this integral
  std::vector<int> integral_ids(IntegralType type) const
  {
    std::vector<int> ids;
    for (const auto& integral : _integrals[static_cast<int>(type)])
      ids.push_back(integral.id);
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
    return _integrals.at(static_cast<int>(type)).at(i).active_entities;
  }

  /// Set the valid domains for the integrals of a given type from a
  /// MeshTags "marker". Note the MeshTags is not stored, so if there
  /// any changes to the integration domain this must be called again.
  /// @param[in] type Integral type
  /// @param[in] marker MeshTags mapping entities to integrals
  void set_domains(IntegralType type, const mesh::MeshTags<int>& marker)
  {
    std::vector<struct Integral>& integrals
        = _integrals.at(static_cast<int>(type));
    if (integrals.size() == 0)
      return;

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

    // Create a reverse map
    std::map<int, int> id_to_integral;
    for (std::size_t i = 0; i < integrals.size(); ++i)
    {
      if (integrals[i].id != -1)
      {
        integrals[i].active_entities.clear();
        id_to_integral.insert({integrals[i].id, i});
      }
    }

    // Get mesh tag data
    const std::vector<int>& values = marker.values();
    const std::vector<std::int32_t>& tagged_entities = marker.indices();
    assert(topology.index_map(dim));
    const auto entity_end
        = std::lower_bound(tagged_entities.begin(), tagged_entities.end(),
                           topology.index_map(dim)->size_local());

    if (type == IntegralType::exterior_facet)
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
          const std::size_t i = std::distance(tagged_entities.cbegin(), f);

          // All "owned" facets connected to one cell, that are not shared,
          // should be external.
          if (f_to_c->num_links(*f) == 1
              and fwd_shared.find(*f) == fwd_shared.end())
          {
            if (auto it = id_to_integral.find(values[i]);
                it != id_to_integral.end())
            {
              integrals[it->second].active_entities.push_back(*f);
            }
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
            if (const auto it = id_to_integral.find(values[i]);
                it != id_to_integral.end())
            {
              integrals[it->second].active_entities.push_back(*f);
            }
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
        if (const auto it = id_to_integral.find(values[i]);
            it != id_to_integral.end())
        {
          integrals[it->second].active_entities.push_back(*e);
        }
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
    std::vector<struct Integral>& cell_integrals
        = _integrals[static_cast<int>(IntegralType::cell)];

    // Cells. If there is a default integral, define it on all owned cells
    if (cell_integrals.size() > 0 and cell_integrals[0].id == -1)
    {
      const int num_cells = topology.index_map(tdim)->size_local();
      cell_integrals[0].active_entities.resize(num_cells);
      std::iota(cell_integrals[0].active_entities.begin(),
                cell_integrals[0].active_entities.end(), 0);
    }

    // Exterior facets. If there is a default integral, define it only on
    // owned surface facets.
    std::vector<struct Integral>& exf_integrals
        = _integrals[static_cast<int>(IntegralType::exterior_facet)];
    if (exf_integrals.size() > 0 and exf_integrals[0].id == -1)
    {
      // If there is a default integral, define it only on surface facets
      exf_integrals[0].active_entities.clear();

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
          exf_integrals[0].active_entities.push_back(f);
      }
    }

    // Interior facets. If there is a default integral, define it only on
    // owned interior facets.
    std::vector<struct FormIntegrals::Integral>& inf_integrals
        = _integrals[static_cast<int>(IntegralType::interior_facet)];
    if (!inf_integrals.empty() and inf_integrals[0].id == -1)
    {
      // If there is a default integral, define it only on interior facets
      inf_integrals[0].active_entities.clear();

      // Get number of facets owned by this process
      mesh.topology_mutable().create_connectivity(tdim - 1, tdim);
      assert(topology.index_map(tdim - 1));
      const int num_facets = topology.index_map(tdim - 1)->size_local();
      auto f_to_c = topology.connectivity(tdim - 1, tdim);
      inf_integrals[0].active_entities.reserve(num_facets);
      for (int f = 0; f < num_facets; ++f)
      {
        if (f_to_c->num_links(f) == 2)
          inf_integrals[0].active_entities.push_back(f);
      }
    }
  }

  /// Get bool indicating whether permutation data needs to be passed
  /// into these integrals
  /// @return True if cell permutation data is required
  bool needs_permutation_data() const { return _needs_permutation_data; }

private:
  // Collect together the function, id, and indices of entities to
  // integrate on
  struct Integral
  {
    std::function<void(T*, const T*, const T*, const double*, const int*,
                       const std::uint8_t*, const std::uint32_t)>
        tabulate;
    int id;
    std::vector<std::int32_t> active_entities;
  };

  // Array of vectors of integrals, arranged by type (see Type enum, and
  // struct Integral above)
  std::array<std::vector<struct Integral>, 4> _integrals;

  // A bool indicating whether permutation data needs to be passed into
  // these integrals
  bool _needs_permutation_data;
};
} // namespace fem
} // namespace dolfinx
