#pragma once

#include <memory>
#include <set>
#include <stdexcept>

#include <dolfinx/common/types.h>
#include <dolfinx/la/SparsityPattern.h>

#include "Form.h"
#include "sparsitybuild.h"

namespace dolfinx::fem
{

/// @brief Create a sparsity pattern for a given form.
/// @note The pattern is not finalised, i.e. the caller is responsible
/// for calling SparsityPattern::assemble.
/// @param[in] a A bilinear form
/// @return The corresponding sparsity pattern
template <dolfinx::scalar T, std::floating_point U>
la::SparsityPattern create_sparsity_pattern(const Form<T, U>& a)
{
  if (a.rank() != 2)
  {
    throw std::runtime_error(
        "Cannot create sparsity pattern. Form is not a bilinear.");
  }

  // Get dof maps and mesh
  std::array<std::reference_wrapper<const DofMap>, 2> dofmaps{
      *a.function_spaces().at(0)->dofmap(),
      *a.function_spaces().at(1)->dofmap()};
  std::shared_ptr mesh = a.mesh();
  assert(mesh);

  std::shared_ptr mesh0 = a.function_spaces().at(0)->mesh();
  assert(mesh0);
  std::shared_ptr mesh1 = a.function_spaces().at(1)->mesh();
  assert(mesh1);

  const std::set<IntegralType> types = a.integral_types();
  if (types.find(IntegralType::interior_facet) != types.end()
      or types.find(IntegralType::exterior_facet) != types.end())
  {
    // FIXME: cleanup these calls? Some of the happen internally again.
    int tdim = mesh->topology()->dim();
    mesh->topology_mutable()->create_entities(tdim - 1);
    mesh->topology_mutable()->create_connectivity(tdim - 1, tdim);
  }

  common::Timer t0("Build sparsity");

  // Get common::IndexMaps for each dimension
  const std::array index_maps{dofmaps[0].get().index_map,
                              dofmaps[1].get().index_map};
  const std::array bs
      = {dofmaps[0].get().index_map_bs(), dofmaps[1].get().index_map_bs()};

  auto extract_cells = [](std::span<const std::int32_t> facets)
  {
    assert(facets.size() % 2 == 0);
    std::vector<std::int32_t> cells;
    cells.reserve(facets.size() / 2);
    for (std::size_t i = 0; i < facets.size(); i += 2)
      cells.push_back(facets[i]);
    return cells;
  };

  // Create and build sparsity pattern
  la::SparsityPattern pattern(mesh->comm(), index_maps, bs);
  for (auto type : types)
  {
    std::vector<int> ids = a.integral_ids(type);
    switch (type)
    {
    case IntegralType::cell:
      for (int id : ids)
      {
        sparsitybuild::cells(
            pattern, {a.domain(type, id, *mesh0), a.domain(type, id, *mesh1)},
            {{dofmaps[0], dofmaps[1]}});
      }
      break;
    case IntegralType::interior_facet:
      for (int id : ids)
      {
        sparsitybuild::interior_facets(
            pattern,
            {extract_cells(a.domain(type, id, *mesh0)),
             extract_cells(a.domain(type, id, *mesh1))},
            {{dofmaps[0], dofmaps[1]}});
      }
      break;
    case IntegralType::exterior_facet:
      for (int id : ids)
      {
        sparsitybuild::cells(pattern,
                             {extract_cells(a.domain(type, id, *mesh0)),
                              extract_cells(a.domain(type, id, *mesh1))},
                             {{dofmaps[0], dofmaps[1]}});
      }
      break;
    default:
      throw std::runtime_error("Unsupported integral type");
    }
  }

  t0.stop();

  return pattern;
}

} // dolfinx::fem