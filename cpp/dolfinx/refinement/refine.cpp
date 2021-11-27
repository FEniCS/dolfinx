// Copyright (C) 2010 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "refine.h"
#include "plaza.h"
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/log.h>
#include <dolfinx/mesh/Mesh.h>

using namespace dolfinx;

//-----------------------------------------------------------------------------
std::vector<std::int32_t>
refinement::compute_marked_edges(const mesh::Mesh& mesh,
                                 const xtl::span<const std::int32_t> entities,
                                 int dim)
{
  auto map_e = mesh.topology().index_map(1);
  assert(map_e);
  auto map_ent = mesh.topology().index_map(dim);
  assert(map_ent);
  auto ent_to_edge = mesh.topology().connectivity(dim, 1);
  if (!ent_to_edge)
  {
    throw std::runtime_error("Connectivity missing: (" + std::to_string(dim)
                             + ", 1)");
  }

  std::vector<std::int32_t> edges;
  for (std::int32_t entity : entities)
  {
    auto e = ent_to_edge->links(entity);
    edges.insert(edges.end(), e.begin(), e.end());
  }

  std::sort(edges.begin(), edges.end());
  edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
  return edges;
}
//-----------------------------------------------------------------------------
mesh::Mesh refinement::refine(const mesh::Mesh& mesh, bool redistribute)
{
  if (mesh.topology().cell_type() != mesh::CellType::triangle
      and mesh.topology().cell_type() != mesh::CellType::tetrahedron)
  {
    throw std::runtime_error("Refinement only defined for simplices");
  }

  mesh::Mesh refined_mesh = plaza::refine(mesh, redistribute);

  // Report the number of refined cells
  const int D = mesh.topology().dim();
  const std::int64_t n0 = mesh.topology().index_map(D)->size_global();
  const std::int64_t n1 = refined_mesh.topology().index_map(D)->size_global();
  LOG(INFO) << "Number of cells increased from " << n0 << " to " << n1 << " ("
            << 100.0 * (static_cast<double>(n1) / static_cast<double>(n0) - 1.0)
            << "%% increase).";

  return refined_mesh;
}
//-----------------------------------------------------------------------------
mesh::Mesh refinement::refine(const mesh::Mesh& mesh,
                              const xtl::span<const std::int32_t>& edges,
                              bool redistribute)
{
  if (mesh.topology().cell_type() != mesh::CellType::triangle
      and mesh.topology().cell_type() != mesh::CellType::tetrahedron)
  {
    throw std::runtime_error("Refinement only defined for simplices");
  }

  mesh::Mesh refined_mesh = plaza::refine(mesh, edges, redistribute);

  // Report the number of refined cells
  const int D = mesh.topology().dim();
  const std::int64_t n0 = mesh.topology().index_map(D)->size_global();
  const std::int64_t n1 = refined_mesh.topology().index_map(D)->size_global();
  LOG(INFO) << "Number of cells increased from " << n0 << " to " << n1 << " ("
            << 100.0 * (static_cast<double>(n1) / static_cast<double>(n0) - 1.0)
            << "%% increase).";

  return refined_mesh;
}
//-----------------------------------------------------------------------------
