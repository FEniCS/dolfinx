// Copyright (C) 2010-2021 Garth N. Wells
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
mesh::Mesh refinement::refine(const mesh::Mesh& mesh, bool redistribute)
{
  if (mesh.topology().cell_type() != mesh::CellType::triangle
      and mesh.topology().cell_type() != mesh::CellType::tetrahedron)
  {
    throw std::runtime_error("Refinement only defined for simplices");
  }

  auto [refined_mesh, parent_cell, parent_facet]
      = plaza::refine(mesh, redistribute, plaza::RefinementOptions::none);

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
                              std::span<const std::int32_t> edges,
                              bool redistribute)
{
  if (mesh.topology().cell_type() != mesh::CellType::triangle
      and mesh.topology().cell_type() != mesh::CellType::tetrahedron)
  {
    throw std::runtime_error("Refinement only defined for simplices");
  }

  auto [refined_mesh, parent_cell, parent_facet] = plaza::refine(
      mesh, edges, redistribute, plaza::RefinementOptions::none);

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
