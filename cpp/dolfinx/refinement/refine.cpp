// Copyright (C) 2010 Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "refine.h"
#include "PlazaRefinementND.h"
#include <dolfinx/common/log.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/MeshTags.h>

using namespace dolfinx;
using namespace refinement;

//-----------------------------------------------------------------------------
mesh::Mesh dolfinx::refinement::refine(const mesh::Mesh& mesh,
                                       bool redistribute)
{
  if (mesh.topology().cell_type() != mesh::CellType::triangle
      and mesh.topology().cell_type() != mesh::CellType::tetrahedron)
  {
    throw std::runtime_error("Refinement only defined for simplices");
  }

  mesh::Mesh refined_mesh = PlazaRefinementND::refine(mesh, redistribute);

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
mesh::Mesh
dolfinx::refinement::refine(const mesh::Mesh& mesh,
                            const mesh::MeshTags<std::int8_t>& cell_markers,
                            bool redistribute)
{
  if (mesh.topology().cell_type() != mesh::CellType::triangle
      and mesh.topology().cell_type() != mesh::CellType::tetrahedron)
  {
    throw std::runtime_error("Refinement only defined for simplices");
  }

  mesh::Mesh refined_mesh
      = PlazaRefinementND::refine(mesh, cell_markers, redistribute);

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
