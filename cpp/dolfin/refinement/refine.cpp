// Copyright (C) 2010 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "refine.h"
#include "PlazaRefinementND.h"
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Edge.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/mesh/Vertex.h>
#define LOGURU_WITH_STREAMS 1
#include <dolfin/common/loguru.hpp>

using namespace dolfin;
using namespace refinement;

//-----------------------------------------------------------------------------
mesh::Mesh dolfin::refinement::refine(const mesh::Mesh& mesh, bool redistribute)
{
  if (mesh.type().cell_type() != mesh::CellType::Type::triangle
      and mesh.type().cell_type() != mesh::CellType::Type::tetrahedron)
  {
    LOG_S(ERROR) << "Refinement only defined for simplices";
    throw std::runtime_error("Non simplicial mesh");
  }

  mesh::Mesh refined_mesh = PlazaRefinementND::refine(mesh, redistribute);

  // Report the number of refined cells
  const std::size_t D = mesh.topology().dim();
  const std::size_t n0 = mesh.num_entities_global(D);
  const std::size_t n1 = refined_mesh.num_entities_global(D);
  LOG_S(INFO) << "Number of cells increased from " << n0 << " to " << n1 << " ("
            << 100.0 * (static_cast<double>(n1) / static_cast<double>(n0) - 1.0)
            << "%% increase).";

  return refined_mesh;
}
//-----------------------------------------------------------------------------
mesh::Mesh
dolfin::refinement::refine(const mesh::Mesh& mesh,
                           const mesh::MeshFunction<bool>& cell_markers,
                           bool redistribute)
{
  if (mesh.type().cell_type() != mesh::CellType::Type::triangle
      and mesh.type().cell_type() != mesh::CellType::Type::tetrahedron)
  {
    LOG_S(ERROR) << "Refinement only defined for simplices";
    throw std::runtime_error("Non simplicial mesh");
  }

  mesh::Mesh refined_mesh
      = PlazaRefinementND::refine(mesh, cell_markers, redistribute);

  // Report the number of refined cells
  const std::size_t D = mesh.topology().dim();
  const std::size_t n0 = mesh.num_entities_global(D);
  const std::size_t n1 = refined_mesh.num_entities_global(D);
  LOG_S(INFO) << "Number of cells increased from " << n0 << " to " << n1 << " ("
            << 100.0 * (static_cast<double>(n1) / static_cast<double>(n0) - 1.0)
            << "%% increase).";

  return refined_mesh;
}
//-----------------------------------------------------------------------------
