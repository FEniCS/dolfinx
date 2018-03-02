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

using namespace dolfin;
using namespace refinement;

//-----------------------------------------------------------------------------
mesh::Mesh dolfin::refinement::refine(const mesh::Mesh& mesh, bool redistribute)
{
  if (mesh.type().cell_type() != mesh::CellType::Type::triangle
      and mesh.type().cell_type() != mesh::CellType::Type::tetrahedron)
  {
    log::dolfin_error("refine.cpp", "refine mesh",
                      "Refinement only defined for simplices");
  }

  mesh::Mesh refined_mesh(mesh.mpi_comm());
  PlazaRefinementND::refine(refined_mesh, mesh, redistribute);

  // Report the number of refined cells
  const std::size_t D = mesh.topology().dim();
  const std::size_t n0 = mesh.num_entities_global(D);
  const std::size_t n1 = refined_mesh.num_entities_global(D);
  log::log(TRACE, "Number of cells increased from %d to %d (%.1f%% increase).",
           n0, n1,
           100.0 * (static_cast<double>(n1) / static_cast<double>(n0) - 1.0));

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
    log::dolfin_error("refine.cpp", "refine mesh",
                      "Refinement only defined for simplices");
  }

  mesh::Mesh refined_mesh(mesh.mpi_comm());
  PlazaRefinementND::refine(refined_mesh, mesh, cell_markers, redistribute);

  // Report the number of refined cells
  const std::size_t D = mesh.topology().dim();
  const std::size_t n0 = mesh.num_entities_global(D);
  const std::size_t n1 = refined_mesh.num_entities_global(D);
  log::log(TRACE, "Number of cells increased from %d to %d (%.1f%% increase).",
           n0, n1,
           100.0 * (static_cast<double>(n1) / static_cast<double>(n0) - 1.0));

  return refined_mesh;
}
//-----------------------------------------------------------------------------
