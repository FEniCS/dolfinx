// Copyright (C) 2010 Garth N. Wells
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Anders Logg, 2010-2011.
//
// First added:  2010-02-10
// Last changed: 2013-01-13

#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshFunction.h>
#include "UniformMeshRefinement.h"
#include "LocalMeshRefinement.h"
#include "ParallelRefinement2D.h"
#include "ParallelRefinement3D.h"
#include "refine.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
dolfin::Mesh dolfin::refine(const Mesh& mesh, bool redistribute)
{
  Mesh refined_mesh;
  refine(refined_mesh, mesh, redistribute);
  return refined_mesh;
}
//-----------------------------------------------------------------------------
void dolfin::refine(Mesh& refined_mesh, const Mesh& mesh, bool redistribute)
{
  if(MPI::num_processes() == 1)
    UniformMeshRefinement::refine(refined_mesh, mesh);
  else if(mesh.topology().dim() == 2)
    ParallelRefinement2D::refine(refined_mesh, mesh, redistribute);
  else if(mesh.topology().dim() == 3)
    ParallelRefinement3D::refine(refined_mesh, mesh, redistribute);
  else
    dolfin_error("refine.cpp",
                 "refine mesh",
                 "Unknown dimension in parallel");
}
//-----------------------------------------------------------------------------
dolfin::Mesh dolfin::refine(const Mesh& mesh,
                            const MeshFunction<bool>& cell_markers,
                            bool redistribute)
{
  Mesh refined_mesh;
  refine(refined_mesh, mesh, cell_markers, redistribute);
  return refined_mesh;
}
//-----------------------------------------------------------------------------
void dolfin::refine(Mesh& refined_mesh, const Mesh& mesh,
                    const MeshFunction<bool>& cell_markers, bool redistribute)
{
  // Call local mesh refinement algorithm or parallel, as appropriate
  if (MPI::num_processes() == 1)
    LocalMeshRefinement::refine(refined_mesh, mesh, cell_markers);
  else if (mesh.topology().dim() == 2)
    ParallelRefinement2D::refine(refined_mesh, mesh, cell_markers, redistribute);
  else if (mesh.topology().dim() == 3)
    ParallelRefinement3D::refine(refined_mesh, mesh, cell_markers, redistribute);
  else
  {
    dolfin_error("refine.cpp",
                 "refine mesh",
                 "Unknown dimension in parallel");
  }
}
//-----------------------------------------------------------------------------
