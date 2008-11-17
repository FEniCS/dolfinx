// Copyright (C) 2008 Anders Logg and Magnus Vikstrøm.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-02-04
// Last changed: 2008-04-12
//
// This demonstrates parallel assembly in DOLFIN. Parallel
// assembly is currently experimental and exists in the form
// of the alternate assembler pAssembler. This will be merged
// with the main DOLFIN assembler.
//
// To run this demo in parallel, type
//
//     mpirun -n <num_processes> demo
//
// Note that currently one needs to edit the code generated
// by FFC and replace dolfin::Form with dolfin::pForm. This
// extra step will be removed once Assembler and pAssembler
// have merged.

#include <dolfin.h>
#include "ReactionDiffusion.h"

using namespace dolfin;

int main()
{
  message("Sorry, this demo is currently broken.");
  return 0;

//#ifndef HAS_SCOTCH
//  message("Sorry, this demo requires SCOTCH.");
//  return 0;
//#endif

  // Create mesh
  UnitCube mesh(20, 20, 20);

  // Create function space and form
  ReactionDiffusionFunctionSpace V(mesh); 
  ReactionDiffusionBilinearForm a(V, V); 

  // Partition mesh
  MeshFunction<dolfin::uint> partitions(mesh);
  mesh.partition(partitions);

  // Assemble matrix using parallel assembler
  //Matrix A;
  //pAssembler::assemble(mesh, partitions);
 
  return 0;
}
