// Copyright (C) 2007 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2007-01-17
// Last changed: 2007-03-02

#include <dolfin/dolfin_log.h>
#include <dolfin/Array.h>
#include <dolfin/GenericTensor.h>
#include <dolfin/Mesh.h>
#include <dolfin/Cell.h>
#include <dolfin/Facet.h>
#include <dolfin/BoundaryMesh.h>
#include <dolfin/MeshFunction.h>
#include <dolfin/UFC.h>
#include <dolfin/Assembler.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Assembler::Assembler()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Assembler::~Assembler()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void Assembler::assemble(GenericTensor& A, const ufc::form& form, Mesh& mesh)
{
  cout << "Assembling form " << form.signature() << endl;

  // Update dof maps
  dof_maps.update(form, mesh);

  // Create data structure for local assembly data
  UFC ufc(form, mesh, dof_maps);

  // Initialize global tensor
  A.init(ufc.form.rank(), ufc.global_dimensions);

  // Assemble over cells
  assembleCells(A, mesh, ufc);

  // Assemble over exterior facets
  assembleExteriorFacets(A, mesh, ufc);

  // Assemble over interior facets
  assembleInteriorFacets(A, mesh, ufc);
}
//-----------------------------------------------------------------------------
void Assembler::assembleCells(GenericTensor& A, Mesh& mesh, UFC& ufc) const
{
  // Skip assembly if there is no cell integral
  if ( !ufc.cell_integral )
    return;

  // Assemble over cells
  Progress p("Assembling over cells", mesh.numCells());
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    // Update to current cell
    ufc.update(*cell);

    // Compute local-to-global map for each dimension
    for (uint i = 0; i < ufc.form.rank(); i++)
      ufc.dof_maps[i]->tabulate_dofs(ufc.dofs[i], ufc.mesh, ufc.cell);

    // Tabulate cell tensor
    ufc.cell_integral->tabulate_tensor(ufc.A, ufc.w, ufc.cell);

    // Add entries to global tensor
    A.add(ufc.A, ufc.local_dimensions, ufc.dofs);

    p++;
  }
}
//-----------------------------------------------------------------------------
void Assembler::assembleExteriorFacets(GenericTensor& A,
                                       Mesh& mesh, UFC& ufc) const
{
  // Skip assembly if there is no exterior facet integral
  if ( !ufc.exterior_facet_integral )
    return;

  // Create boundary mesh
  MeshFunction<uint> vertex_map;
  MeshFunction<uint> cell_map;
  BoundaryMesh boundary(mesh, vertex_map, cell_map);
  
  // Assemble over exterior facets
  Progress p("Assembling over exterior facets", boundary.numCells());
  for (CellIterator boundary_cell(boundary); !boundary_cell.end(); ++boundary_cell)
  {
    cout << *boundary_cell << endl;
    p++;  
  }
}
//-----------------------------------------------------------------------------
void Assembler::assembleInteriorFacets(GenericTensor& A,
                                       Mesh& mesh, UFC& ufc) const
{
  // Skip assembly if there is no interior facet integral
  if ( !ufc.interior_facet_integral )
    return;

  // Assemble over interior facets
  Progress p("Assembling over interior facets", mesh.numFacets());
  for (FacetIterator facet(mesh); !facet.end(); ++facet)
  {
    cout << *facet << endl;
    p++;
  }
}
//-----------------------------------------------------------------------------
