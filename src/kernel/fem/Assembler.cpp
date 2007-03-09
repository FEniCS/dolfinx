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
  
  // Assemble over exterior facets (the cells of the boundary)
  Progress p("Assembling over exterior facets", boundary.numCells());
  for (CellIterator boundary_cell(boundary); !boundary_cell.end(); ++boundary_cell)
  {
    // Get mesh facet corresponding to boundary cell
    Facet mesh_facet(mesh, cell_map(*boundary_cell));

    // Get mesh cell to which mesh facet belongs (pick first, there is only one)
    dolfin_assert(mesh_facet.numEntities(mesh.topology().dim()) == 1);
    Cell mesh_cell(mesh, mesh_facet.entities(mesh.topology().dim())[0]);

    // Get local index of facet with respect to the cell
    uint local_facet = mesh_cell.index(mesh_facet);
      
    // Update to current cell
    ufc.update(mesh_cell);

    // Compute local-to-global map for each dimension
    for (uint i = 0; i < ufc.form.rank(); i++)
      ufc.dof_maps[i]->tabulate_dofs(ufc.dofs[i], ufc.mesh, ufc.cell);

    // Tabulate exterior facet tensor
    ufc.exterior_facet_integral->tabulate_tensor(ufc.A, ufc.w, ufc.cell, local_facet);

    // Add entries to global tensor
    A.add(ufc.A, ufc.local_dimensions, ufc.dofs);

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

  // Compute facets and facet - cell connectivity if not already computed
  mesh.init(mesh.topology().dim() - 1);
  mesh.init(mesh.topology().dim() - 1, mesh.topology().dim());

  // Assemble over interior facets (the facets of the mesh)
  Progress p("Assembling over interior facets", mesh.numFacets());
  for (FacetIterator facet(mesh); !facet.end(); ++facet)
  {
    // Check if we have an interior facet
    if ( facet->numEntities(mesh.topology().dim()) != 2 )
    {
      p++;
      continue;
    }

    // Get cells incident with facet
    Cell cell0(mesh, facet->entities(mesh.topology().dim())[0]);
    Cell cell1(mesh, facet->entities(mesh.topology().dim())[1]);
      
    // Get local index of facet with respect to each cell
    uint facet0 = cell0.index(*facet);
    uint facet1 = cell1.index(*facet);

    // Update to current pair of cells
    ufc.update(cell0, cell1);

    // Compute local-to-global map for each dimension on macro element
    for (uint i = 0; i < ufc.form.rank(); i++)
    {
      const uint offset = ufc.local_dimensions[i];
      ufc.dof_maps[i]->tabulate_dofs(ufc.macro_dofs[i], ufc.mesh, ufc.cell0);
      ufc.dof_maps[i]->tabulate_dofs(ufc.macro_dofs[i] + offset*0, ufc.mesh, ufc.cell1);
    }

    // Tabulate exterior interior facet tensor on macro element
    ufc.interior_facet_integral->tabulate_tensor(ufc.macro_A, ufc.w, ufc.cell0, ufc.cell1, facet0, facet1);

    // Add entries to global tensor
    A.add(ufc.macro_A, ufc.macro_local_dimensions, ufc.macro_dofs);

    p++;
  }
}
//-----------------------------------------------------------------------------
