// Copyright (C) 2007 Anders Logg and ...
// Licensed under the GNU GPL Version 2.
//
// First added:  2007-01-17
// Last changed: 2007-01-17

#include <dolfin/dolfin_log.h>
#include <dolfin/GenericTensor.h>
#include <dolfin/Mesh.h>
#include <dolfin/Cell.h>
#include <dolfin/Facet.h>
#include <dolfin/BoundaryMesh.h>
#include <dolfin/MeshFunction.h>
#include <dolfin/AssemblyData.h>
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

  // Create data structure for local assembly data
  AssemblyData data(form);

  // Update dof map storage for current form
  dof_maps.update(form, mesh);

  // Assemble over cells
  assembleCells(A, data, mesh);

  // Assemble over exterior facets
  assembleExteriorFacets(A, data, mesh);

  // Assemble over interior facets
  assembleInteriorFacets(A, data, mesh);
}
//-----------------------------------------------------------------------------
void Assembler::assembleCells(GenericTensor& A,
                              AssemblyData& data,
                              Mesh& mesh)
{
  // Skip assembly if there is no cell integral
  if ( !data.cell_integral )
    return;

  // Assemble over cells
  Progress p("Assembling over cells", mesh.numCells());
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    cout << *cell << endl;
    p++;
  }
}
//-----------------------------------------------------------------------------
void Assembler::assembleExteriorFacets(GenericTensor& A,
                                       AssemblyData& data,
                                       Mesh& mesh)
{
  // Skip assembly if there is no exterior facet integral
  if ( !data.exterior_facet_integral )
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
                                       AssemblyData& data,
                                       Mesh& mesh)
{
  // Skip assembly if there is no interior facet integral
  if ( !data.interior_facet_integral )
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
