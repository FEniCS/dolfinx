// Copyright (C) 2007 Anders Logg and ...
// Licensed under the GNU GPL Version 2.
//
// First added:  2007-01-17
// Last changed: 2007-03-01

#include <dolfin/dolfin_log.h>
#include <dolfin/Array.h>
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

  // Initialize mesh entities used by dof maps
  initMeshEntities(mesh, data);

  // Initialize global tensor
  initGlobalTensor(A, mesh, data);

  // Assemble over cells
  assembleCells(A, mesh, data);

  // Assemble over exterior facets
  assembleExteriorFacets(A, mesh, data);

  // Assemble over interior facets
  assembleInteriorFacets(A, mesh, data);
}
//-----------------------------------------------------------------------------
void Assembler::assembleCells(GenericTensor& A,
                              Mesh& mesh, AssemblyData& data) const
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
                                       Mesh& mesh, AssemblyData& data) const
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
                                       Mesh& mesh, AssemblyData& data) const
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
void Assembler::initMeshEntities(Mesh& mesh, AssemblyData& data) const
{
  dolfin_info("Initializing mesh entities.");
  
  // Array of mesh entities used by dof maps
  Array<bool> entities(mesh.topology().dim() + 1);
  entities = false;

  // Iterate over all dof maps and mark entities
  for (uint i = 0; i < data.num_arguments; i++)
  {
    // Iterate over topological dimensions
    for (uint d = 0; d <= mesh.topology().dim(); d++)
    {
      if ( data.dof_maps[i]->needs_mesh_entities(d) )
        entities[d] = true;
    }
  }

  // Compute mesh entitites
  for (uint d = 0; d <= mesh.topology().dim(); d++)
  {
    if ( entities[d] )
      mesh.init(d);
  }
}
//-----------------------------------------------------------------------------
void Assembler::initGlobalTensor(GenericTensor& A, Mesh& mesh,
                                 AssemblyData& data) const
{
  dolfin_info("Initializing global tensor.");

  

}
//-----------------------------------------------------------------------------
