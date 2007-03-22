// Copyright (C) 2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Garth N. Wells 2005, 2007.
//
// First added:  2005-05-02
// Last changed: 2007-03-28

#include <dolfin/BoundaryCondition.h>
#include <dolfin/BoundaryValue.h>
#include <dolfin/dolfin_log.h>
#include <dolfin/GenericMatrix.h>
#include <dolfin/GenericVector.h>
#include <dolfin/Mesh.h>
#include <dolfin/BoundaryMesh.h>
#include <dolfin/Cell.h>
#include <dolfin/Facet.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
BoundaryCondition::BoundaryCondition() : TimeDependent() 
{
  // Do nothing
}
//-----------------------------------------------------------------------------
BoundaryCondition::~BoundaryCondition()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void BoundaryCondition::applyBC(GenericMatrix& A, GenericVector& b, Mesh& mesh, 
                                FiniteElement& element)
{
  dolfin_error("Application of boundary conditions not yet implemented.");
}
//-----------------------------------------------------------------------------
void BoundaryCondition::applyBC(GenericMatrix& A, Mesh& mesh, 
                                FiniteElement& element)
{
  dolfin_error("Application of boundary conditions to a matrix is not yet implemented.");
}
//-----------------------------------------------------------------------------
void BoundaryCondition::applyBC(GenericVector& b, Mesh& mesh, 
                                FiniteElement& element)
{
  dolfin_error("Application of boundary conditions to a vector is not yet implemented.");
}
//-----------------------------------------------------------------------------
void BoundaryCondition::apply(GenericMatrix* A, GenericVector* b, 
                    const GenericVector* x, Mesh& mesh, FiniteElement& element)
{
  cout << "Applying boundary conditions " << endl;

  // Create boundary value
  BoundaryValue bv;

  // Create boundary mesh
  MeshFunction<uint> vertex_map;
  MeshFunction<uint> cell_map;
  BoundaryMesh boundary(mesh, vertex_map, cell_map);

  // Iterate over all cells in the boundary mesh
  Progress p("Applying Dirichlet boundary conditions", boundary.numCells());
  for (CellIterator boundary_cell(boundary); !boundary_cell.end(); ++boundary_cell)
  {
    // Get mesh facet corresponding to boundary cell
    Facet mesh_facet(mesh, cell_map(*boundary_cell));

    // Get mesh cell to which mesh facet belongs (pick first, there is only one)
    dolfin_assert(mesh_facet.numEntities(mesh.topology().dim()) == 1);
    Cell mesh_cell(mesh, mesh_facet.entities(mesh.topology().dim())[0]);


    // Apply boundary condtions here

    p++;
  }
}
//-----------------------------------------------------------------------------


