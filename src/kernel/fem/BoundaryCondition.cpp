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
#include <dolfin/Vertex.h>

#include <dolfin/UFCMesh.h>
#include <dolfin/UFCCell.h>
#include <ufc.h>

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
                           ufc::finite_element& element)
{
  dolfin_error("Application of boundary conditions to a matrix is not yet implemented.");
}
//-----------------------------------------------------------------------------
void BoundaryCondition::applyBC(GenericMatrix& A, Mesh& mesh, 
                                ufc::finite_element& element, ufc::dof_map& dof_map)
{
  dolfin_warning("Application of boundary conditions not yet implemented.");
  apply(&A, 0, 0, mesh, element, dof_map);
}
//-----------------------------------------------------------------------------
void BoundaryCondition::applyBC(GenericVector& b, Mesh& mesh, 
                                ufc::finite_element& element)
{
  dolfin_error("Application of boundary conditions to a vector is not yet implemented.");
}
//-----------------------------------------------------------------------------
void BoundaryCondition::apply(GenericMatrix* A, GenericVector* b, 
            const GenericVector* x, Mesh& mesh,  ufc::finite_element& element, 
            ufc::dof_map& dof_map)
{
  cout << "Applying boundary conditions " << endl;

  // Create boundary value
  BoundaryValue bv;

  // Create boundary mesh
  MeshFunction<uint> vertex_map;
  MeshFunction<uint> cell_map;
  BoundaryMesh boundary(mesh, vertex_map, cell_map);

  // Get number of dofs per facet
  const uint num_facet_dofs = dof_map.num_facet_dofs();

  // Create array for dof mapping
  uint* dofs = new uint[num_facet_dofs];

  UFCMesh ufc_mesh(mesh);

  // Iterate over all cells in the boundary mesh
  Progress p("Applying Dirichlet boundary conditions", boundary.numCells());
  for (CellIterator boundary_cell(boundary); !boundary_cell.end(); ++boundary_cell)
  {
    // Get mesh facet corresponding to boundary cell
    Facet mesh_facet(mesh, cell_map(*boundary_cell));

    // Get mesh cell to which mesh facet belongs (pick first, there is only one)
    dolfin_assert(mesh_facet.numEntities(mesh.topology().dim()) == 1);
    Cell mesh_cell(mesh, mesh_facet.entities(mesh.topology().dim())[0]);

    // Get local index of facet with respect to the cell
    uint local_facet = mesh_cell.index(mesh_facet);

    // Tabulate dof mapping for facet (this should really come from DofMap)
    UFCCell ufc_cell(mesh_cell);
    dof_map.tabulate_facet_dofs(dofs, ufc_mesh, ufc_cell, local_facet);

    // FIXME: Waiting on FFC to genrate the necessary UFC functions and the see
    //        what the UFC/DOLFIN function interface looks like.  

    // Evaluate boundary condition function at facet nodes
    //real evaluate_dof(unsigned int i, const ufc::function& f, const ufc::cell& c) const
//    real value = evaluate_dof(i, f, ufc_cell)

    dolfin_error("Work on applying boundary conditions not complete");

    // Apply boundary condtions here

    p++;
  }
  delete [] dofs;
}
//-----------------------------------------------------------------------------


