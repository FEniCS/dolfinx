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

  // FIXME: Should iterate over all nodes, but don't know how to do this yet
  //        Curently limited to scalar linear elements only.
  const ufc::shape cell_shape = element.cell_shape();
  const uint space_dimension = element.space_dimension();
  if(cell_shape == ufc::triangle)
    dolfin_assert(space_dimension == 3);
  if(cell_shape == ufc::tetrahedron)
    dolfin_assert(space_dimension == 4);

  UFCMesh ufc_mesh(mesh);
  uint* dofs = new uint[dof_map.local_dimension()];


  // Iterate over all cells in the boundary mesh
  Progress p("Applying Dirichlet boundary conditions", boundary.numCells());
  for (CellIterator boundary_cell(boundary); !boundary_cell.end(); ++boundary_cell)
  {
    // Get mesh facet corresponding to boundary cell
    Facet mesh_facet(mesh, cell_map(*boundary_cell));

    // Get mesh cell to which mesh facet belongs (pick first, there is only one)
    dolfin_assert(mesh_facet.numEntities(mesh.topology().dim()) == 1);
    Cell mesh_cell(mesh, mesh_facet.entities(mesh.topology().dim())[0]);

    // Compute local-to-global map
    UFCCell ufc_cell(mesh_cell);
    dof_map.tabulate_dofs(dofs, ufc_mesh, ufc_cell);

    // Iterate over cell vertexes
    for (VertexIterator vertex(mesh_cell); !vertex.end(); ++vertex)
    {
      cout << "Point " << vertex->point() << endl; 
      dolfin_error("Work on applying boundary conditions not complete");
    }

    // Apply boundary condtions here

    p++;
  }
  delete [] dofs;
}
//-----------------------------------------------------------------------------


