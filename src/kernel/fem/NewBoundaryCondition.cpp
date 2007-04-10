// Copyright (C) 2007 Anders Logg and Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// First added:  2007-04-10
// Last changed: 2007-04-10

#include <dolfin/Mesh.h>
#include <dolfin/Facet.h>
#include <dolfin/Form.h>
#include <dolfin/NewBoundaryCondition.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
NewBoundaryCondition::NewBoundaryCondition(Function& g,
                                           Mesh& mesh,
                                           MeshFunction<uint>& sub_domains,
                                           uint sub_domain)
  : g(g), mesh(mesh), sub_domains(&sub_domains), sub_domain(sub_domain)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
NewBoundaryCondition::~NewBoundaryCondition()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void NewBoundaryCondition::apply(GenericMatrix& A, GenericVector& b,
                                 const Form& form)
{
  apply(A, b, form.form());
}
//-----------------------------------------------------------------------------
void NewBoundaryCondition::apply(GenericMatrix& A, GenericVector& b,
                                 const ufc::form& form)
{
  cout << "Applying boundary conditions to linear system" << endl;
   
  // Make sure we have the facets
  const uint D = mesh.topology().dim();
  mesh.init(D - 1);

  // Iterate over the facets of the mesh
  Progress p("Applying Dirichlet boundary conditions", mesh.size(D - 1));
  for (FacetIterator facet(mesh); !facet.end(); ++facet)
  {

    
    p++;
  }


  /*


  // Create finite elements
  ufc::finite_element** finite_elements = new ufc::finite_element*[form.form().rank()];
  for (uint i = 0; i < form.form().rank(); i++)
    finite_elements[i] = form.form().create_finite_element(i);

  // Create dof maps
  ufc::dof_map** dof_maps = new ufc::dof_map*[form.form().rank()];
  for (uint i = 0; i < form.form().rank(); i++)
    dof_maps[i] = form.form().create_dof_map(i);

  // Create boundary value
  BoundaryValue bv;

  // Create boundary mesh
  MeshFunction<uint> vertex_map;
  MeshFunction<uint> cell_map;
  BoundaryMesh boundary(mesh, vertex_map, cell_map);

  // Get number of dofs per facet
  uint* num_facet_dofs = new uint[form.form().rank()];
  for (uint i = 0; i < form.form().rank(); i++)
    num_facet_dofs[i] = dof_maps[i]->num_facet_dofs();

  // Create array for dof mapping
  uint** dofs = new uint*[form.form().rank()];
  for (uint i = 0; i < form.form().rank(); i++)
    dofs[i] = new uint[dof_maps[i]->num_facet_dofs()];


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
    for (uint i = 0; i < form.form().rank(); i++)
      dof_maps[i]->tabulate_facet_dofs(dofs[i], ufc_mesh, ufc_cell, local_facet);

    // FIXME
    // Set homogeneous Dirichlet boundary condition on entire boundary (for testing)
    // Waiting for tabulate_facet_dofs() to be implemented by FFC
    // Still need to figiure out how to set supplied boundary conditions.
    if( A )
      A->ident(dofs[0], num_facet_dofs[0]);
    if( b )
    {
      real* values = new real[form.form().rank()];
      for(uint i = 0; i < form.form().rank(); ++i)
        values[i] = 0.0;      
      b->set(values, num_facet_dofs, dofs); 
    }


    // FIXME: Waiting on FFC to generate the necessary UFC functions and the see
    //        what the UFC/DOLFIN function interface looks like.  

    // Evaluate boundary condition function at facet nodes
    //real evaluate_dof(unsigned int i, const ufc::function& f, const ufc::cell& c) const
//    real value = evaluate_dof(i, f, ufc_cell)

    dolfin_error("Work on applying boundary conditions not complete");

    p++;
  }
//  delete [] dofs;

  // Delete finite elements and dof maps
  for (uint i = 0; i < form.form().rank(); i++)
  {
    delete finite_elements[i];
    delete dof_maps[i];
    delete dofs[i];
  }
  delete [] finite_elements;
  delete [] dof_maps;
  delete [] dofs;
  delete [] num_facet_dofs;
  */
}
//-----------------------------------------------------------------------------
