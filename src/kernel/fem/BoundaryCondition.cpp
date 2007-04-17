// Copyright (C) 2007 Anders Logg and Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// First added:  2007-04-10
// Last changed: 2007-04-17

#include <dolfin/Mesh.h>
#include <dolfin/Vertex.h>
#include <dolfin/Cell.h>
#include <dolfin/Facet.h>
#include <dolfin/SubDomain.h>
#include <dolfin/Form.h>
#include <dolfin/UFCMesh.h>
#include <dolfin/UFCCell.h>
#include <dolfin/GenericMatrix.h>
#include <dolfin/GenericVector.h>
#include <dolfin/BoundaryCondition.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
BoundaryCondition::BoundaryCondition(Function& g,
                                     Mesh& mesh,
                                     SubDomain& sub_domain)
  : g(g), mesh(mesh), sub_domains(0), sub_domain(0),
    sub_domains_local(false)
{
  // Make sure we have the facets and the incident cells
  const uint D = mesh.topology().dim();
  mesh.init(D - 1);
  mesh.init(D - 1, D);
  
  // Compute sub domain markers
  sub_domains = new MeshFunction<uint>(mesh, D - 1);
  sub_domains_local = true;
  for (FacetIterator facet(mesh); !facet.end(); ++facet)
  {
    // Check if facet is on the boundary
    const bool on_boundary = facet->numEntities(D) == 1;

    // Mark facets with all vertices inside as sub domain 0, others as 1
    (*sub_domains)(*facet) = 0;
    for (VertexIterator vertex(facet); !vertex.end(); ++vertex)
    {
      if ( !sub_domain.inside(vertex->x(), on_boundary) )
        (*sub_domains)(*facet) = 1;
    }
  }
}
//-----------------------------------------------------------------------------
BoundaryCondition::BoundaryCondition(Function& g,
                                     Mesh& mesh,
                                     MeshFunction<uint>& sub_domains,
                                     uint sub_domain)
  : g(g), mesh(mesh), sub_domains(&sub_domains), sub_domain(sub_domain),
    sub_domains_local(false)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
BoundaryCondition::~BoundaryCondition()
{
  // Delete sub domain markers if created locally
  if ( sub_domains_local )
    delete sub_domains;
}
//-----------------------------------------------------------------------------
void BoundaryCondition::apply(GenericMatrix& A, GenericVector& b,
                              const Form& form)
{
  apply(A, b, form.form());
}
//-----------------------------------------------------------------------------
void BoundaryCondition::apply(GenericMatrix& A, GenericVector& b,
                              const ufc::form& form)
{
  cout << "Applying boundary conditions to linear system" << endl;

  // FIXME: How do we reuse the dof map for u?
  // FIXME: Perhaps we should make DofMaps a member of Form?

  // Create local data for solution u (second argument of form)
  ufc::dof_map* dof_map = form.create_dof_map(1);
  ufc::finite_element* finite_element = form.create_finite_element(1);
  real* w = new real[finite_element->space_dimension()];
  uint* cell_dofs = new uint[finite_element->space_dimension()];
  uint* facet_dofs = new uint[dof_map->num_facet_dofs()];
  uint* rows = new uint[dof_map->num_facet_dofs()];
  real* values = new real[dof_map->num_facet_dofs()];
  UFCMesh ufc_mesh(mesh);

  // Make sure we have the facets
  const uint D = mesh.topology().dim();
  mesh.init(D - 1);   
 
  // Iterate over the facets of the mesh
  Progress p("Applying Dirichlet boundary conditions", mesh.size(D - 1));
  for (FacetIterator facet(mesh); !facet.end(); ++facet)
  {
    // Skip facets not inside the sub domain
    if ( (*sub_domains)(*facet) != sub_domain )
      continue;

    // Get cell to which facet belongs (there may be two, but pick first)
    Cell cell(mesh, facet->entities(D)[0]);
    UFCCell ufc_cell(cell);

    // Get local index of facet with respect to the cell
    const uint local_facet = cell.index(*facet);

    // Interpolate function on cell
    g.interpolate(w, ufc_cell, *finite_element);
    
    // Tabulate dofs on cell
    dof_map->tabulate_dofs(cell_dofs, ufc_mesh, ufc_cell);

    // Tabulate which dofs are on the facet
    dof_map->tabulate_facet_dofs(facet_dofs, ufc_mesh, ufc_cell, local_facet);
    
    // Pick values for facet
    for (uint i = 0; i < dof_map->num_facet_dofs(); i++)
    {
      rows[i] = cell_dofs[facet_dofs[i]];
      values[i] = w[facet_dofs[i]];
    }    

    // Modify linear system for facet dofs (A_ij = delta_ij and b[i] = value)
    A.ident(rows, dof_map->num_facet_dofs());
    b.set(values, dof_map->num_facet_dofs(), rows);

    p++;
  }

  // Delete dof map data
  delete dof_map;
  delete finite_element;
  delete [] w;
  delete [] cell_dofs;
  delete [] facet_dofs;
  delete [] rows;
  delete [] values;
}
//-----------------------------------------------------------------------------
