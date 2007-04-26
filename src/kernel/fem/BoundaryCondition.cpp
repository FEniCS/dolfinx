// Copyright (C) 2007 Anders Logg and Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-04-10
// Last changed: 2007-04-26

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
#include <dolfin/SubSystem.h>
#include <dolfin/BoundaryCondition.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
BoundaryCondition::BoundaryCondition(Function& g,
                                     Mesh& mesh,
                                     SubDomain& sub_domain)
  : g(g), mesh(mesh),
    sub_domains(0), sub_domain(0), sub_domains_local(false)
{
  // Initialize sub domain markers
  init(sub_domain);
}
//-----------------------------------------------------------------------------
BoundaryCondition::BoundaryCondition(Function& g,
                                     MeshFunction<uint>& sub_domains,
                                     uint sub_domain)
  : g(g), mesh(sub_domains.mesh()),
    sub_domains(&sub_domains), sub_domain(sub_domain), sub_domains_local(false)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
BoundaryCondition::BoundaryCondition(Function& g,
                                     Mesh& mesh,
                                     SubDomain& sub_domain,
                                     const SubSystem& sub_system)
  : g(g), mesh(mesh),
    sub_domains(0), sub_domain(0), sub_domains_local(false),
    sub_system(sub_system)
{
  // Set sub domain markers
  init(sub_domain);
}
//-----------------------------------------------------------------------------
BoundaryCondition::BoundaryCondition(Function& g,
                                     MeshFunction<uint>& sub_domains,
                                     uint sub_domain,
                                     const SubSystem& sub_system)
  : g(g), mesh(sub_domains.mesh()),
    sub_domains(&sub_domains), sub_domain(sub_domain), sub_domains_local(false),
    sub_system(sub_system)
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
  cout << "Applying boundary conditions to linear system" << endl;

  // FIXME: How do we reuse the dof map for u?
  // FIXME: Perhaps we should make DofMaps a member of Form?
  
  // Create local data for application of boundary conditions
  //LocalData data(form);
  
  // Create finite element and dof map for solution (second argument of form)
  ufc::dof_map* dof_map = form.form().create_dof_map(1);
  ufc::finite_element* finite_element = form.form().create_finite_element(1);

  // Create local data for solution u (second argument of form)
  real* w = new real[10*finite_element->space_dimension()];
  uint* cell_dofs = new uint[10*finite_element->space_dimension()];
  uint* facet_dofs = new uint[10*dof_map->num_facet_dofs()];
  uint* rows = new uint[10*dof_map->num_facet_dofs()];
  real* values = new real[10*dof_map->num_facet_dofs()];
  UFCMesh ufc_mesh(mesh);

  // Make sure we have the facets
  const uint D = mesh.topology().dim();
  mesh.init(D - 1);   
 
  // A set to hold dofs to which Dirichlet boundary conidtions are applied
  std::set<uint> row_set;
  row_set.clear();

  // Iterate over the facets of the mesh
  Progress p("Applying Dirichlet boundary conditions", mesh.size(D - 1));
  for (FacetIterator facet(mesh); !facet.end(); ++facet)
  {
    // Skip facets not inside the sub domain
    if ( (*sub_domains)(*facet) != sub_domain )
    {
      p++;
      continue;
    }

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
      row_set.insert( rows[i] );
      values[i] = w[facet_dofs[i]];
    }    

    // Modify RHS vector for facet dofs (b[i] = value)
    b.set(values, dof_map->num_facet_dofs(), rows);

    p++;
  }

  // Copy contents of boundary condition set into an array
  uint i = 0;
  uint* rows_temp = new uint[row_set.size()];
  std::set<uint>::const_iterator row;
  for(row = row_set.begin(); row != row_set.end(); ++row)
    rows_temp[i++] = *row;

  // Modify linear system for facet dofs (A_ij = delta_ij)
  A.ident(rows_temp, row_set.size());

  delete [] rows_temp;

  // Finalise changes to b
  b.apply();

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
void BoundaryCondition::init(SubDomain& sub_domain)
{
  cout << "Creating sub domain markers for boundary condition." << endl;

  // Create mesh function for sub domain markers on facets
  mesh.init(mesh.topology().dim() - 1);
  sub_domains = new MeshFunction<uint>(mesh, mesh.topology().dim() - 1);
  sub_domains_local = true;

  // Mark everything as sub domain 1
  (*sub_domains) = 1;
  
  // Mark the sub domain as sub domain 0
  sub_domain.mark(*sub_domains, 0);
}
//-----------------------------------------------------------------------------
/*
BoundaryCondition::LocalData::LocalData(const ufc::form& form)
{
  // Check arity of form
  if (form.rank() != 2)
    dolfin_error("Form must be bilinear for application of boundary conditions.");

  // Extract sub finite element for sub system
  //sub_element = SubSystem::extractFiniteElement(finite_element, sub_systems);


  // Extract sub dof map for sub system
  
  
}
//-----------------------------------------------------------------------------
BoundaryCondition::LocalData::~LocalData()
{


}
//-----------------------------------------------------------------------------
*/
