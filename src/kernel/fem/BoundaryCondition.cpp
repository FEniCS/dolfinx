// Copyright (C) 2007 Anders Logg and Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-04-10
// Last changed: 2007-06-05

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
  apply(A, b, 0, form);
}
//-----------------------------------------------------------------------------
void BoundaryCondition::apply(GenericMatrix& A, GenericVector& b, 
                              const GenericVector& x, const Form& form)
{
  apply(A, b, &x, form);
}
//-----------------------------------------------------------------------------
void BoundaryCondition::apply(GenericMatrix& A, GenericVector& b,
                              const GenericVector* x, const Form& form)
{
  cout << "Applying boundary conditions to linear system." << endl;

  // FIXME: How do we reuse the dof map for u?
  // FIXME: Perhaps we should make DofMaps a member of Form?
  
  // Create local data for application of boundary conditions
  LocalData data(form, mesh, sub_system);

  // Make sure we have the facet - cell connectivity
  const uint D = mesh.topology().dim();
  mesh.init(D - 1, D);
  
  // Create UFC view of mesh
  UFCMesh ufc_mesh(mesh);
 
  // A set to hold dofs to which Dirichlet boundary conditions are applied
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
    g.interpolate(data.w, ufc_cell, *data.finite_element, cell, local_facet);
    
    // Tabulate dofs on cell
    data.dof_map->tabulate_dofs(data.cell_dofs, ufc_mesh, ufc_cell);

    // Tabulate which dofs are on the facet
    data.dof_map->tabulate_facet_dofs(data.facet_dofs, local_facet);
    
    //for (uint i = 0; i < 12; i++)
    //  message("cell_dofs[%d] = %d", i, data.cell_dofs[i]);

    //for (uint i = 0; i < 6; i++)
    //  message("facet_dofs[%d] = %d", i, data.facet_dofs[i]);


    // Pick values for facet
    for (uint i = 0; i < data.dof_map->num_facet_dofs(); i++)
    {
      data.rows[i] = data.offset + data.cell_dofs[data.facet_dofs[i]];
      row_set.insert(data.rows[i]);
      data.values[i] = data.w[data.facet_dofs[i]];
    } 
 
    // Get current solution values for nonlinear problems
    if ( x )
    {
      x->get(data.x_values, data.dof_map->num_facet_dofs(), data.rows);
      for (uint i = 0; i < data.dof_map->num_facet_dofs(); i++)
        data.values[i] = data.values[i] - data.x_values[i];
    } 

    // Modify RHS vector for facet dofs (b[i] = value)
    b.set(data.values, data.dof_map->num_facet_dofs(), data.rows);

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
BoundaryCondition::LocalData::LocalData(const Form& form,
                                        Mesh& mesh,
                                        const SubSystem& sub_system)
  : finite_element(0), dof_map(0), offset(0),
    w(0), cell_dofs(0), values(0), x_values(0), facet_dofs(0), rows(0)
{
  // FIXME: Change behaviour of num_sub_elements() in FFC (return 0 when
  // FIXME: there are no nested elements

  // Check arity of form
  if (form.form().rank() != 2)
    error("Form must be bilinear for application of boundary conditions.");

  // Create finite element and dof map for solution (second argument of form)
  finite_element = form.form().create_finite_element(1);
  dof_map = form.form().create_dof_map(1);
  
  // Extract sub element and sub dof map if we have a sub system
  if (sub_system.depth() > 0)
  {
    // Finite element
    ufc::finite_element* sub_finite_element = sub_system.extractFiniteElement(*finite_element);
    delete finite_element;
    finite_element = sub_finite_element;

    // Dof map
    ufc::dof_map* sub_dof_map = sub_system.extractDofMap(*dof_map, mesh, offset);
    delete dof_map;
    dof_map = sub_dof_map;
  }

  // Create local data used to set boundary conditions
  w = new real[finite_element->space_dimension()];
  cell_dofs = new uint[finite_element->space_dimension()];
  for (uint i = 0; i < finite_element->space_dimension(); i++)
  {
    w[i] = 0.0;
    cell_dofs[i] = 0;
  }
  values = new real[dof_map->num_facet_dofs()];
  x_values = new real[dof_map->num_facet_dofs()];
  facet_dofs = new uint[dof_map->num_facet_dofs()];
  rows = new uint[dof_map->num_facet_dofs()];
  for (uint i = 0; i < dof_map->num_facet_dofs(); i++)
  {
    values[i] = 0.0;
    x_values[i] = 0.0;
    facet_dofs[i] = 0;
    rows[i] = 0;
  }
}
//-----------------------------------------------------------------------------
BoundaryCondition::LocalData::~LocalData()
{
  if (finite_element)
    delete finite_element;

  if (dof_map)
    delete dof_map;

  if (w)
    delete [] w;

  if (cell_dofs)
    delete [] cell_dofs;

  if (values)
    delete [] values;

  if (x_values)
    delete [] x_values;

  if (facet_dofs)
    delete [] facet_dofs;

  if (rows)
    delete [] rows;
}
//-----------------------------------------------------------------------------
