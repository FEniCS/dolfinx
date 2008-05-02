// Copyright (C) 2007-2008 Anders Logg and Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Kristian Oelgaard, 2007
//
// First added:  2007-04-10
// Last changed: 2008-04-22

#include <dolfin/common/constants.h>
#include <dolfin/log/log.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/Vertex.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Facet.h>
#include <dolfin/mesh/Point.h>
#include <dolfin/mesh/SubDomain.h>
#include <dolfin/la/GenericMatrix.h>
#include <dolfin/la/GenericVector.h>
#include "Form.h"
#include "UFCMesh.h"
#include "UFCCell.h"
#include "SubSystem.h"
#include "DirichletBC.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
DirichletBC::DirichletBC(Function& g,
                         Mesh& mesh,
                         SubDomain& sub_domain,
                         BCMethod method)
  : BoundaryCondition(), g(g), _mesh(mesh),
    sub_domains(0), sub_domain(0), sub_domains_local(false), method(method),
    user_sub_domain(&sub_domain)

{
  // Initialize sub domain markers
  init(sub_domain);
}
//-----------------------------------------------------------------------------
DirichletBC::DirichletBC(Function& g,
                         MeshFunction<uint>& sub_domains,
                         uint sub_domain,
                         BCMethod method)
  : BoundaryCondition(), g(g), _mesh(sub_domains.mesh()),
    sub_domains(&sub_domains), sub_domain(sub_domain), sub_domains_local(false),
    method(method)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
DirichletBC::DirichletBC(Function& g,
                         Mesh& mesh,
                         SubDomain& sub_domain,
                         const SubSystem& sub_system,
                         BCMethod method)
  : BoundaryCondition(), g(g), _mesh(mesh),
    sub_domains(0), sub_domain(0), sub_domains_local(false),
    sub_system(sub_system), method(method), user_sub_domain(&sub_domain)
{
  // Set sub domain markers
  init(sub_domain);
}
//-----------------------------------------------------------------------------
DirichletBC::DirichletBC(Function& g,
                         MeshFunction<uint>& sub_domains,
                         uint sub_domain,
                         const SubSystem& sub_system,
                         BCMethod method)
  : BoundaryCondition(), g(g), _mesh(sub_domains.mesh()),
    sub_domains(&sub_domains), sub_domain(sub_domain), sub_domains_local(false),
    sub_system(sub_system), method(method)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
DirichletBC::DirichletBC(Function& g,
                         Mesh& mesh,
                         BCMethod method)
  : BoundaryCondition(), g(g), _mesh(mesh),
    sub_domains(0), sub_domain(0), sub_domains_local(false), method(method),
    user_sub_domain(0)
{
  // Create sub domain for entire boundary
  class EntireBoundary : public SubDomain
  {
  public:
    bool inside(const real* x, bool on_boundary) const
    {
      return on_boundary;
    }
  };

  EntireBoundary sub_domain;

  // Initialize sub domain markers
  init(sub_domain);
}
//-----------------------------------------------------------------------------
DirichletBC::~DirichletBC()
{
  // Delete sub domain markers if created locally
  if ( sub_domains_local )
    delete sub_domains;
}
//-----------------------------------------------------------------------------
void DirichletBC::apply(GenericMatrix& A, GenericVector& b, const Form& form)
{
  apply(A, b, 0, form.dofMaps()[1], form.form());
}
//-----------------------------------------------------------------------------
void DirichletBC::apply(GenericMatrix& A, GenericVector& b, const DofMap& dof_map,
                        const ufc::form& form)
{
  apply(A, b, 0, dof_map, form);
}
//-----------------------------------------------------------------------------
void DirichletBC::apply(GenericMatrix& A, GenericVector& b,
                        const GenericVector& x, const Form& form)
{
  apply(A, b, &x, form.dofMaps()[1], form.form());
}
//-----------------------------------------------------------------------------
void DirichletBC::apply(GenericMatrix& A, GenericVector& b,
                        const GenericVector& x, const DofMap& dof_map, const ufc::form& form)
{
  apply(A, b, &x, dof_map, form);
}
//-----------------------------------------------------------------------------
void DirichletBC::apply(GenericMatrix& A, GenericVector& b,
                        const GenericVector* x, const DofMap& dof_map, const ufc::form& form)
{
  // FIXME: How do we reuse the dof map for u?
  
  // Simple check
  const uint N = dof_map.global_dimension();
  if (N != A.size(0) || N != A.size(1))
    error("Incorrect dimension of matrix for application of boundary conditions. Did you assemble it on a different mesh?");
  if (N != b.size())
    error("Incorrect dimension of matrix for application of boundary conditions. Did you assemble it on a different mesh?");
  
  // Set message string
  std::string s;
  if (method == topological)
    s = "Computing Dirichlet boundary values";
  else if (method == geometrical)
    s = "Computing Dirichlet boundary values (geometrical approach)";
  else
    s = "Computing Dirichlet boundary values (pointwise approach)";
  
  // Make sure we have the facet - cell connectivity
  const uint D = _mesh.topology().dim();
  if (method == topological)
    _mesh.init(D - 1, D);

  // Create local data for application of boundary conditions
  BoundaryCondition::LocalData data(form, _mesh, dof_map, sub_system);
  
  // A map to hold the mapping from boundary dofs to boundary values
  std::map<uint, real> boundary_values;

  if (method == pointwise)
  {
    Progress p(s, _mesh.size(D));
    for (CellIterator cell(_mesh); !cell.end(); ++cell)
    {
      computeBCPointwise(boundary_values, *cell, data);
      p++;
    }
  }
  else
  {
    // Iterate over the facets of the mesh
    Progress p(s, _mesh.size(D - 1));
    for (FacetIterator facet(_mesh); !facet.end(); ++facet)
    {
      // Skip facets not inside the sub domain
      if ((*sub_domains)(*facet) != sub_domain)
      {
        p++;
        continue;
      }

      // Chose strategy
      if (method == topological)
        computeBCTopological(boundary_values, *facet, data);
      else
        computeBCGeometrical(boundary_values, *facet, data);
    
      // Update process
      p++;
    }
  }

  // Copy boundary value data to arrays
  uint* dofs = new uint[boundary_values.size()];
  real* values = new real[boundary_values.size()];
  std::map<uint, real>::const_iterator boundary_value;
  uint i = 0;
  for (boundary_value = boundary_values.begin(); boundary_value != boundary_values.end(); ++boundary_value)
  {
    dofs[i]     = boundary_value->first;
    values[i++] = boundary_value->second;
  }

  // Modify boundary values for nonlinear problems
  if (x)
  {
    real* x_values = new real[boundary_values.size()];
    x->get(x_values, boundary_values.size(), dofs);
    for (uint i = 0; i < boundary_values.size(); i++)
      values[i] -= x_values[i];
  }

  message("Applying boundary conditions to linear system.");
  
  // Modify RHS vector (b[i] = value)
  b.set(values, boundary_values.size(), dofs);

  // Modify linear system (A_ii = 1)
  A.ident(boundary_values.size(), dofs);

  // Clear temporary arrays
  delete [] dofs;
  delete [] values;

  // Finalise changes to A
  A.apply();

  // Finalise changes to b
  b.apply();
}
//-----------------------------------------------------------------------------
Mesh& DirichletBC::mesh()
{
  return _mesh;
}
//-----------------------------------------------------------------------------
void DirichletBC::init(SubDomain& sub_domain)
{
  cout << "Creating sub domain markers for boundary condition." << endl;

  // Create mesh function for sub domain markers on facets
  _mesh.init(_mesh.topology().dim() - 1);
  sub_domains = new MeshFunction<uint>(_mesh, _mesh.topology().dim() - 1);
  sub_domains_local = true;

  // Mark everything as sub domain 1
  (*sub_domains) = 1;
  
  // Mark the sub domain as sub domain 0
  sub_domain.mark(*sub_domains, 0);
}
//-----------------------------------------------------------------------------
void DirichletBC::computeBCTopological(std::map<uint, real>& boundary_values,
                                       Facet& facet,
                                       BoundaryCondition::LocalData& data)
{
  // Get cell to which facet belongs (there may be two, but pick first)
  Cell cell(_mesh, facet.entities(facet.dim() + 1)[0]);
  UFCCell ufc_cell(cell);
  
  // Get local index of facet with respect to the cell
  const uint local_facet = cell.index(facet);
  
  // Interpolate function on cell
  g.interpolate(data.w, ufc_cell, *data.finite_element, cell, local_facet);
  
  // Tabulate dofs on cell
  data.dof_map->tabulate_dofs(data.cell_dofs, ufc_cell);

  // Tabulate which dofs are on the facet
  data.dof_map->tabulate_facet_dofs(data.facet_dofs, local_facet);
  
  // Debugging print:
  /* 
  cout << endl << "Handling BC's for:" << endl;
  cout << "Cell:  " << facet.entities(facet.dim() + 1)[0] << endl;
  cout << "Facet: " << local_facet << endl;
  */
  
  // Pick values for facet
  for (uint i = 0; i < data.dof_map->num_facet_dofs(); i++)
  {
    const uint dof = data.offset + data.cell_dofs[data.facet_dofs[i]];
    const real value = data.w[data.facet_dofs[i]];
    boundary_values[dof] = value;
    //cout << "Setting BC value: i = " << i << ", dof = " << dof << ", value = " << value << endl;
  }

}
//-----------------------------------------------------------------------------
void DirichletBC::computeBCGeometrical(std::map<uint, real>& boundary_values,
                                       Facet& facet,
                                       BoundaryCondition::LocalData& data)
{
  // Loop the vertices associated with the facet
  for (VertexIterator vertex(facet); !vertex.end(); ++vertex)
  {
    // Loop the cells associated with the vertex
    for (CellIterator c(*vertex); !c.end(); ++c)
    {
      UFCCell ufc_cell(*c);
      
      // Interpolate function on cell
      g.interpolate(data.w, ufc_cell, *data.finite_element, *c);
      
      // Tabulate dofs on cell, and their coordinates
      data.dof_map->tabulate_dofs(data.cell_dofs, ufc_cell);
      data.dof_map->tabulate_coordinates(data.coordinates, ufc_cell);
      
      // Loop all dofs on cell
      for (uint i = 0; i < data.dof_map->local_dimension(); ++i)
      {
        // Check if the coordinates are on current facet and thus on boundary
        if (!onFacet(data.coordinates[i], facet))
          continue;
        
        // Set boundary value
        const uint dof = data.offset + data.cell_dofs[i];
        const real value = data.w[i];
        boundary_values[dof] = value;
      }
    }
  }
}
//-----------------------------------------------------------------------------
void DirichletBC::computeBCPointwise(std::map<uint, real>& boundary_values,
                                       Cell& cell,
                                       BoundaryCondition::LocalData& data)
{
  UFCCell ufc_cell(cell);

  // Interpolate function on cell
  g.interpolate(data.w, ufc_cell, *data.finite_element, cell);
      
  // Tabulate dofs on cell, and their coordinates
  data.dof_map->tabulate_dofs(data.cell_dofs, ufc_cell);
  data.dof_map->tabulate_coordinates(data.coordinates, ufc_cell);
      
  // Loop all dofs on cell
  for (uint i = 0; i < data.dof_map->local_dimension(); ++i)
  {
    // Check if the coordinates are part of the sub domain
    if ( !user_sub_domain->inside(data.coordinates[i], false) )
      continue;

    // Set boundary value
    const uint dof = data.offset + data.cell_dofs[i];
    const real value = data.w[i];
    boundary_values[dof] = value;
  }
}
//-----------------------------------------------------------------------------
bool DirichletBC::onFacet(real* coordinates, Facet& facet)
{
  // Check if the coordinates are on the same line as the line segment
  if ( facet.dim() == 1 )
  {
    // Create points
    Point p(coordinates[0], coordinates[1]);
    Point v0 = Vertex(facet.mesh(), facet.entities(0)[0]).point();
    Point v1 = Vertex(facet.mesh(), facet.entities(0)[1]).point();

    // Create vectors
    Point v01 = v1 - v0;
    Point vp0 = v0 - p;
    Point vp1 = v1 - p;

    // Check if the length of the sum of the two line segments vp0 and vp1 is
    // equal to the total length of the facet
    if ( std::abs(v01.norm() - vp0.norm() - vp1.norm()) < DOLFIN_EPS )
      return true;
    else
      return false;
  }
  // Check if the coordinates are in the same plane as the triangular facet
  else if ( facet.dim() == 2 )
  {
    // Create points
    Point p(coordinates[0], coordinates[1], coordinates[2]);
    Point v0 = Vertex(facet.mesh(), facet.entities(0)[0]).point();
    Point v1 = Vertex(facet.mesh(), facet.entities(0)[1]).point();
    Point v2 = Vertex(facet.mesh(), facet.entities(0)[2]).point();

    // Create vectors
    Point v01 = v1 - v0;
    Point v02 = v2 - v0;
    Point vp0 = v0 - p;
    Point vp1 = v1 - p;
    Point vp2 = v2 - p;

    // Check if the sum of the area of the sub triangles is equal to the total
    // area of the facet
    if ( std::abs(v01.cross(v02).norm() - vp0.cross(vp1).norm() - vp1.cross(vp2).norm()
        - vp2.cross(vp0).norm()) < DOLFIN_EPS )
      return true;
    else
      return false;
  }

  error("Unable to determine if given point is on facet (not implemented for given facet dimension).");

  return false;
}
//-----------------------------------------------------------------------------
void DirichletBC::zero(GenericMatrix& A, const Form& form)
{
  zero(A, form.dofMaps()[1], form.form());
}
//-----------------------------------------------------------------------------
void DirichletBC::zero(GenericMatrix& A, const DofMap& dof_map, const ufc::form& form)
{
  // FIXME: How do we reuse the dof map for u?

  std::string s;
  if (method == topological)
    s = "Applying Dirichlet boundary conditions to linear system";
  else if (method == geometrical)
    s = "Applying Dirichlet boundary conditions to linear system (geometrical approach)";
  else
    s = "Applying Dirichlet boundary conditions to linear system (pointwise approach)";
  
  // Make sure we have the facet - cell connectivity
  const uint D = _mesh.topology().dim();
  if (method == topological)
    _mesh.init(D - 1, D);

  // Create local data for application of boundary conditions
  BoundaryCondition::LocalData data(form, _mesh, dof_map, sub_system);
  
  // A map to hold the mapping from boundary dofs to boundary values
  std::map<uint, real> boundary_values;

  if (method == pointwise)
  {
    Progress p(s, _mesh.size(D));
    for (CellIterator cell(_mesh); !cell.end(); ++cell)
    {
      computeBCPointwise(boundary_values, *cell, data);
      p++;
    }
  }
  else
  {
    // Iterate over the facets of the mesh
    Progress p("Applying Dirichlet boundary conditions", _mesh.size(D - 1));
    for (FacetIterator facet(_mesh); !facet.end(); ++facet)
    {
      // Skip facets not inside the sub domain
      if ((*sub_domains)(*facet) != sub_domain)
      {
        p++;
        continue;
      }

      // Chose strategy
      if (method == topological)
        computeBCTopological(boundary_values, *facet, data);
      else
        computeBCGeometrical(boundary_values, *facet, data);
    
      // Update process
      p++;
    }
  }

  // Copy boundary value data to arrays
  uint* dofs = new uint[boundary_values.size()];
  std::map<uint, real>::const_iterator boundary_value;
  uint i = 0;
  for (boundary_value = boundary_values.begin(); boundary_value != boundary_values.end(); ++boundary_value)
  {
    dofs[i++]     = boundary_value->first;
  }

  // Modify linear system (A_ii = 1)
  A.zero(boundary_values.size(), dofs);

  // Finalise changes to A
  A.apply();

  // Clear temporary arrays
  delete [] dofs;
}

