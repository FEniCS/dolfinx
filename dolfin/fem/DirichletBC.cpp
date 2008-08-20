// Copyright (C) 2007-2008 Anders Logg and Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Kristian Oelgaard, 2007
// Modified by Martin Sandve Alnes, 2008
//
// First added:  2007-04-10
// Last changed: 2008-06-30

#include <dolfin/common/constants.h>
#include <dolfin/log/log.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshData.h>
#include <dolfin/mesh/MeshFunction.h>
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
    method(method), user_sub_domain(&sub_domain)
{
  initFromSubDomain(sub_domain);
}
//-----------------------------------------------------------------------------
DirichletBC::DirichletBC(Function& g,
                         MeshFunction<uint>& sub_domains,
                         uint sub_domain,
                         BCMethod method)
  : BoundaryCondition(), g(g), _mesh(sub_domains.mesh()),
    method(method), user_sub_domain(0)
{
  initFromMeshFunction(sub_domains, sub_domain);
}
//-----------------------------------------------------------------------------
DirichletBC::DirichletBC(Function& g,
                         Mesh& mesh,
                         uint sub_domain,
                         BCMethod method)
  : BoundaryCondition(), g(g), _mesh(mesh),
    method(method), user_sub_domain(0)
{
  initFromMesh(sub_domain);
}
//-----------------------------------------------------------------------------
DirichletBC::DirichletBC(Function& g,
                         Mesh& mesh,
                         SubDomain& sub_domain,
                         const SubSystem& sub_system,
                         BCMethod method)
  : BoundaryCondition(), g(g), _mesh(mesh),
    method(method), user_sub_domain(&sub_domain),
    sub_system(sub_system)
{
  initFromSubDomain(sub_domain);
}
//-----------------------------------------------------------------------------
DirichletBC::DirichletBC(Function& g,
                         MeshFunction<uint>& sub_domains,
                         uint sub_domain,
                         const SubSystem& sub_system,
                         BCMethod method)
  : BoundaryCondition(), g(g), _mesh(sub_domains.mesh()),
    method(method), user_sub_domain(0),
    sub_system(sub_system)
{
  initFromMeshFunction(sub_domains, sub_domain);
}
//-----------------------------------------------------------------------------
DirichletBC::DirichletBC(Function& g,
                         Mesh& mesh,
                         uint sub_domain,
                         const SubSystem& sub_system,
                         BCMethod method)
  : BoundaryCondition(), g(g), _mesh(mesh),
    method(method), user_sub_domain(0),
    sub_system(sub_system)
{
  initFromMesh(sub_domain);
}
//-----------------------------------------------------------------------------
DirichletBC::~DirichletBC()
{
  // Do nothing
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
  // Simple check
  const uint N = dof_map.global_dimension();
  if (N != A.size(0) /*  || N != A.size(1) alow for rectangular matrices */)
    error("Incorrect dimension of matrix for application of boundary conditions. Did you assemble it on a different mesh?");
  if (N != b.size())
    error("Incorrect dimension of matrix for application of boundary conditions. Did you assemble it on a different mesh?");

  // A map to hold the mapping from boundary dofs to boundary values
  std::map<uint, real> boundary_values;

  // Create local data for application of boundary conditions
  BoundaryCondition::LocalData data(form, _mesh, dof_map, sub_system);

  // Compute dofs and values
  computeBC(boundary_values, data);

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
    delete [] x_values;
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
void DirichletBC::zero(GenericMatrix& A, const Form& form)
{
  zero(A, form.dofMaps()[1], form.form());
}
//-----------------------------------------------------------------------------
void DirichletBC::zero(GenericMatrix& A, const DofMap& dof_map, const ufc::form& form)
{
  // Simple check
  const uint N = dof_map.global_dimension();
  if (N != A.size(0))
    error("Incorrect dimension of matrix for application of boundary conditions. Did you assemble it on a different mesh?");

  // A map to hold the mapping from boundary dofs to boundary values
  std::map<uint, real> boundary_values;

  // Create local data for application of boundary conditions
  BoundaryCondition::LocalData data(form, _mesh, dof_map, sub_system);

  // Compute dofs and values
  computeBC(boundary_values, data);

  // Copy boundary value data to arrays
  uint* dofs = new uint[boundary_values.size()];
  std::map<uint, real>::const_iterator boundary_value;
  uint i = 0;
  for (boundary_value = boundary_values.begin(); boundary_value != boundary_values.end(); ++boundary_value)
    dofs[i++] = boundary_value->first;

  // Modify linear system (A_ii = 1)
  A.zero(boundary_values.size(), dofs);

  // Finalise changes to A
  A.apply();

  // Clear temporary arrays
  delete [] dofs;
}
//-----------------------------------------------------------------------------
Mesh& DirichletBC::mesh()
{
  return _mesh;
}
//-----------------------------------------------------------------------------
void DirichletBC::initFromSubDomain(SubDomain& sub_domain)
{
  dolfin_assert(facets.size() == 0);

  // FIXME: This can be made more efficient, we should be able to
  // FIXME: extract the facets without first creating a MeshFunction on
  // FIXME: the entire mesh and then extracting the subset. This is done
  // FIXME: mainly for convenience (we may reuse mark() in SubDomain).

  // Make sure the mesh has been ordered
  _mesh.order();

  // Create mesh function for sub domain markers on facets
  const uint dim = _mesh.topology().dim();
  _mesh.init(dim - 1);
  MeshFunction<uint> sub_domains(_mesh, dim - 1);

  // Mark everything as sub domain 1
  sub_domains = 1;
  
  // Mark the sub domain as sub domain 0
  sub_domain.mark(sub_domains, 0);

  // Initialize from mesh function
  initFromMeshFunction(sub_domains, 0);
}
//-----------------------------------------------------------------------------
void DirichletBC::initFromMeshFunction(MeshFunction<uint>& sub_domains,
                                       uint sub_domain)
{
  dolfin_assert(facets.size() == 0);

  // Make sure we have the facet - cell connectivity
  const uint dim = _mesh.topology().dim();
  _mesh.init(dim - 1, dim);

  // Make sure the mesh has been ordered
  _mesh.order();

  // Build set of boundary facets
  for (FacetIterator facet(_mesh); !facet.end(); ++facet)
  {
    // Skip facets not on this boundary
    if (sub_domains(*facet) != sub_domain)
      continue;

    // Get cell to which facet belongs (there may be two, but pick first)
    Cell cell(_mesh, facet->entities(dim)[0]);
    
    // Get local index of facet with respect to the cell
    const uint facet_number = cell.index(*facet);

    // Copy data
    facets.push_back(std::pair<uint, uint>(cell.index(), facet_number));
  }
}
//-----------------------------------------------------------------------------
void DirichletBC::initFromMesh(uint sub_domain)
{
  dolfin_assert(facets.size() == 0);

  // For this to work, the mesh *needs* to be ordered according to
  // the UFC ordering before it gets here. So reordering the mesh
  // here will either have no effect (if the mesh is already ordered
  // or it won't do anything good (since the markers are wrong anyway).
  // In conclusion: we don't need to order the mesh here.
  
  cout << "Creating sub domain markers for boundary condition." << endl;

  // Get data
  Array<uint>* facet_cells   = _mesh.data().array("boundary facet cells");
  Array<uint>* facet_numbers = _mesh.data().array("boundary facet numbers");
  Array<uint>* indicators    = _mesh.data().array("boundary indicators");

  // Check data
  if (!facet_cells)
    error("Mesh data \"boundary facet cells\" not available.");
  if (!facet_numbers)
    error("Mesh data \"boundary facet numbers\" not available.");
  if (!indicators)
    error("Mesh data \"boundary indicators\" not available.");

  // Get size
  const uint size = facet_cells->size();
  dolfin_assert(size == facet_numbers->size());
  dolfin_assert(size == indicators->size());

  // Build set of boundary facets
  for (uint i = 0; i < size; i++)
  {
    // Skip facets not on this boundary
    if ((*indicators)[i] != sub_domain)
      continue;

    // Copy data
    facets.push_back(std::pair<uint, uint>((*facet_cells)[i], (*facet_numbers)[i]));
  }
}
//-----------------------------------------------------------------------------
void DirichletBC::computeBC(std::map<uint, real>& boundary_values,
                            BoundaryCondition::LocalData& data)
{
  // Choose strategy
  switch (method)
  {
  case topological:
    computeBCTopological(boundary_values, data);
    break;
  case geometric:
    computeBCGeometric(boundary_values, data);
    break;
  case pointwise:
    computeBCPointwise(boundary_values, data);
    break;
  default:
    error("Unknown method for application of boundary conditions.");
  }
}
//-----------------------------------------------------------------------------
void DirichletBC::computeBCTopological(std::map<uint, real>& boundary_values,
                                       BoundaryCondition::LocalData& data)
{
  // Special case
  if (facets.size() == 0)
  {
    warning("Found no facets matching domain for boundary condition.");
    return;
  }

  // Iterate over facets
  Progress p("Computing Dirichlet boundary values, topological search", facets.size());
  for (uint f = 0; f < facets.size(); f++)
  {
    // Get cell number and local facet number
    uint cell_number = facets[f].first;
    uint facet_number = facets[f].second;

    // Create cell
    Cell cell(_mesh, cell_number);
    UFCCell ufc_cell(cell);

    // Interpolate function on cell
    g.interpolate(data.w, ufc_cell, *data.finite_element, cell, facet_number);
    
    // Tabulate dofs on cell
    data.dof_map->tabulate_dofs(data.cell_dofs, ufc_cell);
    
    // Tabulate which dofs are on the facet
    data.dof_map->tabulate_facet_dofs(data.facet_dofs, facet_number);
    
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

    p++;
  }
}
//-----------------------------------------------------------------------------
void DirichletBC::computeBCGeometric(std::map<uint, real>& boundary_values,
                                     BoundaryCondition::LocalData& data)
{
  // Special case
  if (facets.size() == 0)
  {
    warning("Found no facets matching domain for boundary condition.");
    return;
  }

  // Initialize facets, needed for geometric search
  message("Computing facets, needed for geometric application of boundary conditions.");
  _mesh.init(_mesh.topology().dim() - 1);

  // Iterate over facets
  Progress p("Computing Dirichlet boundary values, geometric search", facets.size());
  for (uint f = 0; f < facets.size(); f++)
  {
    // Get cell number and local facet number
    uint cell_number = facets[f].first;
    uint facet_number = facets[f].second;

    // Create facet
    Cell cell(_mesh, cell_number);
    Facet facet(_mesh, cell.entities(_mesh.topology().dim() - 1)[facet_number]);

    // Loop the vertices associated with the facet
    for (VertexIterator vertex(facet); !vertex.end(); ++vertex)
    {
      // Loop the cells associated with the vertex
      for (CellIterator c(*vertex); !c.end(); ++c)
      {
        UFCCell ufc_cell(*c);
        
        bool interpolated = false;
        
        // Tabulate coordinates of dofs on cell
        data.dof_map->tabulate_coordinates(data.coordinates, ufc_cell);
        
        // Loop over all dofs on cell
        for (uint i = 0; i < data.dof_map->local_dimension(); ++i)
        {
          // Check if the coordinates are on current facet and thus on boundary
          if (!onFacet(data.coordinates[i], facet))
            continue;
          
          if(!interpolated)
          {
            // Tabulate dofs on cell
            data.dof_map->tabulate_dofs(data.cell_dofs, ufc_cell);
            // Interpolate function on cell
            g.interpolate(data.w, ufc_cell, *data.finite_element, *c);
          }
          
          // Set boundary value
          const uint dof = data.offset + data.cell_dofs[i];
          const real value = data.w[i];
          boundary_values[dof] = value;
        }
      }
    }
  }
}
//-----------------------------------------------------------------------------
void DirichletBC::computeBCPointwise(std::map<uint, real>& boundary_values,
                                     BoundaryCondition::LocalData& data)
{
  dolfin_assert(user_sub_domain);

  // Iterate over cells
  Progress p("Computing Dirichlet boundary values, pointwise search", _mesh.numCells());
  for (CellIterator cell(_mesh); !cell.end(); ++cell)
  {
    UFCCell ufc_cell(*cell);
    
    // Tabulate coordinates of dofs on cell
    data.dof_map->tabulate_coordinates(data.coordinates, ufc_cell);
    
    // Interpolate function only once and only on cells where necessary
    bool interpolated = false;
    
    // Loop all dofs on cell
    for (uint i = 0; i < data.dof_map->local_dimension(); ++i)
    {
      // Check if the coordinates are part of the sub domain
      if ( !user_sub_domain->inside(data.coordinates[i], false) )
        continue;
      
      if(!interpolated)
      {
        interpolated = true;
        // Tabulate dofs on cell
        data.dof_map->tabulate_dofs(data.cell_dofs, ufc_cell);
        // Interpolate function on cell
        g.interpolate(data.w, ufc_cell, *data.finite_element, *cell);
      }
      
      // Set boundary value
      const uint dof = data.offset + data.cell_dofs[i];
      const real value = data.w[i];
      boundary_values[dof] = value;
    }

    p++;
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
void DirichletBC::setSubSystem(SubSystem sub_system)
{
  this->sub_system = sub_system;
}
//-----------------------------------------------------------------------------
void DirichletBC::getBC(uint n, uint* indicators, real* values, 
                        const DofMap& dof_map, const ufc::form& form)
{
  // A map to hold the mapping from boundary dofs to boundary values
  std::map<uint, real> boundary_values;

  // Create local data for application of boundary conditions
  BoundaryCondition::LocalData data(form, _mesh, dof_map, sub_system);

  // Compute dofs and values
  computeBC(boundary_values, data);

  if ( n != dof_map.global_dimension() )
    error("The n should be the same as dof_map.global_dimension()");  

  std::map<uint, real>::const_iterator boundary_value;
  uint i = 0;
  for (boundary_value = boundary_values.begin(); boundary_value != boundary_values.end(); ++boundary_value)
  {
    i = boundary_value->first; 
    indicators[i] = 1;  
    values[i]     = boundary_value->second;
  }
}
//-----------------------------------------------------------------------------

