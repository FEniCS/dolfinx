// Copyright (C) 2007-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2007,-2008
// Modified by Ola Skavhaug, 2007-2008
// Modified by Kent-Andre Mardal, 2008
//
// First added:  2007-01-17
// Last changed: 2008-08-29

#include <ufc.h>
#include <dolfin/main/MPI.h>
#include <dolfin/log/dolfin_log.h>
#include <dolfin/common/Array.h>
#include <dolfin/common/Timer.h>
#include <dolfin/la/GenericMatrix.h>
#include <dolfin/la/GenericTensor.h>
#include <dolfin/la/GenericVector.h>
#include <dolfin/la/Scalar.h>
#include <dolfin/la/SparsityPattern.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Facet.h>
#include <dolfin/mesh/MeshData.h>
#include <dolfin/mesh/BoundaryMesh.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/mesh/SubDomain.h>
#include <dolfin/function/Function.h>
#include "Form.h"
#include "UFC.h"
#include "Assembler.h"
#include "SparsityPatternBuilder.h"
#include "DofMapSet.h"
#include "DirichletBC.h"
#include "FiniteElement.h"

#include <dolfin/common/timing.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Assembler::Assembler(Mesh& mesh) : mesh(mesh), parallel(MPI::num_processes()>0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Assembler::~Assembler()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void Assembler::assemble(GenericTensor& A, Form& form, bool reset_tensor)
{
  form.updateDofMaps(mesh);
  assemble(A, form.form(), form.coefficients(), form.dofMaps(), 0, 0, 0, reset_tensor);
}
//-----------------------------------------------------------------------------
void Assembler::assemble(GenericMatrix& A, Form& a, GenericVector& b, Form& L, 
                         DirichletBC& bc, bool reset_tensor)
{
  Array<DirichletBC*> bcs;
  bcs.push_back(&bc);
  assemble(A, a, b, L, bcs, reset_tensor); 
}
//-----------------------------------------------------------------------------
void Assembler::assemble(GenericMatrix& A, Form& a, GenericVector& b, Form& L, 
                         Array<DirichletBC*>& bcs, bool reset_tensor)
{
  a.updateDofMaps(mesh);
  L.updateDofMaps(mesh);
  assemble_system(A, a.form(), a.coefficients(), a.dofMaps(), 
                  b, L.form(), L.coefficients(), L.dofMaps(),
                  0, bcs, 0, 0 , 0, reset_tensor); 
}
//-----------------------------------------------------------------------------
void Assembler::assemble(GenericTensor& A, Form& form, const SubDomain& sub_domain, bool reset_tensor)
{
  // Extract cell domains
  MeshFunction<uint>* cell_domains = 0;
  if (form.form().num_cell_integrals() > 0)
  {
    cell_domains = new MeshFunction<uint>(mesh, mesh.topology().dim());
    (*cell_domains) = 1;
    sub_domain.mark(*cell_domains, 0);
  }

  // Extract facet domains
  MeshFunction<uint>* facet_domains = 0;
  if (form.form().num_exterior_facet_integrals() > 0 ||
      form.form().num_interior_facet_integrals() > 0)
  {
    facet_domains = new MeshFunction<uint>(mesh, mesh.topology().dim() - 1);
    (*facet_domains) = 1;
    sub_domain.mark(*facet_domains, 0);
  }

  // Assemble
  form.updateDofMaps(mesh);
  assemble(A, form.form(), form.coefficients(), form.dofMaps(),
           cell_domains, facet_domains, facet_domains, reset_tensor);

  // Delete domains
  if (cell_domains)
    delete cell_domains;
  if (facet_domains)
    delete facet_domains;
}
//-----------------------------------------------------------------------------
void Assembler::assemble(GenericTensor& A, Form& form,
                         const MeshFunction<uint>& cell_domains,
                         const MeshFunction<uint>& exterior_facet_domains,
                         const MeshFunction<uint>& interior_facet_domains,
                         bool reset_tensor)
{
  form.updateDofMaps(mesh);
  assemble(A, form.form(), form.coefficients(), form.dofMaps(), &cell_domains, 
           &exterior_facet_domains, &interior_facet_domains, reset_tensor);
}
//-----------------------------------------------------------------------------
dolfin::real Assembler::assemble(Form& form,
                                 bool reset_tensor)
{
  Scalar value;
  assemble(value, form, reset_tensor);
  return value;
}
//-----------------------------------------------------------------------------
dolfin::real Assembler::assemble(Form& form, const SubDomain& sub_domain,
                                 bool reset_tensor)
{
  Scalar value;
  assemble(value, form, sub_domain, reset_tensor);
  return value;
}
//-----------------------------------------------------------------------------
dolfin::real Assembler::assemble(Form& form,
                                 const MeshFunction<uint>& cell_domains,
                                 const MeshFunction<uint>& exterior_facet_domains,
                                 const MeshFunction<uint>& interior_facet_domains,
                                 bool reset_tensor)
{
  Scalar value;
  assemble(value, form,
           cell_domains, exterior_facet_domains, interior_facet_domains,
           reset_tensor);
  return value;
}
//-----------------------------------------------------------------------------
void Assembler::assemble(GenericTensor& A, const ufc::form& form,
                         const Array<Function*>& coefficients,
                         const DofMapSet& dof_map_set,
                         const MeshFunction<uint>* cell_domains,
                         const MeshFunction<uint>* exterior_facet_domains,
                         const MeshFunction<uint>* interior_facet_domains,
                         bool reset_tensor)
{
  // Note the importance of treating empty mesh functions as null pointers
  // for the PyDOLFIN interface.
  
  // Check arguments
  check(form, coefficients, mesh);

  // Create data structure for local assembly data
  UFC ufc(form, mesh, dof_map_set);

  // Initialize global tensor
  initGlobalTensor(A, dof_map_set, ufc, reset_tensor);

  // Assemble over cells
  assembleCells(A, coefficients, dof_map_set, ufc, cell_domains, 0);

  // Assemble over exterior facets 
  assembleExteriorFacets(A, coefficients, dof_map_set, ufc, exterior_facet_domains, 0);

  // Assemble over interior facets
  assembleInteriorFacets(A, coefficients, dof_map_set, ufc, interior_facet_domains, 0);

  // Finalise assembly of global tensor
  A.apply();
}
//-----------------------------------------------------------------------------
void Assembler::assembleCells(GenericTensor& A,
                              const Array<Function*>& coefficients,
                              const DofMapSet& dof_map_set,
                              UFC& ufc,
                              const MeshFunction<uint>* domains,
                              Array<real>* values) const
{
  // Skip assembly if there are no cell integrals
  if (ufc.form.num_cell_integrals() == 0)
    return;
  Timer timer("Assemble cells");

  // Cell integral
  ufc::cell_integral* integral = ufc.cell_integrals[0];

  // Assemble over cells
  Progress p(progressMessage(A.rank(), "cells"), mesh.numCells());
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    // Get integral for sub domain (if any)
    if (domains && domains->size() > 0)
    {
      const uint domain = (*domains)(*cell);
      if (domain < ufc.form.num_cell_integrals())
        integral = ufc.cell_integrals[domain];
      else
        continue;
    }

    // Update to current cell
    ufc.update(*cell);

    // Interpolate coefficients on cell
    for (uint i = 0; i < coefficients.size(); i++)
      coefficients[i]->interpolate(ufc.w[i], ufc.cell, *ufc.coefficient_elements[i], *cell);
    
    // Tabulate dofs for each dimension
    for (uint i = 0; i < ufc.form.rank(); i++)
      dof_map_set[i].tabulate_dofs(ufc.dofs[i], ufc.cell, cell->index());

    // Tabulate cell tensor
    integral->tabulate_tensor(ufc.A, ufc.w, ufc.cell);

    // Add entries to global tensor
    if (values && ufc.form.rank() == 0)
      (*values)[cell->index()] = ufc.A[0];
    else
      A.add(ufc.A, ufc.local_dimensions, ufc.dofs);
    
    p++;
  }
}
//-----------------------------------------------------------------------------
void Assembler::assembleExteriorFacets(GenericTensor& A,
                                       const Array<Function*>& coefficients,
                                       const DofMapSet& dof_map_set,
                                       UFC& ufc,
                                       const MeshFunction<uint>* domains,
                                       Array<real>* values) const
{
  // Skip assembly if there are no exterior facet integrals
  if (ufc.form.num_exterior_facet_integrals() == 0)
    return;
  Timer timer("Assemble exterior facets");
  
  // Exterior facet integral
  ufc::exterior_facet_integral* integral = ufc.exterior_facet_integrals[0];

  // Create boundary mesh
  BoundaryMesh boundary(mesh);
  MeshFunction<uint>* cell_map = boundary.data().meshFunction("cell map");
  dolfin_assert(cell_map);

  // Assemble over exterior facets (the cells of the boundary)
  Progress p(progressMessage(A.rank(), "exterior facets"), boundary.numCells());
  for (CellIterator boundary_cell(boundary); !boundary_cell.end(); ++boundary_cell)
  {
    // Get mesh facet corresponding to boundary cell
    Facet mesh_facet(mesh, (*cell_map)(*boundary_cell));

    // Get integral for sub domain (if any)
    if (domains && domains->size() > 0)
    {
      const uint domain = (*domains)(mesh_facet);
      if (domain < ufc.form.num_exterior_facet_integrals())
        integral = ufc.exterior_facet_integrals[domain];
      else
        continue;
    }

    // Get mesh cell to which mesh facet belongs (pick first, there is only one)
    dolfin_assert(mesh_facet.numEntities(mesh.topology().dim()) == 1);
    Cell mesh_cell(mesh, mesh_facet.entities(mesh.topology().dim())[0]);

    // Get local index of facet with respect to the cell
    const uint local_facet = mesh_cell.index(mesh_facet);
      
    // Update to current cell
    ufc.update(mesh_cell);

    // Interpolate coefficients on cell
    for (uint i = 0; i < coefficients.size(); i++)
      coefficients[i]->interpolate(ufc.w[i], ufc.cell, *ufc.coefficient_elements[i], mesh_cell, local_facet);

    // Tabulate dofs for each dimension
    for (uint i = 0; i < ufc.form.rank(); i++)
      dof_map_set[i].tabulate_dofs(ufc.dofs[i], ufc.cell, mesh_cell.index());

    // Tabulate exterior facet tensor
    integral->tabulate_tensor(ufc.A, ufc.w, ufc.cell, local_facet);
    
    // Add entries to global tensor
    A.add(ufc.A, ufc.local_dimensions, ufc.dofs);

    p++;  
  }
}
//-----------------------------------------------------------------------------
void Assembler::assembleInteriorFacets(GenericTensor& A,
                                       const Array<Function*>& coefficients,
                                       const DofMapSet& dof_map_set,
                                       UFC& ufc,
                                       const MeshFunction<uint>* domains,
                                       Array<real>* values) const
{
  // Skip assembly if there are no interior facet integrals
  if (ufc.form.num_interior_facet_integrals() == 0)
    return;
  Timer timer("Assemble interior facets");
  
  // Interior facet integral
  ufc::interior_facet_integral* integral = ufc.interior_facet_integrals[0];

  // Compute facets and facet - cell connectivity if not already computed
  mesh.init(mesh.topology().dim() - 1);
  mesh.init(mesh.topology().dim() - 1, mesh.topology().dim());
  mesh.order();
  
  // Assemble over interior facets (the facets of the mesh)
  Progress p(progressMessage(A.rank(), "interior facets"), mesh.numFacets());
  for (FacetIterator facet(mesh); !facet.end(); ++facet)
  {
    // Check if we have an interior facet
    if ( facet->numEntities(mesh.topology().dim()) != 2 )
    {
      p++;
      continue;
    }

    // Get integral for sub domain (if any)
    if (domains && domains->size() > 0)
    {
      const uint domain = (*domains)(*facet);
      if (domain < ufc.form.num_interior_facet_integrals())
        integral = ufc.interior_facet_integrals[domain];
      else
        continue;
    }

    // Get cells incident with facet
    Cell cell0(mesh, facet->entities(mesh.topology().dim())[0]);
    Cell cell1(mesh, facet->entities(mesh.topology().dim())[1]);
      
    // Get local index of facet with respect to each cell
    uint facet0 = cell0.index(*facet);
    uint facet1 = cell1.index(*facet);

    // Update to current pair of cells
    ufc.update(cell0, cell1);
    
    // Interpolate coefficients on cell
    for (uint i = 0; i < coefficients.size(); i++)
    {
      const uint offset = ufc.coefficient_elements[i]->spaceDimension();
      coefficients[i]->interpolate(ufc.macro_w[i], ufc.cell0, *ufc.coefficient_elements[i], cell0, facet0);
      coefficients[i]->interpolate(ufc.macro_w[i] + offset, ufc.cell1, *ufc.coefficient_elements[i], cell1, facet1);
    }

    // Tabulate dofs for each dimension on macro element
    for (uint i = 0; i < ufc.form.rank(); i++)
    {
      const uint offset = ufc.local_dimensions[i];
      dof_map_set[i].tabulate_dofs(ufc.macro_dofs[i],          ufc.cell0, cell0.index());
      dof_map_set[i].tabulate_dofs(ufc.macro_dofs[i] + offset, ufc.cell1, cell1.index());
    }

    // Tabulate exterior interior facet tensor on macro element
    integral->tabulate_tensor(ufc.macro_A, ufc.macro_w, ufc.cell0, ufc.cell1, facet0, facet1);

    // Add entries to global tensor
    A.add(ufc.macro_A, ufc.macro_local_dimensions, ufc.macro_dofs);

    p++;
  }
}
//-----------------------------------------------------------------------------
void Assembler::check(const ufc::form& form, 
                      const Array<Function*>& coefficients, 
                      const Mesh& mesh) const
{
  // Check that we get the correct number of coefficients
  if (coefficients.size() != form.num_coefficients())
    error("Incorrect number of coefficients for form: %d given but %d required.",
          coefficients.size(), form.num_coefficients());
  
  // Check that all coefficients have valid value dimensions
  for(uint i=0; i<coefficients.size(); ++i)
  {
    if(!coefficients[i])
      error("Got NULL Function as coefficient %d.", i);
    
    try
    {
      // auto_ptr deletes its object when it exits its scope
      std::auto_ptr<ufc::finite_element> fe( form.create_finite_element(i+form.rank()) );
      
      uint r = coefficients[i]->rank();
      uint fe_r = fe->value_rank();
      if(fe_r != r)
        warning("Invalid value rank of Function %d, got %d but expecting %d. \
You may need to provide the rank of a user defined Function.", i, r, fe_r);
      
      for(uint j=0; j<r; ++j)
      {
        uint dim = coefficients[i]->dim(j);
        uint fe_dim = fe->value_dimension(j);
        if(dim != fe_dim)
          warning("Invalid value dimension %d of Function %d, got %d but expecting %d. \
You may need to provide the dimension of a user defined Function.", j, i, dim, fe_dim);
      }
    }
    catch(std::exception & e)
    {
      warning("Function %d is invalid.", i);
    }
  }
  
  // Check that the cell dimension matches the mesh dimension
  if (form.rank() + form.num_coefficients() > 0)
  {
    ufc::finite_element* element = form.create_finite_element(0);
    dolfin_assert(element);
    if (mesh.type().cellType() == CellType::interval && element->cell_shape() != ufc::interval)
      error("Mesh cell type (intervals) does not match cell type of form.");
    if (mesh.type().cellType() == CellType::triangle && element->cell_shape() != ufc::triangle)
      error("Mesh cell type (triangles) does not match cell type of form.");
    if (mesh.type().cellType() == CellType::tetrahedron && element->cell_shape() != ufc::tetrahedron)
      error("Mesh cell type (tetrahedra) does not match cell type of form.");
    delete element;
  }
}
//-----------------------------------------------------------------------------
void Assembler::initGlobalTensor(GenericTensor& A, const DofMapSet& dof_map_set, 
                                 UFC& ufc, bool reset_tensor) const
{
  if (reset_tensor)
  {
    //dof_map_set.build(ufc);

    // Build sparsity pattern
    Timer t0("Build sparsity");
    GenericSparsityPattern* sparsity_pattern = A.factory().createPattern();
    if (sparsity_pattern)
      SparsityPatternBuilder::build(*sparsity_pattern, mesh, ufc, dof_map_set);
    t0.stop();
    
    // Initialize tensor
    Timer t1("Init tensor");
    if (sparsity_pattern)
      A.init(*sparsity_pattern);
    else
      A.init(ufc.form.rank(), ufc.global_dimensions);
    t1.stop();

    // Delete sparsity pattern
    Timer t2("Delete sparsity");
    delete sparsity_pattern;
    t2.stop();
  }
  else
    A.zero();
}
//-----------------------------------------------------------------------------
std::string Assembler::progressMessage(uint rank, std::string integral_type) const
{
  std::stringstream s;
  s << "Assembling ";
  
  switch (rank)
  {
  case 0:
    s << "scalar value over ";
    break;
  case 1:
    s << "vector over ";
    break;
  case 2:
    s << "matrix over ";
    break;
  default:
    s << "rank " << rank << " tensor over ";
    break;
  }
 
  s << integral_type;

  return s.str();
}
//-----------------------------------------------------------------------------
void Assembler::assemble_system(GenericMatrix& A, const ufc::form& A_form, 
                                const Array<Function*>& A_coefficients, 
                                const DofMapSet& A_dof_map_set,
                                GenericVector& b, const ufc::form& b_form, 
                                const Array<Function*>& b_coefficients, 
                                const DofMapSet& b_dof_map_set, 
                                const GenericVector* x0,
                                Array<DirichletBC*> bcs, 
                                const MeshFunction<uint>* cell_domains,
                                const MeshFunction<uint>* exterior_facet_domains,
                                const MeshFunction<uint>* interior_facet_domains,
                                bool reset_tensors)
{
   Timer timer("Assemble system");

  // Note the importance of treating empty mesh functions as null pointers
  // for the PyDOLFIN interface.

  // Check arguments
  check(A_form, A_coefficients, mesh);
  check(b_form, b_coefficients, mesh);

  // FIXME: consistency check between A_dof_map_set and b_dof_map_set 

  // Create data structure for local assembly data
  UFC A_ufc(A_form, mesh, A_dof_map_set);
  UFC b_ufc(b_form, mesh, b_dof_map_set);

  // Initialize global tensor
  initGlobalTensor(A, A_dof_map_set, A_ufc, reset_tensors);
  initGlobalTensor(b, b_dof_map_set, b_ufc, reset_tensors);

  // Pointers to element matrix and vector
  real* Ae = 0; 
  real* be = 0; 

  // Get boundary values (global) 
  const uint N = A_dof_map_set[1].global_dimension();  
  uint* indicators = new uint[N];
  real* g  = new real[N];
  for (uint i = 0; i < N; i++) 
  {
    indicators[i] = 0; 
    g[i]          = 0.0; 
  }
  for(uint i = 0; i < bcs.size(); ++i)
    bcs[i]->getBC(N, indicators, g, A_dof_map_set[1], A_form); 

  // Modify boundary values for incremental (typically nonlinear) problems
  if (x0)
  {
    warning("Symmetric application of Dirichlet boundary conditions for incremental problems is untested.");
    dolfin_assert( x0->size() == N);
    real* x0_values = new real[N];
    x0->get(x0_values);
    for (uint i = 0; i < N; i++)
      g[i] -= x0_values[i];
    delete [] x0_values;
  }

  // If there are no interior facet integrals
  if (A_ufc.form.num_interior_facet_integrals() == 0 && b_ufc.form.num_interior_facet_integrals() == 0) 
  {
    // Allocate memory for Ae and be 
    uint A_num_entries = 1;
    for (uint i = 0; i < A_form.rank(); i++)
      A_num_entries *= A_dof_map_set[i].local_dimension();
    Ae = new real[A_num_entries];

    uint b_num_entries = 1;
    for (uint i = 0; i < b_form.rank(); i++)
      b_num_entries *= b_dof_map_set[i].local_dimension();
    be = new real[b_num_entries];

    for (CellIterator cell(mesh); !cell.end(); ++cell) 
    {
      for (uint i = 0; i < A_num_entries; i++) 
        Ae[i] = 0.0; 
      for (uint i = 0; i < b_num_entries; i++) 
        be[i] = 0.0; 

      // Update to current cell
      A_ufc.update(*cell);

      // Interpolate coefficients on cell
      for (uint i = 0; i < A_coefficients.size(); i++)
        A_coefficients[i]->interpolate(A_ufc.w[i], A_ufc.cell, *A_ufc.coefficient_elements[i], *cell);

      // Tabulate dofs for each dimension
      for (uint i = 0; i < A_ufc.form.rank(); i++)
        A_dof_map_set[i].tabulate_dofs(A_ufc.dofs[i], A_ufc.cell, cell->index());

      // Update to current cell
      b_ufc.update(*cell);

      // Interpolate coefficients on cell
      for (uint i = 0; i < b_coefficients.size(); i++)
        b_coefficients[i]->interpolate(b_ufc.w[i], b_ufc.cell, *b_ufc.coefficient_elements[i], *cell);

      // Tabulate dofs for each dimension
      for (uint i = 0; i < b_ufc.form.rank(); i++)
        b_dof_map_set[i].tabulate_dofs(b_ufc.dofs[i], b_ufc.cell, cell->index());

      // Compute cell integral for A 
      if (A_ufc.form.num_cell_integrals() > 0) 
      {
        ufc::cell_integral* A_cell_integral =  A_ufc.cell_integrals[0];
        if (cell_domains && cell_domains->size() > 0)
        {
          const uint cell_domain = (*cell_domains)(*cell);
          if (cell_domain < A_ufc.form.num_cell_integrals()) 
            A_cell_integral = A_ufc.cell_integrals[cell_domain];
          else 
            continue;
        }
        // Tabulate cell tensor
        A_cell_integral->tabulate_tensor(A_ufc.A, A_ufc.w, A_ufc.cell);
        for (uint i=0; i<A_num_entries; i++) 
          Ae[i] += A_ufc.A[i]; 
      }

      // Compute cell integral for b 
      if (b_ufc.form.num_cell_integrals() > 0) 
      {
        ufc::cell_integral* b_cell_integral = b_ufc.cell_integrals[0];
        if (cell_domains && cell_domains->size() > 0) 
        {
          const uint cell_domain = (*cell_domains)(*cell);
          if (cell_domain < b_ufc.form.num_cell_integrals()) 
            b_cell_integral = b_ufc.cell_integrals[cell_domain];
          else 
            continue;
        }
        // Tabulate cell tensor
        b_cell_integral->tabulate_tensor(b_ufc.A, b_ufc.w, b_ufc.cell);
        for (uint i=0; i<b_num_entries; i++) 
          be[i] += b_ufc.A[i]; 
      }

      // Compute exterior facet integral
      if (A_ufc.form.num_exterior_facet_integrals() > 0 || b_ufc.form.num_exterior_facet_integrals() > 0) 
      {
        const uint D = mesh.topology().dim(); 

        if (A_ufc.form.num_exterior_facet_integrals() > 0) 
        {
          for (FacetIterator facet(*cell); !facet.end(); ++facet)
          {
            if (facet->numEntities(D) != 2) 
            {
              ufc::exterior_facet_integral* A_integral = A_ufc.exterior_facet_integrals[0]; 
              if (exterior_facet_domains && exterior_facet_domains->size() > 0)
              {
                const uint exterior_facet_domain= (*exterior_facet_domains)(*facet);
                if (exterior_facet_domain < A_ufc.form.num_exterior_facet_integrals())
                  A_integral = A_ufc.exterior_facet_integrals[exterior_facet_domain];
                else
                  continue;
              }
              const uint local_facet = cell->index(*facet);
              A_integral->tabulate_tensor(A_ufc.A, A_ufc.w, A_ufc.cell, local_facet);
              for (uint i=0; i<A_num_entries; i++) 
                Ae[i] += A_ufc.A[i]; 
            }
          }
        }

        if (b_ufc.form.num_exterior_facet_integrals() > 0) 
        {
          for (FacetIterator facet(*cell); !facet.end(); ++facet)
          {
            if (facet->numEntities(D) != 2) 
            {
              ufc::exterior_facet_integral* b_integral = b_ufc.exterior_facet_integrals[0]; 
              if (exterior_facet_domains && exterior_facet_domains->size() > 0)
              {
                const uint exterior_facet_domain= (*exterior_facet_domains)(*facet);
                if (exterior_facet_domain < b_ufc.form.num_exterior_facet_integrals())
                  b_integral = b_ufc.exterior_facet_integrals[exterior_facet_domain];
                else
                  continue;
              }
              const uint local_facet = cell->index(*facet);
              b_integral->tabulate_tensor(b_ufc.A, b_ufc.w, b_ufc.cell, local_facet);
              for (uint i=0; i<b_num_entries; i++) 
                be[i] += b_ufc.A[i]; 
            }
          }
        }
      }

      // Enforce Dirichlet boundary conditions
      uint m = A_ufc.local_dimensions[0]; 
      uint n = A_ufc.local_dimensions[1]; 
      for (uint i=0; i<n; i++) 
      {  
        uint ii = A_ufc.dofs[1][i]; 
        if (indicators[ii] > 0) 
        {  
          be[i] = g[ii]; 
          for (uint k=0; k<n; k++) 
            Ae[k+i*n] = 0.0; 
          for (uint j=0; j<m; j++) 
          {
            be[j] -= Ae[i+j*n]*g[ii]; 
            Ae[i+j*n] = 0.0; 
          }
          Ae[i+i*n] = 1.0; 
        }
      }

      // Add entries to global tensor
      A.add(Ae, A_ufc.local_dimensions, A_ufc.dofs);
      b.add(be, b_ufc.local_dimensions, b_ufc.dofs);
    }
  }


  if (A_ufc.form.num_interior_facet_integrals() > 0 || b_ufc.form.num_interior_facet_integrals() > 0) 
  {
    // Create data structure for local assembly data
    UFC A_macro_ufc(A_form, mesh, A_dof_map_set);
    UFC b_macro_ufc(b_form, mesh, b_dof_map_set);

    // ---create some temporal storage for Ae, Ae_macro 
    uint A_num_entries = 1;
    for (uint i = 0; i < A_form.rank(); i++)
      A_num_entries *= A_dof_map_set[i].local_dimension();
    uint A_macro_num_entries = A_num_entries*4; 
    Ae = new real[A_num_entries];
    real* Ae_macro = new real[A_macro_num_entries]; 

    // ---create some temporal storage for be, be_macro 
    uint b_num_entries = 1;
    for (uint i = 0; i < b_form.rank(); i++)
      b_num_entries *= b_dof_map_set[i].local_dimension();
    uint b_macro_num_entries = b_num_entries*2; 
    be = new real[b_num_entries];
    real* be_macro = new real[b_macro_num_entries]; 

    for (FacetIterator facet(mesh); !facet.end(); ++facet)
    {
      // Check if we have an interior facet
      if ( facet->numEntities(mesh.topology().dim()) == 2 ) 
      {
        for (uint i=0; i<A_macro_num_entries; i++) Ae_macro[i] = 0.0; 
        for (uint i=0; i<b_macro_num_entries; i++) be_macro[i] = 0.0; 

        // Get cells incident with facet
        Cell cell0(mesh, facet->entities(mesh.topology().dim())[0]);
        Cell cell1(mesh, facet->entities(mesh.topology().dim())[1]);

        // Update to current pair of cells
        A_macro_ufc.update(cell0, cell1);
        b_macro_ufc.update(cell0, cell1);

        // Get local index of facet with respect to each cell
        uint facet0 = cell0.index(*facet);
        uint facet1 = cell1.index(*facet);

        // Tabulate dofs for each dimension on macro element
        for (uint i = 0; i < A_macro_ufc.form.rank(); i++)
        {
          const uint offset = A_macro_ufc.local_dimensions[i];
          A_dof_map_set[i].tabulate_dofs(A_macro_ufc.macro_dofs[i],          A_macro_ufc.cell0, cell0.index());
          A_dof_map_set[i].tabulate_dofs(A_macro_ufc.macro_dofs[i] + offset, A_macro_ufc.cell1, cell1.index());
        }

        // Tabulate dofs for each dimension on macro element
        for (uint i = 0; i < b_macro_ufc.form.rank(); i++)
        {
          const uint offset = b_macro_ufc.local_dimensions[i];
          b_dof_map_set[i].tabulate_dofs(b_macro_ufc.macro_dofs[i],          b_macro_ufc.cell0, cell0.index());
          b_dof_map_set[i].tabulate_dofs(b_macro_ufc.macro_dofs[i] + offset, b_macro_ufc.cell1, cell1.index());
        }

        if ( A_ufc.form.num_interior_facet_integrals() ) 
        {  
          ufc::interior_facet_integral* interior_facet_integral = A_macro_ufc.interior_facet_integrals[0];

          // Get integral for sub domain (if any)
          if (interior_facet_domains && interior_facet_domains->size() > 0)
          {
            const uint domain = (*interior_facet_domains)(*facet);
            if (domain < A_macro_ufc.form.num_interior_facet_integrals())
              interior_facet_integral = A_macro_ufc.interior_facet_integrals[domain];
            else
              continue;
          }


          // Interpolate coefficients on cell
          for (uint i = 0; i < A_coefficients.size(); i++)
          {
            const uint offset = A_macro_ufc.coefficient_elements[i]->spaceDimension();
            A_coefficients[i]->interpolate(A_macro_ufc.macro_w[i],          A_macro_ufc.cell0, 
                *A_macro_ufc.coefficient_elements[i], cell0, facet0);
            A_coefficients[i]->interpolate(A_macro_ufc.macro_w[i] + offset, A_macro_ufc.cell1, 
                *A_macro_ufc.coefficient_elements[i], cell1, facet1);
          }


          // Get integral for sub domain (if any)
          if (interior_facet_domains && interior_facet_domains->size() > 0)
          {
            const uint domain = (*interior_facet_domains)(*facet);
            if (domain < A_macro_ufc.form.num_interior_facet_integrals())
              interior_facet_integral = A_macro_ufc.interior_facet_integrals[domain];
            else
              continue;
          }

          // Tabulate interior facet tensor on macro element
          interior_facet_integral->tabulate_tensor(A_macro_ufc.macro_A, A_macro_ufc.macro_w, 
                                      A_macro_ufc.cell0, A_macro_ufc.cell1, facet0, facet1);
          for (uint i=0; i<A_macro_num_entries; i++) 
            Ae_macro[i] += A_macro_ufc.macro_A[i]; 
        }

        if ( b_ufc.form.num_interior_facet_integrals() > 0 ) 
        {  
          ufc::interior_facet_integral* interior_facet_integral = b_macro_ufc.interior_facet_integrals[0];

          b_macro_ufc.update(cell0, cell1);

          // Interpolate coefficients on cell
          for (uint i = 0; i < b_coefficients.size(); i++)
          {
            const uint offset = b_macro_ufc.coefficient_elements[i]->spaceDimension();
            b_coefficients[i]->interpolate(b_macro_ufc.macro_w[i],          b_macro_ufc.cell0, 
                *b_macro_ufc.coefficient_elements[i], cell0, facet0);
            b_coefficients[i]->interpolate(b_macro_ufc.macro_w[i] + offset, b_macro_ufc.cell1, 
                *b_macro_ufc.coefficient_elements[i], cell1, facet1);
          }

          // Get integral for sub domain (if any)
          if (interior_facet_domains && interior_facet_domains->size() > 0)
          {
            const uint domain = (*interior_facet_domains)(*facet);
            if (domain < b_macro_ufc.form.num_interior_facet_integrals())
              interior_facet_integral = b_macro_ufc.interior_facet_integrals[domain];
            else
              continue;
          }

          interior_facet_integral->tabulate_tensor(b_macro_ufc.macro_A, b_macro_ufc.macro_w, b_macro_ufc.cell0, 
                                                   b_macro_ufc.cell1, facet0, facet1);

          for (uint i=0; i<b_macro_num_entries; i++) 
            be_macro[i] += b_macro_ufc.macro_A[i]; 
        }

        if (facet0 == 0) 
        {
          if (A_ufc.form.num_cell_integrals() > 0) 
          {
            A_ufc.update(cell0);

            // Interpolate coefficients on cell
            for (uint i = 0; i < A_coefficients.size(); i++)
              A_coefficients[i]->interpolate(A_ufc.w[i], A_ufc.cell, *A_ufc.coefficient_elements[i], cell0);

            // Tabulate dofs for each dimension
            for (uint i = 0; i < A_ufc.form.rank(); i++)
              A_dof_map_set[i].tabulate_dofs(A_ufc.dofs[i], A_ufc.cell, cell0.index());

            ufc::cell_integral* A_cell_integral =  A_ufc.cell_integrals[0];

            if (cell_domains && cell_domains->size() > 0)
            {
              const uint cell_domain = (*cell_domains)(cell0);
              if (cell_domain < A_ufc.form.num_cell_integrals()) 
                A_cell_integral = A_ufc.cell_integrals[cell_domain];
              else 
                continue;
            }
            // Tabulate cell tensor
            A_cell_integral->tabulate_tensor(A_ufc.A, A_ufc.w, A_ufc.cell0);

            uint nn = A_ufc.local_dimensions[0]; 
            uint mm = A_ufc.local_dimensions[1];
            for (uint i=0; i<mm; i++) 
              for (uint j=0; j<nn; j++) 
                Ae_macro[2*i*nn+j] += A_ufc.A[i*nn+j]; 
          }
          if (b_ufc.form.num_cell_integrals() > 0) 
          {
            // Update to cell0 
            b_ufc.update(cell0);

            // Interpolate coefficients on cell
            for (uint i = 0; i < b_coefficients.size(); i++)
              b_coefficients[i]->interpolate(b_ufc.w[i], b_ufc.cell, *b_ufc.coefficient_elements[i], cell0);

            // Tabulate dofs for each dimension
            for (uint i = 0; i < b_ufc.form.rank(); i++)
              b_dof_map_set[i].tabulate_dofs(b_ufc.dofs[i], b_ufc.cell, cell0.index());

            ufc::cell_integral* b_cell_integral =  b_ufc.cell_integrals[0];

            if (cell_domains && cell_domains->size() > 0)
            {
              const uint cell_domain = (*cell_domains)(cell0);
              if (cell_domain < b_ufc.form.num_cell_integrals())
                b_cell_integral = b_ufc.cell_integrals[cell_domain];
              else 
                continue;
            }
            // Tabulate cell tensor
            b_cell_integral->tabulate_tensor(b_ufc.A, b_ufc.w, b_ufc.cell0);
            for (uint i=0; i<b_num_entries; i++) 
              be_macro[i] += b_ufc.A[i]; 
          }
        }

        if (facet1 == 0) 
        { 
          if (A_ufc.form.num_cell_integrals() > 0) 
          {
            A_ufc.update(cell1);

            // Interpolate coefficients on cell
            for (uint i = 0; i < A_coefficients.size(); i++)
              A_coefficients[i]->interpolate(A_ufc.w[i], A_ufc.cell, *A_ufc.coefficient_elements[i], cell1);

            // Tabulate dofs for each dimension
            for (uint i = 0; i < A_ufc.form.rank(); i++)
              A_dof_map_set[i].tabulate_dofs(A_ufc.dofs[i], A_ufc.cell, cell1.index());

            ufc::cell_integral* A_cell_integral =  A_ufc.cell_integrals[0];

            if (cell_domains && cell_domains->size() > 0)
            {
              const uint cell_domain = (*cell_domains)(cell1);
              if (cell_domain < A_ufc.form.num_cell_integrals()) 
                A_cell_integral = A_ufc.cell_integrals[cell_domain];
              else 
                continue;
            }

            // Tabulate cell tensor 
            A_cell_integral->tabulate_tensor(A_ufc.A, A_ufc.w, A_ufc.cell1);
            uint nn = A_ufc.local_dimensions[0]; 
            uint mm = A_ufc.local_dimensions[1];
            for (uint i=0; i<mm; i++)
              for (uint j=0; j<nn; j++)
                Ae_macro[2*nn*mm + 2*i*nn + nn + j] += A_ufc.A[i*nn+j]; 
          }

          if (b_ufc.form.num_cell_integrals() > 0) 
          {
            b_ufc.update(cell1);

            // Interpolate coefficients on cell
            for (uint i = 0; i < b_coefficients.size(); i++)
              b_coefficients[i]->interpolate(b_ufc.w[i], b_ufc.cell, *b_ufc.coefficient_elements[i], cell1);

            // Tabulate dofs for each dimension
            for (uint i = 0; i < b_ufc.form.rank(); i++)
              b_dof_map_set[i].tabulate_dofs(b_ufc.dofs[i], b_ufc.cell, cell1.index());

            ufc::cell_integral* b_cell_integral =  b_ufc.cell_integrals[0];

            if (cell_domains && cell_domains->size() > 0)
            {
              const uint cell_domain = (*cell_domains)(cell1);
              if (cell_domain < b_ufc.form.num_cell_integrals())
                b_cell_integral = b_ufc.cell_integrals[cell_domain];
              else 
                continue;
            }
            // Tabulate cell tensor
            b_cell_integral->tabulate_tensor(b_ufc.A, b_ufc.w, b_ufc.cell1);
            for (uint i=0; i<b_num_entries; i++) 
              be_macro[b_num_entries + i] += b_ufc.A[i]; 
          }
        }
        // enforce BC  ---------------------------------------

        const uint m = A_macro_ufc.macro_local_dimensions[0]; 
        const uint n = A_macro_ufc.macro_local_dimensions[1]; 

        for (uint i=0; i<n; i++) 
        {  
          uint ii = A_macro_ufc.macro_dofs[1][i]; 
          if (indicators[ii] > 0) 
          {  
            be[i] = g[ii]; 
            for (uint k=0; k<n; k++) 
              Ae_macro[k+i*n] = 0.0; 
            for (uint j=0; j<m; j++) 
            {
              be_macro[j] -= Ae_macro[i+j*n]*g[ii]; 
              Ae_macro[i+j*n] = 0.0; 
            }
            Ae_macro[i+i*n] = 1.0; 
          }
        }
        // enforce BC done  ------------------------------------------

        // Add entries to global tensor
        A.add(Ae_macro, A_macro_ufc.macro_local_dimensions, A_macro_ufc.macro_dofs);
        b.add(be_macro, b_macro_ufc.macro_local_dimensions, b_macro_ufc.macro_dofs);
      }

      // Check if we have an exterior facet
      if ( facet->numEntities(mesh.topology().dim()) != 2 )  
      {
        // set element matrix and vector to zero 
        for (uint i=0; i<A_num_entries; i++) 
          Ae[i] = 0.0; 
        for (uint i=0; i<b_num_entries; i++) 
          be[i] = 0.0; 

        // Get mesh cell to which mesh facet belongs (pick first, there is only one)
        Cell cell(mesh, facet->entities(mesh.topology().dim())[0]);

        // Get local index of facet with respect to the cell
        const uint local_facet = cell.index(*facet);

        // Update to current cell
        A_ufc.update(cell);
        b_ufc.update(cell);

        // Interpolate coefficients on cell
        for (uint i = 0; i < A_coefficients.size(); i++)
          A_coefficients[i]->interpolate(A_ufc.w[i], A_ufc.cell, *A_ufc.coefficient_elements[i], cell, local_facet);
        // Tabulate dofs for each dimension
        for (uint i = 0; i < A_ufc.form.rank(); i++)
          A_dof_map_set[i].tabulate_dofs(A_ufc.dofs[i], A_ufc.cell, cell.index());

        // Interpolate coefficients on cell
        for (uint i = 0; i < b_coefficients.size(); i++)
          b_coefficients[i]->interpolate(b_ufc.w[i], b_ufc.cell, *b_ufc.coefficient_elements[i], cell, local_facet);

        // Tabulate dofs for each dimension
        for (uint i = 0; i < b_ufc.form.rank(); i++)
          b_dof_map_set[i].tabulate_dofs(b_ufc.dofs[i], b_ufc.cell, cell.index());

        if (local_facet == 0) 
        {
          // compute cell integrals ---------------------------------

          // compute cell for A integral ---------------------------- 
          if (A_ufc.form.num_cell_integrals() > 0) 
          {
            ufc::cell_integral* A_cell_integral =  A_ufc.cell_integrals[0];
            if (cell_domains && cell_domains->size() > 0)
            {
              const uint cell_domain = (*cell_domains)(cell);
              if (cell_domain < A_ufc.form.num_cell_integrals())
                A_cell_integral = A_ufc.cell_integrals[cell_domain];
              else 
                continue;
            }
            // Tabulate cell tensor
            A_cell_integral->tabulate_tensor(A_ufc.A, A_ufc.w, A_ufc.cell);
            for (uint i=0; i<A_num_entries; i++) 
              Ae[i] += A_ufc.A[i]; 
          }
          // compute cell A integral done ---------------------------- 

          // compute cell b integral ---------------------------- 
          if (b_ufc.form.num_cell_integrals() > 0) 
          {
            ufc::cell_integral* b_cell_integral = b_ufc.cell_integrals[0];
            if (cell_domains && cell_domains->size() > 0) 
            {
              const uint cell_domain = (*cell_domains)(cell);
              if (cell_domain < b_ufc.form.num_cell_integrals())
                b_cell_integral = b_ufc.cell_integrals[cell_domain];
              else 
                continue;
            }
            // Tabulate cell tensor
            b_cell_integral->tabulate_tensor(b_ufc.A, b_ufc.w, b_ufc.cell);
            for (uint i=0; i<b_num_entries; i++) 
              be[i] += b_ufc.A[i]; 
          }
          // compute cell b integral done ---------------------------- 

          // compute cell integral done ------------------------------ 
        }

        // compute exterior facet integral ------------------------- 

        if (A_ufc.form.num_exterior_facet_integrals() > 0 ) 
        {
          const uint D = mesh.topology().dim(); 
          if (facet->numEntities(D) != 2) 
          {
            ufc::exterior_facet_integral* A_integral = A_ufc.exterior_facet_integrals[0]; 

            if (exterior_facet_domains && exterior_facet_domains->size() > 0)
            {
              const uint exterior_facet_domain= (*exterior_facet_domains)(*facet);
              if (exterior_facet_domain < A_ufc.form.num_exterior_facet_integrals())
                A_integral = A_ufc.exterior_facet_integrals[exterior_facet_domain];
              else
                continue;
            }
            A_integral->tabulate_tensor(A_ufc.A, A_ufc.w, A_ufc.cell, local_facet);
            for (uint i=0; i<A_num_entries; i++) 
              Ae[i] += A_ufc.A[i]; 
          }
        }

        if (b_ufc.form.num_exterior_facet_integrals() > 0) 
        {
          const uint D = mesh.topology().dim(); 
          if (facet->numEntities(D) != 2) 
          {
            ufc::exterior_facet_integral* b_integral = b_ufc.exterior_facet_integrals[0]; 
            if (exterior_facet_domains && exterior_facet_domains->size() > 0)
            {
              const uint exterior_facet_domain= (*exterior_facet_domains)(*facet);
              if (exterior_facet_domain < b_ufc.form.num_exterior_facet_integrals())
                b_integral = b_ufc.exterior_facet_integrals[exterior_facet_domain];
              else
                continue;
            }
            const uint local_facet = cell.index(*facet);
            b_integral->tabulate_tensor(b_ufc.A, b_ufc.w, b_ufc.cell, local_facet);
            for (uint i=0; i<b_num_entries; i++) 
              be[i] += b_ufc.A[i]; 
          }
        }
        // enforce BC  ---------------------------------------

        uint m = A_ufc.local_dimensions[0]; 
        uint n = A_ufc.local_dimensions[1]; 

        for (uint i=0; i<n; i++) 
        {  
          uint ii = A_ufc.dofs[1][i]; 
          if (indicators[ii] > 0) 
          {  
            be[i] = g[ii]; 
            for (uint k=0; k<n; k++) 
              Ae[k+i*n] = 0.0; 
            for (uint j=0; j<m; j++) 
            {
              be[j] -= Ae[i+j*n]*g[ii]; 
              Ae[i+j*n] = 0.0; 
            }
            Ae[i+i*n] = 1.0; 
          }
        }

        // enforce BC done  ------------------------------------------

        // Add entries to global tensor
        A.add(Ae, A_ufc.local_dimensions, A_ufc.dofs);
        b.add(be, b_ufc.local_dimensions, b_ufc.dofs);
      }
    }
  }

  // -- Finalize tensors 
  A.apply();
  b.apply();

  delete [] Ae;
  delete [] be;
  delete [] g;
  delete [] indicators;
}
//-----------------------------------------------------------------------------

