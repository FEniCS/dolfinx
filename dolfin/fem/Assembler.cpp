// Copyright (C) 2007-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2007, 2008
// Modified by Ola Skavhaug, 2007
//
// First added:  2007-01-17
// Last changed: 2008-08-07

#include <dolfin/log/dolfin_log.h>
#include <dolfin/common/Array.h>
#include <dolfin/common/Timer.h>
#include <dolfin/la/GenericTensor.h>
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

#include <dolfin/common/timing.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Assembler::Assembler(Mesh& mesh) : mesh(mesh)
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
  assembleCells(A, coefficients, dof_map_set, ufc, cell_domains);

  // Assemble over exterior facets 
  assembleExteriorFacets(A, coefficients, dof_map_set, ufc, exterior_facet_domains);

  // Assemble over interior facets
  assembleInteriorFacets(A, coefficients, dof_map_set, ufc, interior_facet_domains);

  // Finalise assembly of global tensor
  A.apply();
}
//-----------------------------------------------------------------------------
void Assembler::assembleCells(GenericTensor& A,
                              const Array<Function*>& coefficients,
                              const DofMapSet& dof_map_set,
                              UFC& ufc,
                              const MeshFunction<uint>* domains) const
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
    A.add(ufc.A, ufc.local_dimensions, ufc.dofs);
    
    p++;
  }

  //t = toc() - t;
  //printf("assembly loop (s): %.3e\n", t);
}
//-----------------------------------------------------------------------------
void Assembler::assembleExteriorFacets(GenericTensor& A,
                                       const Array<Function*>& coefficients,
                                       const DofMapSet& dof_map_set,
                                       UFC& ufc,
                                       const MeshFunction<uint>* domains) const
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
                                       const MeshFunction<uint>* domains) const
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
      const uint offset = ufc.coefficient_elements[i]->space_dimension();
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
void Assembler::assemble_system(GenericTensor& A, const ufc::form& A_form, 
                                const Array<Function*>& A_coefficients, 
                                const DofMapSet& A_dof_map_set,
                                GenericTensor& b, const ufc::form& b_form, 
                                const Array<Function*>& b_coefficients, 
                                const DofMapSet& b_dof_map_set, DirichletBC& bc, 
                                const MeshFunction<uint>* cell_domains,
                                const MeshFunction<uint>* exterior_facet_domains,
                                const MeshFunction<uint>* interior_facet_domains,
                                bool reset_tensors)
{
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

  // Assemble over cells
  assembleCells(A, A_coefficients, A_dof_map_set, A_ufc, cell_domains);
  assembleCells(b, b_coefficients, b_dof_map_set, b_ufc, cell_domains);

  // Assemble over exterior facets 
  assembleExteriorFacets(A, A_coefficients, A_dof_map_set, A_ufc, exterior_facet_domains);
  assembleExteriorFacets(b, b_coefficients, b_dof_map_set, b_ufc, exterior_facet_domains);

  // Assemble over interior facets
  assembleInteriorFacets(A, A_coefficients, A_dof_map_set, A_ufc, interior_facet_domains);
  assembleInteriorFacets(b, b_coefficients, b_dof_map_set, b_ufc, interior_facet_domains);

  // Flush assembly of global tensor
  A.apply(FLUSH);
  b.apply(FLUSH);

  // Apply Trace constraints (Dirichlet boundary conditions) 
  applyTraces(A, b, bc, A_dof_map_set, b_dof_map_set, A_form, b_form, exterior_facet_domains);  

  // Finalise BC (need to finalize twice for each tensor) 
  A.apply();
  b.apply();
}
//-----------------------------------------------------------------------------
void Assembler::applyTraces(GenericTensor& globalA, GenericTensor& globalb, 
                            DirichletBC& bc, const DofMapSet& A_dof_map_set,
                            const DofMapSet& b_dof_map_set,
                            const ufc::form& A_form, const ufc::form& b_form, 
                            const MeshFunction<uint>* domains) 
{
  // FIXME check that A and b have proper rank 
  // Create data structure for local assembly data

  // get BC indicators and values from DirichletBC
  uint N = A_dof_map_set[1].global_dimension();  
//  uint M = A_dof_map_set[0].global_dimension();  
  uint* indicators = new uint[N];
  real* x = new real[N];
  for (uint i = 0; i < N; i++) 
  {
    indicators[i] = 0; 
    x[i] = 0.0; 
  }
  bc.getBC(N, indicators, x, A_dof_map_set[1], A_form); 

  /* // For debugging 
  // print out BC indicator and values  
  for (uint k=0; k<N; k++) {
    std::cout <<" indicator        "<<k<<" has value "<<indicators[k]; 
    std::cout <<" corresponding bc "<<k<<" has value "<<x[k]<<std::endl; 
  }
  */ 

  // create local UFC data holders 
  UFC A_ufc(A_form, mesh, A_dof_map_set);
  UFC b_ufc(b_form, mesh, b_dof_map_set);

  // fetch pointers to element matrix and vector  
  BoundaryMesh boundary(mesh);
  MeshFunction<uint>* cell_map = boundary.data().meshFunction("cell map");
  Progress p(progressMessage(globalA.rank(), "exterior facets"), boundary.numCells());
  for (CellIterator boundary_cell(boundary); !boundary_cell.end(); ++boundary_cell)
  {
    // Get mesh facet corresponding to boundary cell
    Facet mesh_facet(mesh, (*cell_map)(*boundary_cell));

    // Get mesh cell to which mesh facet belongs (pick first, there is only one)
    dolfin_assert(mesh_facet.numEntities(mesh.topology().dim()) == 1);
    Cell mesh_cell(mesh, mesh_facet.entities(mesh.topology().dim())[0]);

    // Update to current cell
    A_ufc.update(mesh_cell);
    b_ufc.update(mesh_cell);

    // Tabulate dofs for each dimension 
    A_dof_map_set[0].tabulate_dofs(A_ufc.dofs[0], A_ufc.cell, mesh_cell.index());
    A_dof_map_set[1].tabulate_dofs(A_ufc.dofs[1], A_ufc.cell, mesh_cell.index());
    b_dof_map_set[0].tabulate_dofs(b_ufc.dofs[0], b_ufc.cell, mesh_cell.index());

    // Aji = A[i+j*n]

    // PETSc needs this
    globalA.apply(PETSC_HACK);
    globalb.apply(PETSC_HACK);

    // fetch local element matrix and vector  
    globalA.get(A_ufc.A, A_ufc.local_dimensions, A_ufc.dofs);  
    globalb.get(b_ufc.A, b_ufc.local_dimensions, b_ufc.dofs);  

    uint m = A_ufc.local_dimensions[0]; 
    uint n = A_ufc.local_dimensions[1]; 

    /* 
    // For debugging 
    for (uint i=0; i<m; i++) {
      std::cout << "dof_map 0["<<i<<"]="<<A_ufc.dofs[0][i]<<std::endl; 
    }
    for (uint i=0; i<n; i++) {
      std::cout << "dof_map 1["<<i<<"]="<<A_ufc.dofs[1][i]<<std::endl; 
    }
    */ 

    real* A = A_ufc.A; 
    real* b = b_ufc.A; 

    /*
    // For debugging 
    std::cout <<"-------------------"<<std::endl; 
    std::cout <<"before enforcing bc "<<std::endl; 
    for (uint i=0; i<n; i++) {  
      for (uint j=0; j<m; j++) {
        std::cout <<"i+j*n ="<<j+i*n; 
        std::cout <<" A["<<j<<","<<i<<"]="<<A[j+i*n]<<std::endl; 
      }
    }
    for (uint i=0; i<n; i++) {  
      std::cout <<"b["<<i<<"]="<<b[i]<<std::endl; 
    }
    std::cout <<"-------------------"<<std::endl; 
    */

    // for each dof, check if it is associated with Dirichlet condition   
    for (uint i=0; i<n; i++) 
    {  
      uint ii = A_ufc.dofs[1][i]; 
      if (indicators[ii]) 
      {  
        b[i] = x[ii]; 
        for (uint k=0; k<n; k++) 
          A[k+i*n] = 0.0; 
        for (uint j=0; j<m; j++) 
        {
          b[j] -= A[i+j*n]*x[ii]; 
          A[i+j*n] = 0.0; 
        }
        A[i+i*n] = 1.0; 
      }
    }

    /*
    // For debugging 
    std::cout <<"-------------------"<<std::endl; 
    std::cout <<"after enforcing bc "<<std::endl; 
    for (uint i=0; i<n; i++) {  
      for (uint j=0; j<m; j++) {
        std::cout <<"i+j*n ="<<j+i*n; 
        std::cout <<" A["<<j<<","<<i<<"]="<<A[j+i*n]<<std::endl; 
      }
    }
    for (uint i=0; i<n; i++) {  
      std::cout <<"b["<<i<<"]="<<b[i]<<std::endl; 
    }
    std::cout <<"-------------------"<<std::endl; 
    */ 

    // PETSc needs this
    globalA.apply(PETSC_HACK);
    globalb.apply(PETSC_HACK);

    globalA.set(A_ufc.A, A_ufc.local_dimensions, A_ufc.dofs);  
    globalb.set(b_ufc.A, b_ufc.local_dimensions, b_ufc.dofs);  
  }

  delete [] indicators; 
  delete [] x; 
}
//-----------------------------------------------------------------------------
