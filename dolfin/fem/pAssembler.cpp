// Copyright (C) 2007-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2007
// Modified by Ola Skavhaug, 2007
// Modified by Magnus Vikstrøm, 2007
//
// First added:  2007-01-17
// Last changed: 2008-08-12

#include <dolfin/log/dolfin_log.h>
#include <dolfin/common/Array.h>
#include <dolfin/la/GenericTensor.h>
#include <dolfin/la/Scalar.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Facet.h>
#include <dolfin/mesh/BoundaryMesh.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/mesh/SubDomain.h>
#include <dolfin/mesh/MeshData.h>
#include <dolfin/function/Function.h>
#include "Form.h"
#include "UFC.h"
#include "pAssembler.h"
#include <dolfin/la/SparsityPattern.h>
#include "SparsityPatternBuilder.h"
#include "DofMapSet.h"
#include "FiniteElement.h"
#include <dolfin/main/MPI.h>

#include <dolfin/common/timing.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
pAssembler::pAssembler(Mesh& mesh, MeshFunction<uint>& partitions) : mesh(mesh), partitions(&partitions)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
pAssembler::pAssembler(Mesh& mesh) : mesh(mesh)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
pAssembler::~pAssembler()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void pAssembler::assemble(GenericTensor& A, Form& form, bool reset_tensor)
{
  form.updateDofMaps(mesh, *partitions);
  assemble(A, form.form(), form.coefficients(), form.dofMaps(), 0, 0, 0, reset_tensor);
}
//-----------------------------------------------------------------------------
void pAssembler::assemble(GenericTensor& A, Form& form,
                         const SubDomain& sub_domain, bool reset_tensor)
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
  form.updateDofMaps(mesh, *partitions);
  assemble(A, form.form(), form.coefficients(), form.dofMaps(),
           cell_domains, facet_domains, facet_domains, reset_tensor);

  // Delete domains
  if (cell_domains)
    delete cell_domains;
  if (facet_domains)
    delete facet_domains;
}
//-----------------------------------------------------------------------------
void pAssembler::assemble(GenericTensor& A, Form& form,
                         const MeshFunction<uint>& cell_domains,
                         const MeshFunction<uint>& exterior_facet_domains,
                         const MeshFunction<uint>& interior_facet_domains,
                         bool reset_tensor)
{
  form.updateDofMaps(mesh, *partitions);
  assemble(A, form.form(), form.coefficients(), form.dofMaps(), &cell_domains, 
           &exterior_facet_domains, &interior_facet_domains, reset_tensor);
}
//-----------------------------------------------------------------------------
dolfin::real pAssembler::assemble(Form& form)
{
  Scalar value;
  assemble(value, form);
  return value;
}
//-----------------------------------------------------------------------------
dolfin::real pAssembler::assemble(Form& form,
                                 const SubDomain& sub_domain)
{
  Scalar value;
  assemble(value, form, sub_domain);
  return value;
}
//-----------------------------------------------------------------------------
dolfin::real pAssembler::assemble(Form& form,
                                 const MeshFunction<uint>& cell_domains,
                                 const MeshFunction<uint>& exterior_facet_domains,
                                 const MeshFunction<uint>& interior_facet_domains)
{
  Scalar value;
  assemble(value, form,
           cell_domains, exterior_facet_domains, interior_facet_domains);
  return value;
}
//-----------------------------------------------------------------------------
void pAssembler::assemble(GenericTensor& A, const ufc::form& form,
                         const Array<Function*>& coefficients,
                         const DofMapSet& dof_map_set,
                         const MeshFunction<uint>* cell_domains,
                         const MeshFunction<uint>* exterior_facet_domains,
                         const MeshFunction<uint>* interior_facet_domains,
                         bool reset_tensor)
{
  message("Assembling rank %d form.", form.rank());

  // Note the importance of treating empty mesh functions as null pointers
  // for the PyDOLFIN interface.
  
  // Check arguments
  check(form, coefficients);

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
void pAssembler::assembleCells(GenericTensor& A,
                              const Array<Function*>& coefficients,
                              const DofMapSet& dof_map_set,
                              UFC& ufc,
                              const MeshFunction<uint>* domains) const
{
  // Skip assembly if there are no cell integrals
  if (ufc.form.num_cell_integrals() == 0)
    return;

  // Cell integral
  ufc::cell_integral* integral = ufc.cell_integrals[0];

  // Assemble over cells
  message("Assembling over %d cells.", mesh.numCells());
  Progress p("Assembling over cells", mesh.numCells());
  
  real t = toc();
  printf("pAssembler: start\n");

  const uint this_process = MPI::process_number();
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    // Assemble only cells in this processors partition
    if (partitions && (*partitions)(*cell) != this_process)
      continue;

    // Get integral for sub domain (if any)
    if (domains && domains->size() > 0)
    {
      if (uint domain = (*domains)(*cell) < ufc.form.num_cell_integrals())
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

  t = toc() - t;
  printf("assembly loop (p): %.3e\n", t);

}
//-----------------------------------------------------------------------------
void pAssembler::assembleExteriorFacets(GenericTensor& A,
                                       const Array<Function*>& coefficients,
                                       const DofMapSet& dof_map_set,
                                       UFC& ufc,
                                       const MeshFunction<uint>* domains) const
{
  // Skip assembly if there are no exterior facet integrals
  if (ufc.form.num_exterior_facet_integrals() == 0)
    return;
  
  // Exterior facet integral
  ufc::exterior_facet_integral* integral = ufc.exterior_facet_integrals[0];

  // Create boundary mesh
  BoundaryMesh boundary(mesh);
  MeshFunction<uint>* cell_map = boundary.data().meshFunction("cell map");
  dolfin_assert(cell_map);
  
  // Assemble over exterior facets (the cells of the boundary)
  message("Assembling over %d exterior facets.", boundary.numCells());
  Progress p("Assembling over exterior facets", boundary.numCells());
  for (CellIterator boundary_cell(boundary); !boundary_cell.end(); ++boundary_cell)
  {
    // Get mesh facet corresponding to boundary cell
    Facet mesh_facet(mesh, (*cell_map)(*boundary_cell));

    // Get integral for sub domain (if any)
    if (domains && domains->size() > 0)
    {
      if (uint domain = (*domains)(mesh_facet) < ufc.form.num_exterior_facet_integrals())
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
    ufc.exterior_facet_integrals[0]->tabulate_tensor(ufc.A, ufc.w, ufc.cell, local_facet);

    // Add entries to global tensor
    A.add(ufc.A, ufc.local_dimensions, ufc.dofs);

    p++;  
  }
}
//-----------------------------------------------------------------------------
void pAssembler::assembleInteriorFacets(GenericTensor& A,
                                       const Array<Function*>& coefficients,
                                       const DofMapSet& dof_map_set,
                                       UFC& ufc,
                                       const MeshFunction<uint>* domains) const
{
  // Skip assembly if there are no interior facet integrals
  if (ufc.form.num_interior_facet_integrals() == 0)
    return;
  
  // Interior facet integral
  ufc::interior_facet_integral* integral = ufc.interior_facet_integrals[0];

  // Compute facets and facet - cell connectivity if not already computed
  mesh.init(mesh.topology().dim() - 1);
  mesh.init(mesh.topology().dim() - 1, mesh.topology().dim());
  mesh.order();
  
  // Assemble over interior facets (the facets of the mesh)
  message("Assembling over %d interior facets.", mesh.numFacets());
  Progress p("Assembling over interior facets", mesh.numFacets());
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
      if (uint domain = (*domains)(*facet) < ufc.form.num_interior_facet_integrals())
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
      dof_map_set[i].tabulate_dofs(ufc.macro_dofs[i], ufc.cell0, cell0.index());
      dof_map_set[i].tabulate_dofs(ufc.macro_dofs[i] + offset, ufc.cell1, cell1.index());
    }

    // Tabulate exterior interior facet tensor on macro element
    ufc.interior_facet_integrals[0]->tabulate_tensor(ufc.macro_A, ufc.macro_w, ufc.cell0, ufc.cell1, facet0, facet1);

    // Add entries to global tensor
    A.add(ufc.macro_A, ufc.macro_local_dimensions, ufc.macro_dofs);

    p++;
  }
}
//-----------------------------------------------------------------------------
void pAssembler::check(const ufc::form& form,
                      const Array<Function*>& coefficients) const
{
  // Check that we get the correct number of coefficients
  if ( coefficients.size() != form.num_coefficients() )
    error("Incorrect number of coefficients for form: %d given but %d required.",
                  coefficients.size(), form.num_coefficients());
}
//-----------------------------------------------------------------------------
void pAssembler::initGlobalTensor(GenericTensor& A, const DofMapSet& dof_map_set, UFC& ufc,
                                 bool reset_tensor) const
{
  if (reset_tensor)
  {
    // Build parallel dof map
    dof_map_set.build(ufc);
    
    // Build sparsity pattern from dof map
    GenericSparsityPattern* sparsity_pattern = A.factory().create_pattern(); 
    SparsityPatternBuilder::build(*sparsity_pattern, mesh, ufc, dof_map_set);
    A.init(*sparsity_pattern);
    delete sparsity_pattern;
  }
  else
    A.zero();
}
//-----------------------------------------------------------------------------
