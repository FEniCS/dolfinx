// Copyright (C) 2009 Bartosz Sawicki.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-04-03
// Last changed: 2009-04-10

#include <vector>

#include <dolfin/common/constants.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/Vertex.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Facet.h>
#include <dolfin/mesh/SubDomain.h>
#include <dolfin/la/GenericMatrix.h>
#include <dolfin/la/GenericVector.h>
#include "DofMap.h"
#include "UFCMesh.h"
#include "UFCCell.h"
#include "BoundaryCondition.h"
#include "EqualityBC.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
EqualityBC::EqualityBC(const FunctionSpace& V,
                       const SubDomain& sub_domain)
  : BoundaryCondition(V)
{
  init_from_sub_domain(sub_domain);
}
//-----------------------------------------------------------------------------
EqualityBC::EqualityBC(const FunctionSpace& V,
                         uint sub_domain)
  : BoundaryCondition(V)
{
  init_from_mesh(sub_domain);
}
//-----------------------------------------------------------------------------
EqualityBC::~EqualityBC()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void EqualityBC::apply(GenericMatrix& A) const
{
  dolfin_not_implemented();
}
//-----------------------------------------------------------------------------
void EqualityBC::apply(GenericVector& b) const
{
  dolfin_not_implemented();
}
//-----------------------------------------------------------------------------
void EqualityBC::apply(GenericMatrix& A, GenericVector& b) const
{
  info("Applying equality boundary conditions to linear system.");

  if (equal_dofs.size() < 2)
  {
    warning("No enough dofs to set equality boundary condition.");
    return;
  }

  // Insert -1 at (dof0, dof1) and 0 on right-hand side
  uint* rows = new uint[1];
  uint* cols = new uint[1];
  double* vals = new double[1];
  double* zero = new double[1];

  // First dof is our reference
  const int dof0 = equal_dofs[0];

  for (std::vector<uint>::const_iterator it = equal_dofs.begin(); it != equal_dofs.end(); ++it)
  {
    // FIXME: This can be done more efficiently. A.apply() should not be called
    //        from within a loop.

    // Get dof1
    const int dof1 = *it;

    // Avoid modification for reference
    if (dof1 == dof0) continue;

    // Set x_0 - x_i = 0
    rows[0] = static_cast<uint>(dof0);
    cols[0] = static_cast<uint>(dof1);
    vals[0] = -1.0;
    zero[0] = 0.0;

    std::vector<uint> columns;
    std::vector<double> values;

    // Add slave-dof-row to master-dof-row
    A.getrow(dof0, columns, values);
    A.add(&values[0], 1, &cols[0], columns.size(), &columns[0]);

    // Add slave-dof-entry to master-dof-entry
    values.resize(1);
    b.get(&values[0], 1, &rows[0]);
    b.add(&values[0], 1, &cols[0]);

    // Apply changes before using set
    A.apply();
    b.apply();

    // Replace slave-dof equation by relation enforcing equality
    A.ident(1, rows);
    A.set(vals, 1, rows, 1, cols);
    b.set(zero, 1, rows);

    // Apply changes
    A.apply();
    b.apply();
  }

  // Cleanup
  delete [] rows;
  delete [] cols;
  delete [] vals;
  delete [] zero;
}
//-----------------------------------------------------------------------------
void EqualityBC::apply(GenericVector& b, const GenericVector& x) const
{
  dolfin_not_implemented();
}
//-----------------------------------------------------------------------------
void EqualityBC::apply(GenericMatrix& A,
                       GenericVector& b,
                       const GenericVector& x) const
{
  dolfin_not_implemented();
}
//-----------------------------------------------------------------------------
void EqualityBC::init_from_sub_domain(const SubDomain& sub_domain)
{
  dolfin_assert(equal_dofs.size() == 0);

  // Get mesh and dofmap
  const Mesh& mesh = V->mesh();
  const DofMap& dofmap = V->dofmap();

  // Create local data for application of boundary conditions
  BoundaryCondition::LocalData data(*V);

  // Make sure we have the facet - cell connectivity
  const uint D = mesh.topology().dim();
  mesh.init(D - 1, D);

  // Create UFC view of mesh
  UFCMesh ufc_mesh(mesh);

  // Iterate over the facets of the mesh
  for (FacetIterator facet(mesh); !facet.end(); ++facet)
  {
    // Get cell to which facet belongs (there may be two, but pick first)
    Cell cell(mesh, facet->entities(D)[0]);
    UFCCell ufc_cell(cell);

    // Get local index of facet with respect to the cell
    const uint local_facet = cell.index(*facet);

    // Tabulate dofs on cell
    dofmap.tabulate_dofs(data.cell_dofs, ufc_cell, cell.index());

    // Tabulate coordinates on cell
    dofmap.tabulate_coordinates(data.coordinates, ufc_cell);

    // Tabulate which dofs are on the facet
    dofmap.tabulate_facet_dofs(data.facet_dofs, local_facet);

    // Iterate over facet dofs
    for (uint i = 0; i < dofmap.num_facet_dofs(); i++)
    {
      // Get dof and coordinate of dof
      const uint local_dof = data.facet_dofs[i];
      const int global_dof = static_cast<int>(dofmap.offset() + data.cell_dofs[local_dof]);
      double* x = data.coordinates[local_dof];

      // Check if coordinate is inside the domain
      const bool on_boundary = facet->num_entities(D) == 1;
      if (sub_domain.inside(x, on_boundary))
      {
        equal_dofs.push_back(global_dof);
      }
    }
  }
}
//-----------------------------------------------------------------------------
void EqualityBC::init_from_mesh(uint sub_domain)
{
  dolfin_assert(equal_dofs.size() == 0);

  // Get data from MeshData
  std::vector<uint>* facet_cells   = const_cast<Mesh&>(V->mesh()).data().array("boundary facet cells");
  std::vector<uint>* facet_numbers = const_cast<Mesh&>(V->mesh()).data().array("boundary facet numbers");
  std::vector<uint>* indicators    = const_cast<Mesh&>(V->mesh()).data().array("boundary indicators");

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

  // Get mesh and dofmap
  const Mesh& mesh = V->mesh();
  const DofMap& dofmap = V->dofmap();

  // Create local data for application of boundary conditions
  BoundaryCondition::LocalData data(*V);

  // Build set of boundary facets
  for (uint i = 0; i < size; i++)
  {
    // Skip facets not on this boundary
    if ((*indicators)[i] != sub_domain)
      continue;

    // Get cell
    Cell cell(mesh, (*facet_cells)[i]);
    UFCCell ufc_cell(cell);

    // Get local index of facet with respect to the cell
    const uint local_facet = (*facet_numbers)[i];

    // Tabulate dofs on cell
    dofmap.tabulate_dofs(data.cell_dofs, ufc_cell, cell.index());

    // Tabulate coordinates on cell
    dofmap.tabulate_coordinates(data.coordinates, ufc_cell);

    // Tabulate which dofs are on the facet
    dofmap.tabulate_facet_dofs(data.facet_dofs, local_facet);

    // Iterate over facet dofs
    for (uint i = 0; i < dofmap.num_facet_dofs(); i++)
    {
      // Get dof and coordinate of dof
      const uint local_dof = data.facet_dofs[i];
      const int global_dof = static_cast<int>(dofmap.offset() + data.cell_dofs[local_dof]);

      // Store dof
      equal_dofs.push_back(global_dof);
    }
  }
}
//-----------------------------------------------------------------------------
