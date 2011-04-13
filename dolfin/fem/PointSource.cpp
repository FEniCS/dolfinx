// Copyright (C) 2011 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2011-04-13
// Last changed: 2011-04-13

#include <boost/scoped_array.hpp>

#include <dolfin/common/NoDeleter.h>
#include <dolfin/la/GenericVector.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/function/FunctionSpace.h>
#include "FiniteElement.h"
#include "GenericDofMap.h"
#include "PointSource.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
PointSource::PointSource(const FunctionSpace& V, const Point& p)
  : V(reference_to_no_delete_pointer(V)), p(p)
{
  // Check that function space is scalar
  check_is_scalar(V);
}
//-----------------------------------------------------------------------------
PointSource::PointSource(boost::shared_ptr<const FunctionSpace> V,
                         const Point& p)
  : V(V), p(p)
{
  // Check that function space is scalar
  check_is_scalar(*V);
}
//-----------------------------------------------------------------------------
PointSource::~PointSource()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void PointSource::apply(GenericVector& b)
{
  assert(V);

  log(PROGRESS, "Applying point source to right-hand side vector.");

  // Find the cell containing the point (may be more than one cell but
  // we only care about the first). Well-defined if the basis
  // functions are continuous but may give unexpected results for DG.
  const Mesh& mesh = V->mesh();
  int cell_index = mesh.any_intersected_entity(p);
  if (cell_index < 0)
    error("Unable to apply point source; point is outside of domain: %s", p.str().c_str());

  // Create cell
  Cell cell(mesh, static_cast<uint>(cell_index));
  UFCCell ufc_cell(cell);

  // Evaluate all basis functions at the point
  assert(V->element().value_rank() == 0);
  boost::scoped_array<double> values(new double[V->element().space_dimension()]);
  V->element().evaluate_basis_all(values.get(), p.coordinates(), ufc_cell);

  // Compute local-to-global mapping
  boost::scoped_array<uint> dofs(new uint[V->dofmap().cell_dimension(cell.index())]);
  V->dofmap().tabulate_dofs(dofs.get(), cell);

  // Add values to vector
  assert(V->element().space_dimension() == V->dofmap().cell_dimension(cell.index()));
  b.set(values.get(), V->element().space_dimension(), dofs.get());
  b.apply("add");
}
//-----------------------------------------------------------------------------
void PointSource::check_is_scalar(const FunctionSpace& V)
{
  if (V.element().value_rank() != 0)
    error("Unable to create point source; function is not scalar.");
}
//-----------------------------------------------------------------------------
