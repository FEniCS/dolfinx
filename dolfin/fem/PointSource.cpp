// Copyright (C) 2011 Anders Logg
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2011-04-13
// Last changed: 2011-08-10

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
PointSource::PointSource(const FunctionSpace& V,
                         const Point& p,
                         double magnitude)
  : V(reference_to_no_delete_pointer(V)), p(p), magnitude(magnitude)
{
  // Check that function space is scalar
  check_is_scalar(V);
}
//-----------------------------------------------------------------------------
PointSource::PointSource(boost::shared_ptr<const FunctionSpace> V,
                         const Point& p,
                         double magnitude)
  : V(V), p(p), magnitude(magnitude)
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
  assert(V->mesh());
  const Mesh& mesh = *V->mesh();
  int cell_index = mesh.intersected_cell(p);
  if (cell_index < 0)
    error("Unable to apply point source; point is outside of domain: %s", p.str().c_str());

  // Create cell
  Cell cell(mesh, static_cast<uint>(cell_index));
  UFCCell ufc_cell(cell);

  // Evaluate all basis functions at the point()
  assert(V->element());
  assert(V->element()->value_rank() == 0);
  std::vector<double> values(V->element()->space_dimension());
  V->element()->evaluate_basis_all(&values[0], p.coordinates(), ufc_cell);

  // Scale by magnitude
  for (uint i = 0; i < V->element()->space_dimension(); i++)
    values[i] *= magnitude;

  // Compute local-to-global mapping
  assert(V->dofmap());
  std::vector<uint> dofs(V->dofmap()->cell_dimension(cell.index()));
  V->dofmap()->tabulate_dofs(&dofs[0], cell);

  // Add values to vector
  assert(V->element()->space_dimension() == V->dofmap()->cell_dimension(cell.index()));
  b.add(&values[0], V->element()->space_dimension(), &dofs[0]);
  b.apply("add");
}
//-----------------------------------------------------------------------------
void PointSource::check_is_scalar(const FunctionSpace& V)
{
  assert(V.element());
  if (V.element()->value_rank() != 0)
    error("Unable to create point source; function is not scalar.");
}
//-----------------------------------------------------------------------------
