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
// Last changed: 2012-04-17

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
  dolfin_assert(V);

  log(PROGRESS, "Applying point source to right-hand side vector.");

  // Find the cell containing the point (may be more than one cell but
  // we only care about the first). Well-defined if the basis
  // functions are continuous but may give unexpected results for DG.
  dolfin_assert(V->mesh());
  const Mesh& mesh = *V->mesh();
  const int cell_index = mesh.intersected_cell(p);

  // Check that we found the point on at least one processor
  int num_found = 0;
  if (cell_index < 0)
    num_found = MPI::sum(0);
  else
    num_found = MPI::sum(1);
  if (MPI::process_number() == 0 && num_found == 0)
  {
    dolfin_error("PointSource.cpp",
                 "apply point source to vector",
                 "The point is outside of the domain (%s)", p.str().c_str());
  }

  // Only continue if we found the point
  if (cell_index < 0)
  {
    b.apply("add");
    return;
  }

  // Create cell
  Cell cell(mesh, static_cast<uint>(cell_index));
  UFCCell ufc_cell(cell);

  // Evaluate all basis functions at the point()
  dolfin_assert(V->element());
  dolfin_assert(V->element()->value_rank() == 0);
  std::vector<double> values(V->element()->space_dimension());
  V->element()->evaluate_basis_all(&values[0], p.coordinates(), ufc_cell);

  // Scale by magnitude
  for (uint i = 0; i < V->element()->space_dimension(); i++)
    values[i] *= magnitude;

  // Compute local-to-global mapping
  dolfin_assert(V->dofmap());
  const std::vector<std::size_t>& dofs = V->dofmap()->cell_dofs(cell.index());

  // Add values to vector
  dolfin_assert(V->element()->space_dimension() == V->dofmap()->cell_dimension(cell.index()));
  b.add(&values[0], V->element()->space_dimension(), &dofs[0]);
  b.apply("add");
}
//-----------------------------------------------------------------------------
void PointSource::check_is_scalar(const FunctionSpace& V)
{
  dolfin_assert(V.element());
  if (V.element()->value_rank() != 0)
  {
    dolfin_error("PointSource.cpp",
                 "create point source",
                 "Function is not scalar");
  }
}
//-----------------------------------------------------------------------------
