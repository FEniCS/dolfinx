// Copyright (C) 2011-2013 Anders Logg
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
// Last changed: 2014-03-25

#include <limits>
#include <memory>
#include <vector>

#include <dolfin/common/NoDeleter.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/geometry/BoundingBoxTree.h>
#include <dolfin/la/GenericVector.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshEntityIteratorBase.h>
#include <dolfin/mesh/Vertex.h>
#include "FiniteElement.h"
#include "GenericDofMap.h"
#include "PointSource.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
PointSource::PointSource(std::shared_ptr<const FunctionSpace> V,
                         const Point& p,
                         double magnitude)
  : _function_space(V), _p(p), _magnitude(magnitude)
{
  // Check that function space is scalar
  //check_is_scalar(*V);
}
//-----------------------------------------------------------------------------
PointSource::~PointSource()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void PointSource::apply(GenericVector& b)
{
  dolfin_assert(_function_space);

  log(PROGRESS, "Applying point source to right-hand side vector.");

  // Find the cell containing the point (may be more than one cell but
  // we only care about the first). Well-defined if the basis
  // functions are continuous but may give unexpected results for DG.
  dolfin_assert(_function_space->mesh());
  const Mesh& mesh = *_function_space->mesh();
  std::shared_ptr<BoundingBoxTree> tree = mesh.bounding_box_tree();
  const unsigned int cell_index = tree->compute_first_entity_collision(_p);

  // Check that we found the point on at least one processor
  int num_found = 0;
  const bool cell_found_on_process = cell_index != std::numeric_limits<unsigned int>::max();

  if (cell_found_on_process)
    num_found = MPI::sum(mesh.mpi_comm(), 1);
  else
    num_found = MPI::sum(mesh.mpi_comm(), 0);

  if (MPI::rank(mesh.mpi_comm()) == 0 && num_found == 0)
  {
    dolfin_error("PointSource.cpp",
                 "apply point source to vector",
                 "The point is outside of the domain (%s)", _p.str().c_str());
  }

  const int processes_with_cell =
    cell_found_on_process ? MPI::rank(mesh.mpi_comm()) : -1;
  const int selected_process = MPI::max(mesh.mpi_comm(), processes_with_cell);

  // Return if point not found
  if (MPI::rank(mesh.mpi_comm()) != selected_process)
  {
    info("Not found on this processor");
    b.apply("add");
    return;
  }

  // Create cell
  const Cell cell(mesh, static_cast<std::size_t>(cell_index));

  // Cell coordinates
  std::vector<double> coordinate_dofs;
  cell.get_coordinate_dofs(coordinate_dofs);

  // Evaluate all basis functions at the point()
  dolfin_assert(_function_space->element());

  // Compute in tensor (one for scalar function, . . .)
  const std::size_t rank = _function_space->element()->value_rank();
  info("rank " + std::to_string(rank));

  std::size_t size_basis = 1;
  for (std::size_t i = 0; i < rank; ++i)
    size_basis *= _function_space->element()->value_dimension(i);

  info("size basis" + std::to_string(size_basis));

  std::size_t dofs_per_cell = _function_space->element()->space_dimension();
  std::vector<double> basis(size_basis);
  std::vector<double> values(dofs_per_cell);

  info("dofs per cell " + std::to_string(dofs_per_cell));
  ufc::cell ufc_cell;
  cell.get_cell_data(ufc_cell);

  std::string msg = "";
  for (auto& v : values)
    msg += std::to_string(v) + ", ";
  info("values at dofs:\n" + msg);

  for (std::size_t i = 0; i < dofs_per_cell; ++i)
  {
    _function_space->element()->evaluate_basis(i, basis.data(), _p.coordinates(),
                           coordinate_dofs.data(),
                           ufc_cell.orientation);
    info("basis size: " + std::to_string(basis.size()));
    msg = "";
    for (auto& v : basis)
      msg += std::to_string(v) + ", ";
    info("basis:\n" + msg);

    double basis_sum = 0.0;
    for (const auto& v : basis)
      basis_sum += v;
    values[i] = basis_sum;
  }

  msg = "";
  for (auto& v : values)
    msg += std::to_string(v) + ", ";
  info("values, round 2:\n" + msg);

  // Scale by magnitude
  for (std::size_t i = 0; i < dofs_per_cell; i++)
    values[i] *= _magnitude;

  msg = "";
  for (auto& v : values)
    msg += std::to_string(v) + ", ";
  info("values:\n" + msg);

  // Compute local-to-global mapping
  dolfin_assert(_function_space->dofmap());
  const ArrayView<const dolfin::la_index> dofs
    = _function_space->dofmap()->cell_dofs(cell.index());

  msg = "";
  for (auto& v : dofs)
    msg += std::to_string(v) + ", ";
  info("dofs:\n" + msg);

  // Add values to vector
  //dolfin_assert(_function_space->element()->space_dimension()
  //              == _function_space->dofmap()->num_element_dofs(cell.index()));
  b.add_local(values.data(), dofs_per_cell, dofs.data());
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
