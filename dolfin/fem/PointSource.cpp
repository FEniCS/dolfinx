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
#include <dolfin/la/GenericMatrix.h>
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
  : _function_space0(V), _p(p), _magnitude(magnitude)
{
  // Check that function space is scalar
  check_space_supported(*V);
}
//-----------------------------------------------------------------------------
PointSource::PointSource(std::shared_ptr<const FunctionSpace> V0,
			 std::shared_ptr<const FunctionSpace> V1,
                         const Point& p,
                         double magnitude)
  : _function_space0(V0), _function_space1(V1), _p(p), _magnitude(magnitude)
{
  // Check that function space is scalar
  check_space_supported(*V0);
  check_space_supported(*V1);
}
//----------------------------------------------------------------------------
PointSource::PointSource(std::shared_ptr<const FunctionSpace> V0,
			 std::shared_ptr<const FunctionSpace> V1,
			 const std::vector<std::pair<const Point*,
			 double> > sources)
  : _function_space0(V0), _function_space1(V1)
{
  // Copy over from pointers
  for (auto& p : sources)
    _sources.push_back({*(p.first), p.second});


  // Check that function space is scalar
  check_space_supported(*V0);
  check_space_supported(*V1);
  info("instantiated with list");
}
//-----------------------------------------------------------------------------
PointSource::~PointSource()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void PointSource::apply(GenericVector& b)
{
  dolfin_assert(_function_space0);
  dolfin_assert(!_function_space1);

  log(PROGRESS, "Applying point source to right-hand side vector.");

  // Find the cell containing the point (may be more than one cell but
  // we only care about the first). Well-defined if the basis
  // functions are continuous but may give unexpected results for DG.
  dolfin_assert(_function_space0->mesh());
  const Mesh& mesh = *_function_space0->mesh();
  const std::shared_ptr<BoundingBoxTree> tree = mesh.bounding_box_tree();
  const unsigned int cell_index = tree->compute_first_entity_collision(_p);

  // Check that we found the point on at least one processor
  int num_found = 0;
  const bool cell_found_on_process = cell_index != std::numeric_limits<unsigned int>::max();

  if (cell_found_on_process)
    num_found = MPI::sum(mesh.mpi_comm(), 1);
  else
    num_found = MPI::sum(mesh.mpi_comm(), 0);

  info("Rank" + std::to_string(MPI::rank(mesh.mpi_comm())));
  info("num_found" + std::to_string(num_found));

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
    b.apply("add");
    return;
  }

  // Create cell
  const Cell cell(mesh, static_cast<std::size_t>(cell_index));

  // Cell coordinates
  std::vector<double> coordinate_dofs;
  cell.get_coordinate_dofs(coordinate_dofs);

  // Evaluate all basis functions at the point()
  dolfin_assert(_function_space0->element());

  const std::size_t rank = _function_space0->element()->value_rank();
  std::size_t size_basis = 1;
  for (std::size_t i = 0; i < rank; ++i)
    size_basis *= _function_space0->element()->value_dimension(i);

  std::size_t dofs_per_cell = _function_space0->element()->space_dimension();
  std::vector<double> basis(size_basis);
  std::vector<double> values(dofs_per_cell);

  ufc::cell ufc_cell;
  cell.get_cell_data(ufc_cell);

  for (std::size_t i = 0; i < dofs_per_cell; ++i)
  {
    _function_space0->element()->evaluate_basis(i, basis.data(),
						_p.coordinates(),
						coordinate_dofs.data(),
						ufc_cell.orientation);

    double basis_sum = 0.0;
    for (const auto& v : basis)
      basis_sum += v;
    values[i] = _magnitude*basis_sum;
  }

  // Compute local-to-global mapping
  dolfin_assert(_function_space0->dofmap());
  const ArrayView<const dolfin::la_index> dofs
    = _function_space0->dofmap()->cell_dofs(cell.index());

  // Add values to vector
  b.add_local(values.data(), dofs_per_cell, dofs.data());
  b.apply("add");
}
//-----------------------------------------------------------------------------
void PointSource::apply(GenericMatrix& A)
{
  dolfin_assert(_function_space0);

  if (!_function_space1)
    {
      _function_space1=_function_space0;
    }

  dolfin_assert(_function_space1);

  std::shared_ptr<const FunctionSpace> V0 = _function_space0;
  std::shared_ptr<const FunctionSpace> V1 = _function_space1;

  log(PROGRESS, "Applying point source to right-hand side vector.");

  // Find the cell containing the point (may be more than one cell but
  // we only care about the first). Well-defined if the basis
  // functions are continuous but may give unexpected results for DG.
  dolfin_assert(V0->mesh());
  const auto mesh = V0->mesh();

  const std::shared_ptr<BoundingBoxTree> tree = mesh->bounding_box_tree();
  const unsigned int cell_index = tree->compute_first_entity_collision(_p);

  // Check that we found the point on at least one processor
  int num_found = 0;
  const bool cell_found_on_process = cell_index != std::numeric_limits<unsigned int>::max();

  if (cell_found_on_process)
    num_found = MPI::sum(mesh->mpi_comm(), 1);
  else
    num_found = MPI::sum(mesh->mpi_comm(), 0);

  if (MPI::rank(mesh->mpi_comm()) == 0 && num_found == 0)
  {
    dolfin_error("PointSource.cpp",
                 "apply point source to vector",
                 "The point is outside of the domain (%s)", _p.str().c_str());
  }

  const int processes_with_cell =
    cell_found_on_process ? MPI::rank(mesh->mpi_comm()) : -1;
  const int selected_process = MPI::max(mesh->mpi_comm(), processes_with_cell);

  // Return if point not found
  if (MPI::rank(mesh->mpi_comm()) != selected_process)
  {
    A.apply("add");
    return;
  }

  // Create cell
  const Cell cell(*mesh, static_cast<std::size_t>(cell_index));

  // Cell coordinates
  std::vector<double> coordinate_dofs;
  cell.get_coordinate_dofs(coordinate_dofs);

  // Evaluate all basis functions at the point()
  dolfin_assert(V0->element());

  const std::size_t rank = V0->element()->value_rank();
  std::size_t size_basis = 1;
  for (std::size_t i = 0; i < rank; ++i)
    size_basis *= V0->element()->value_dimension(i);

  std::size_t dofs_per_cell0 = V0->element()->space_dimension();
  std::size_t dofs_per_cell1 = V1->element()->space_dimension();
  std::vector<double> basis0(size_basis);
  std::vector<double> basis1(size_basis);
  boost::multi_array<double, 2>  values(boost::extents[dofs_per_cell0][dofs_per_cell1]);

  ufc::cell ufc_cell;
  cell.get_cell_data(ufc_cell);

  for (std::size_t i = 0; i < dofs_per_cell0; ++i)
  {
    V0->element()->evaluate_basis(i, basis0.data(),
				  _p.coordinates(),
				  coordinate_dofs.data(),
				  ufc_cell.orientation);
    for (std::size_t j = 0; j < dofs_per_cell0; ++j)
    {
      V1->element()->evaluate_basis(j, basis1.data(),
				    _p.coordinates(),
				    coordinate_dofs.data(),
				    ufc_cell.orientation);

      double basis_sum0 = 0.0;
      double basis_sum1 = 0.0;
      for (const auto& v : basis0)
	basis_sum0 += v;
      for (const auto& v : basis1)
	basis_sum1 += v;

      values[i][j] = _magnitude*basis_sum0*basis_sum1;
    }
  }

  // Compute local-to-global mapping
  dolfin_assert(V0->dofmap());
  dolfin_assert(V1->dofmap());
  const ArrayView<const dolfin::la_index> dofs0
    = V0->dofmap()->cell_dofs(cell.index());
  const ArrayView<const dolfin::la_index> dofs1
    = V1->dofmap()->cell_dofs(cell.index());

  // Add values to vector
  A.add_local(values.data(),
	      dofs_per_cell0, dofs0.data(),
	      dofs_per_cell1, dofs1.data());
  A.apply("add");
}
//-----------------------------------------------------------------------------
void PointSource::check_space_supported(const FunctionSpace& V)
{
  dolfin_assert(V.element());
  if (V.element()->value_rank() > 1)
  {
    dolfin_error("PointSource.cpp",
                 "create point source",
                 "Function must have rank 0 or 1");
  }
}
//-----------------------------------------------------------------------------
