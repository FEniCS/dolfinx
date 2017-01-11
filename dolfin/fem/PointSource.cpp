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
  : _function_space0(V)
{
  // Puts point and magniude data into a vector
  _sources.push_back({p, magnitude});

  // Check that function space is scalar
  check_space_supported(*V);
}
//-----------------------------------------------------------------------------
PointSource::PointSource(std::shared_ptr<const FunctionSpace> V,
			 const std::vector<std::pair<const Point*,
			 double> > sources)
  : _function_space0(V)
{
  // Copy over from pointers
  for (auto& p : sources)
    _sources.push_back({*(p.first), p.second});

  // Check that function space is scalar
  check_space_supported(*V);
}
//-----------------------------------------------------------------------------
PointSource::PointSource(std::shared_ptr<const FunctionSpace> V0,
			 std::shared_ptr<const FunctionSpace> V1,
                         const Point& p,
                         double magnitude)
  : _function_space0(V0), _function_space1(V1)
{
  // Puts point and magniude data into a vector
  _sources.push_back({p, magnitude});

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

  dolfin_assert(_function_space0->mesh());
  dolfin_assert(_function_space0->element());
  dolfin_assert(_function_space0->dofmap());

  const Mesh& mesh = *_function_space0->mesh();
  const std::shared_ptr<BoundingBoxTree> tree = mesh.bounding_box_tree();
  unsigned int cell_index;

  // Variables for checking that cell is unique
  int num_found;
  bool cell_found_on_process;
  int processes_with_cell;
  unsigned int selected_process;

  // Variables for cell information
  std::vector<double> coordinate_dofs;
  ufc::cell ufc_cell;

  // Variables for evaluating basis
  const std::size_t rank = _function_space0->element()->value_rank();
  std::size_t size_basis = 1;
  for (std::size_t i = 0; i < rank; ++i)
    size_basis *= _function_space0->element()->value_dimension(i);
  std::size_t dofs_per_cell = _function_space0->element()->space_dimension();
  std::vector<double> basis(size_basis);
  std::vector<double> values(dofs_per_cell);

  // Variables for adding local information to vector
  double basis_sum;
  ArrayView<const dolfin::la_index> dofs;

  for (auto & s : _sources)
  {
    Point& p = s.first;
    double magnitude = s.second;

    cell_index = tree->compute_first_entity_collision(p);

    // Check that we found the point on at least one processor
    num_found = 0;
    cell_found_on_process = cell_index
      != std::numeric_limits<unsigned int>::max();
    if (cell_found_on_process)
      {
	num_found = MPI::sum(mesh.mpi_comm(), 1);
      }
    else
      {
	num_found = MPI::sum(mesh.mpi_comm(), 0);
      }
    if (MPI::rank(mesh.mpi_comm()) == 0 && num_found == 0)
    {
      dolfin_error("PointSource.cpp",
		   "apply point source to vector",
		   "The point is outside of the domain (%s)", p.str().c_str());
    }

    processes_with_cell =
      cell_found_on_process ? MPI::rank(mesh.mpi_comm()) : -1;
    selected_process = MPI::max(mesh.mpi_comm(), processes_with_cell);
   
    // Adds point source if found on process
    if (MPI::rank(mesh.mpi_comm()) == selected_process)
    {
      // Create cell
      Cell cell(mesh, static_cast<std::size_t>(cell_index));
      cell.get_coordinate_dofs(coordinate_dofs);

      // Evaluate all basis functions at the point()
      cell.get_cell_data(ufc_cell);

      for (std::size_t i = 0; i < dofs_per_cell; ++i)
      {
	_function_space0->element()->evaluate_basis(i, basis.data(),
						    p.coordinates(),
						    coordinate_dofs.data(),
						    ufc_cell.orientation);

	basis_sum = 0.0;
	for (const auto& v : basis)
	  basis_sum += v;
	values[i] = magnitude*basis_sum;
      }

      // Compute local-to-global mapping

      dofs = _function_space0->dofmap()->cell_dofs(cell.index());

      // Add values to vector
      b.add_local(values.data(), dofs_per_cell, dofs.data());
    }
    b.apply("add");
  }
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

  log(PROGRESS, "Applying point source to matrix.");

  dolfin_assert(V0->mesh());
  dolfin_assert(V0->element());
  dolfin_assert(V1->element());
  dolfin_assert(V0->dofmap());
  dolfin_assert(V1->dofmap());

  const auto mesh = V0->mesh();

  const std::shared_ptr<BoundingBoxTree> tree = mesh->bounding_box_tree();
  unsigned int cell_index;

  // Variables for checking point is unique in cell
  int num_found;
  bool cell_found_on_process;
  int processes_with_cell;
  unsigned int selected_process;

  // Variables for cell information
  std::vector<double> coordinate_dofs;
  ufc::cell ufc_cell;

  // Variables for evaluating basis
  const std::size_t rank = V0->element()->value_rank();
  std::size_t size_basis;
  double basis_sum0;
  double basis_sum1;
  std::size_t num_sub_spaces = V0->element()->num_sub_elements();
  // A scalar function space has 1 sub space but will show as 0
  if (num_sub_spaces == 0)
    num_sub_spaces = 1;
  std::size_t dofs_per_cell0 = V0->element()->space_dimension()/num_sub_spaces;
  std::size_t dofs_per_cell1 = V1->element()->space_dimension()/num_sub_spaces;
  size_basis = 1;
  for (std::size_t i = 0; i < rank; ++i)
    size_basis *= V0->element()->value_dimension(i);
  std::vector<double> basis0(size_basis);
  std::vector<double> basis1(size_basis);
  boost::multi_array<double, 2>  values(boost::extents[dofs_per_cell0*num_sub_spaces][dofs_per_cell1*num_sub_spaces]);
  boost::multi_array<double, 2>  values_sub(boost::extents[dofs_per_cell0][dofs_per_cell1]);

  // Variables for adding local data to matrix
  ArrayView<const dolfin::la_index> dofs0;
  ArrayView<const dolfin::la_index> dofs1;

  for (auto & s : _sources)
  {
    Point& p = s.first;
    double magnitude = s.second;

    MPI::barrier(mesh->mpi_comm());

    cell_index = tree->compute_first_entity_collision(p);

    // Check that we found the point on at least one processor
    num_found= 0;
    cell_found_on_process = cell_index
      != std::numeric_limits<unsigned int>::max();

    if (cell_found_on_process)
      num_found = MPI::sum(mesh->mpi_comm(), 1);
    else
      num_found = MPI::sum(mesh->mpi_comm(), 0);

    if (MPI::rank(mesh->mpi_comm()) == 0 && num_found == 0)
    {
      dolfin_error("PointSource.cpp",
		   "apply point source to vector",
		   "The point is outside of the domain (%s)", p.str().c_str());
    }

    processes_with_cell =
      cell_found_on_process ? MPI::rank(mesh->mpi_comm()) : -1;
    selected_process = MPI::max(mesh->mpi_comm(), processes_with_cell);

    // Return if point not found
    if (MPI::rank(mesh->mpi_comm()) == selected_process)
    {
      // Create cell
      Cell cell(*mesh, static_cast<std::size_t>(cell_index));

      // Cell information
      cell.get_coordinate_dofs(coordinate_dofs);
      cell.get_cell_data(ufc_cell);

      // If a scalar function space calculate values with
      // magnitude*basis_sum_0*basis_sum_1
      if (num_sub_spaces == 0 || num_sub_spaces == 1)
	{
	  for (std::size_t i = 0; i < dofs_per_cell0; ++i)
	    {
	      V0->element()->evaluate_basis(i, basis0.data(),
	                                    p.coordinates(),
	                                    coordinate_dofs.data(),
	                                    ufc_cell.orientation);
	      for (std::size_t j = 0; j < dofs_per_cell0; ++j)
		{
	          V1->element()->evaluate_basis(j, basis1.data(),
				 	        p.coordinates(),
					        coordinate_dofs.data(),
					        ufc_cell.orientation);

		  basis_sum0 = 0.0;
		  basis_sum1 = 0.0;
		  for (const auto& v : basis0)
		    basis_sum0 += v;
		  for (const auto& v : basis1)
		    basis_sum1 += v;
		
		  values[i][j] = magnitude*basis_sum0*basis_sum1;
	        }
	     }
	}

      // If vector function space when sub spaces are all the same,
      // calculates the values for a sub space and then manipulates
      // matrix to add for other sub_spaces
      if (num_sub_spaces > 1)
	{
	  info("Vector");
	  // Only works if sub spaces are the same
	  // FIXME: Extend this check to all subspaces.
	  dolfin_assert(V0->sub(0) = V0->sub(1));

	  // Evaluates basis functions for the first sub space
	  auto V_sub0 = V0->sub(0);
	  auto V_sub1 = V1->sub(0);
	  for (std::size_t i = 0; i < dofs_per_cell0; ++i)
	    {
	      V_sub0->element()->evaluate_basis(i, basis0.data(),
	                                    p.coordinates(),
	                                    coordinate_dofs.data(),
	                                    ufc_cell.orientation);
	      for (std::size_t j = 0; j < dofs_per_cell0; ++j)
		{
	          V_sub1->element()->evaluate_basis(j, basis1.data(),
				 	        p.coordinates(),
					        coordinate_dofs.data(),
					        ufc_cell.orientation);

		  basis_sum0 = 0.0;
		  basis_sum1 = 0.0;
		  for (const auto& v : basis0)
		    basis_sum0 += v;
		  for (const auto& v : basis1)
		    basis_sum1 += v;
		  values_sub[i][j] = magnitude*basis_sum0*basis_sum1;
	        }
	    }

	  // Uses the values calculated on one subspace mirrors them
	  // for all
	  int ii = 0;
	  int jj;

	  for (std::size_t i =0; i< dofs_per_cell0; ++i)
	    { jj = 0;
	      for (std::size_t j = 0; j <dofs_per_cell1; ++j)
		{
		  values[i][j] = values_sub[ii][jj];
		  jj += 1;
		}
	      ii +=1;
	    }

	  ii = 0;
	  for (std::size_t i =dofs_per_cell0; i< dofs_per_cell0*num_sub_spaces; ++i)
	    { jj = 0;
	      for (std::size_t j = dofs_per_cell1; j <dofs_per_cell1*num_sub_spaces; ++j)
		{
		  values[i][j] = values_sub[ii][jj];
		  jj += 1;
		}
	      ii +=1;

	    }
	}

      // Compute local-to-global mapping
      dofs0 = V0->dofmap()->cell_dofs(cell.index());
      dofs1 = V1->dofmap()->cell_dofs(cell.index());

      // Add values to matrix
      A.add_local(values.data(),
		  dofs_per_cell0*num_sub_spaces, dofs0.data(),
		  dofs_per_cell1*num_sub_spaces, dofs1.data());
    }
    A.apply("add");
  }
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
