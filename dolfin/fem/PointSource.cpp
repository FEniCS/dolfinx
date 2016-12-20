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
  std::vector<unsigned int> entities2 = tree->compute_entity_collisions(_p);
  for(int i=0; i<entities2.size(); i++)
  {
    info("Entities" + std::to_string(entities2[i]));
  }
  // Check that we found the point on at least one processor
  int num_found = 0;
  if (cell_index == std::numeric_limits<unsigned int>::max())
    num_found = MPI::sum(mesh.mpi_comm(), 0);
  else
    num_found = MPI::sum(mesh.mpi_comm(), 1);
  if (MPI::rank(mesh.mpi_comm()) == 0 && num_found == 0)
  {
    dolfin_error("PointSource.cpp",
                 "apply point source to vector",
                 "The point is outside of the domain (%s)", _p.str().c_str());
  }

  // Return if point not found
  if (cell_index == std::numeric_limits<unsigned int>::max())
  {
    info("Not found on this processor");
    b.apply("add");
    return;
  }

  // Create cell
  const Cell cell(mesh, static_cast<std::size_t>(cell_index));

  // Finds out if any vertices in that cell are shared
  int shared = 0;
  for (VertexIterator v(cell); !v.end(); ++v)
  {
    if(v->is_shared()==true)
    {
      shared = 1;
    }
    info("Is it shared?: " + std::to_string(shared));
  }

  // If shared only apply point source to lowest ranked process.
  if(shared == 1)
  {
    if (MPI::rank(mesh.mpi_comm()) !=
	MPI::min(mesh.mpi_comm(), MPI::rank(mesh.mpi_comm())))
    {
      info("Added on other processor");
      b.apply("add");
      return;
    }
  }

  // Cell coordinates
  std::vector<double> coordinate_dofs;
  cell.get_coordinate_dofs(coordinate_dofs);

  // Evaluate all basis functions at the point()
  dolfin_assert(_function_space->element());
  //dolfin_assert(_function_space->element()->value_rank() == 0);
  //std::size_t size = _function_space->element()->space_dimension()*std::max(_function_space->element()->num_sub_elements(), (std::size_t) 1);

// Compute in tensor (one for scalar function, . . .)
  const std::size_t rank = _function_space->element()->value_rank();
  info("rank " + std::to_string(rank));

  std::size_t size_basis = 1;
  for (std::size_t i = 0; i < rank; ++i)
    size_basis *= _function_space->element()->value_dimension(i);

  info("size basis" + std::to_string(size_basis));

  std::size_t size_values = _function_space->element()->space_dimension();
  std::vector<double> basis(size_basis);
  std::vector<double> values(size_values);

  ufc::cell ufc_cell;
  cell.get_cell_data(ufc_cell);
  info("here");

  std::string msg = "";
  for (auto& v : values)
    msg += std::to_string(v) + ", ";
  info("values:\n" + msg);

  for (std::size_t i = 0; i < size_values/(size_basis); ++i)
  {
    _function_space->element()->evaluate_basis(i, basis.data(), _p.coordinates(),
                           coordinate_dofs.data(),
                           ufc_cell.orientation);
    info(std::to_string(*basis.data()));
    for (std::size_t j = 0; j < rank+1; ++j)
    {
      info("i" + std::to_string(i));
      info("j" + std::to_string(j));
      values[i+j*size_basis] = *basis.data();
    }

  }

  msg = "";
  for (auto& v : values)
    msg += std::to_string(v) + ", ";
  info("values, round 2:\n" + msg);

  // Scale by magnitude
  for (std::size_t i = 0; i < size_values; i++)
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
  b.add_local(values.data(), size_values, dofs.data());
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
