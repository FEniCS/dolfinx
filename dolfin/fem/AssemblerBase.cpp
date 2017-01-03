// Copyright (C) 2007-2014 Anders Logg
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
// Modified by Garth N. Wells, 2007-2010
// Modified by Ola Skavhaug, 2007-2009
// Modified by Kent-Andre Mardal, 2008
// Modified by Johannes Ring, 2012
// Modified by Martin Alnaes, 2014

#include <memory>

#include <dolfin/common/Timer.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/function/GenericFunction.h>
#include <dolfin/la/GenericMatrix.h>
#include <dolfin/la/GenericTensor.h>
#include <dolfin/la/SparsityPattern.h>
#include <dolfin/la/GenericLinearAlgebraFactory.h>
#include <dolfin/la/TensorLayout.h>
#include <dolfin/log/log.h>
#include <dolfin/common/MPI.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/Cell.h>

#include "FiniteElement.h"
#include "Form.h"
#include "GenericDofMap.h"
#include "SparsityPatternBuilder.h"
#include "AssemblerBase.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void AssemblerBase::init_global_tensor(GenericTensor& A, const Form& a)
{
  dolfin_assert(a.ufc_form());

  // Get dof maps
  std::vector<const GenericDofMap*> dofmaps;
  for (std::size_t i = 0; i < a.rank(); ++i)
    dofmaps.push_back(a.function_space(i)->dofmap().get());

  if (A.empty())
  {
    Timer t0("Build sparsity");

    // Create layout for initialising tensor
    std::shared_ptr<TensorLayout> tensor_layout;
    tensor_layout = A.factory().create_layout(a.rank());
    dolfin_assert(tensor_layout);

    // Get dimensions and mapping across processes for each dimension
    std::vector<std::shared_ptr<const IndexMap>> index_maps;
    for (std::size_t i = 0; i < a.rank(); i++)
    {
      dolfin_assert(dofmaps[i]);
      index_maps.push_back(dofmaps[i]->index_map());
    }

    // Get mesh
    dolfin_assert(a.mesh());
    const Mesh& mesh = *(a.mesh());

    // Initialise tensor layout
    // FIXME: somewhere need to check block sizes are same on both axes
    // NOTE: Jan: that will be done on the backend side; IndexMap will
    //            provide tabulate functions with arbitrary block size;
    //            moreover the functions will tabulate directly using a
    //            correct int type

    tensor_layout->init(mesh.mpi_comm(), index_maps,
                        TensorLayout::Ghosts::UNGHOSTED);

    // Build sparsity pattern if required
    if (tensor_layout->sparsity_pattern())
    {
      SparsityPattern& pattern = *tensor_layout->sparsity_pattern();
      SparsityPatternBuilder::build(pattern,
                                mesh, dofmaps,
                                a.ufc_form()->has_cell_integrals(),
                                a.ufc_form()->has_interior_facet_integrals(),
                                a.ufc_form()->has_exterior_facet_integrals(),
                                a.ufc_form()->has_vertex_integrals(),
                                keep_diagonal);
    }
    t0.stop();

    // Initialize tensor
    Timer t1("Init tensor");
    A.init(*tensor_layout);
    t1.stop();

    // Insert zeros on the diagonal as diagonal entries may be
    // prematurely optimised away by the linear algebra backend when
    // calling GenericMatrix::apply, e.g. PETSc does this then errors
    // when matrices have no diagonal entry inserted.
    if (A.rank() == 2 && keep_diagonal)
    {
      // Down cast to GenericMatrix
      GenericMatrix& _matA = A.down_cast<GenericMatrix>();

      // Loop over rows and insert 0.0 on the diagonal
      const double block = 0.0;
      const std::pair<std::size_t, std::size_t> row_range = A.local_range(0);
      const std::size_t range = std::min(row_range.second, A.size(1));
      for (std::size_t i = row_range.first; i < range; i++)
      {
        dolfin::la_index _i = i;
        _matA.set(&block, 1, &_i, 1, &_i);
      }
      A.apply("flush");
    }

    // Delete sparsity pattern
    Timer t2("Delete sparsity");
    t2.stop();
  }
  else
  {
    // If tensor is not reset, check that dimensions are correct
    for (std::size_t i = 0; i < a.rank(); ++i)
    {
      if (A.size(i) != dofmaps[i]->global_dimension())
      {
        dolfin_error("AssemblerBase.cpp",
                     "assemble form",
                     "Dim %d of tensor does not match form", i);
      }
    }
  }

  if (!add_values)
    A.zero();
}
//-----------------------------------------------------------------------------
void AssemblerBase::check(const Form& a)
{
  dolfin_assert(a.ufc_form());

  // Check the form
  a.check();

  // Extract mesh and coefficients
  dolfin_assert(a.mesh());
  const Mesh& mesh = *(a.mesh());
  const std::vector<std::shared_ptr<const GenericFunction>>
    coefficients = a.coefficients();

  // Check ghost mode for interior facet integrals in parallel
  if (a.ufc_form()->has_interior_facet_integrals()
      && MPI::size(mesh.mpi_comm()) > 1)
  {
    std::string ghost_mode = mesh.ghost_mode();
    if (!(ghost_mode == "shared_vertex" || ghost_mode == "shared_facet"))
    {
      dolfin_error("AssemblerBase.cpp",
                   "assemble form",
                   "Incorrect mesh ghost mode \"%s\" (expected "
                   "\"shared_vertex\" or \"shared_facet\" for "
                   "interior facet intergrals in parallel)",
                   ghost_mode.c_str());
    }
  }

  // Check that we get the correct number of coefficients
  if (coefficients.size() != a.num_coefficients())
  {
    dolfin_error("AssemblerBase.cpp",
                 "assemble form",
                 "Incorrect number of coefficients (got %d but expecting %d)",
                 coefficients.size(), a.num_coefficients());
  }

  // Check that all coefficients have valid value dimensions
  for (std::size_t i = 0; i < coefficients.size(); ++i)
  {
    if (!coefficients[i])
    {
      dolfin_error("AssemblerBase.cpp",
                   "assemble form",
                   "Coefficient number %d (\"%s\") has not been set",
                   i, a.coefficient_name(i).c_str());
    }

    // unique_ptr deletes its object when it exits its scope
    std::unique_ptr<ufc::finite_element>
      fe(a.ufc_form()->create_finite_element(i + a.rank()));

    // Checks out-commented since they only work for Functions, not
    // Expressions
    const std::size_t r = coefficients[i]->value_rank();
    const std::size_t fe_r = fe->value_rank();
    if (fe_r != r)
    {
      dolfin_error("AssemblerBase.cpp",
                   "assemble form",
                   "Invalid value rank for coefficient %d (got %d but expecting %d). \
You might have forgotten to specify the value rank correctly in an Expression subclass", i, r, fe_r);
    }

    for (std::size_t j = 0; j < r; ++j)
    {
      const std::size_t dim = coefficients[i]->value_dimension(j);
      const std::size_t fe_dim = fe->value_dimension(j);
      if (dim != fe_dim)
      {
        dolfin_error("AssemblerBase.cpp",
                     "assemble form",
                     "Invalid value dimension %d for coefficient %d (got %d but expecting %d). \
You might have forgotten to specify the value dimension correctly in an Expression subclass", j, i, dim, fe_dim);
      }
    }
  }

  // Check that the coordinate cell matches the mesh
  std::unique_ptr<ufc::finite_element>
    coordinate_element(a.ufc_form()->create_coordinate_finite_element());
  dolfin_assert(coordinate_element);
  dolfin_assert(coordinate_element->value_rank() == 1);
  if (coordinate_element->value_dimension(0) != mesh.geometry().dim())
  {
    dolfin_error("AssemblerBase.cpp",
                 "assemble form",
                 "Geometric dimension of Mesh does not match value shape of coordinate element in form");
  }

  // Check that the coordinate element degree matches the mesh degree
  if (coordinate_element->degree() != mesh.geometry().degree())
  {
    dolfin_error("AssemblerBase.cpp",
                 "assemble form",
                 "Mesh geometry degree does not match degree of coordinate element in form");
  }

  switch (mesh.type().cell_type())
  {
  case CellType::interval:
    if (coordinate_element->cell_shape() != ufc::shape::interval)
    {
      dolfin_error("AssemblerBase.cpp",
                   "assemble form",
                   "Mesh cell type (intervals) does not match cell type of form");
    }
    break;
  case CellType::triangle:
    if (coordinate_element->cell_shape() != ufc::shape::triangle)
    {
      dolfin_error("AssemblerBase.cpp",
                   "assemble form",
                   "Mesh cell type (triangles) does not match cell type of form");
    }
    break;
  case CellType::tetrahedron:
    if (coordinate_element->cell_shape() != ufc::shape::tetrahedron)
    {
      dolfin_error("AssemblerBase.cpp",
                   "assemble form",
                   "Mesh cell type (tetrahedra) does not match cell type of form");
    }
    break;
  case CellType::quadrilateral:
    if (coordinate_element->cell_shape() != ufc::shape::quadrilateral)
    {
      dolfin_error("AssemblerBase.cpp",
                   "assemble form",
                   "Mesh cell type (quadrilateral) does not match cell type of form");
    }
    break;
  case CellType::hexahedron:
    if (coordinate_element->cell_shape() != ufc::shape::hexahedron)
    {
      dolfin_error("AssemblerBase.cpp",
                   "assemble form",
                   "Mesh cell type (hexahedron) does not match cell type of form");
    }
    break;
  default:
    dolfin_error("AssemblerBase.cpp",
                 "assemble form",
                 "Mesh cell type is unknown!");
  }

  // Check that the mesh is ordered
  if (!mesh.ordered())
  {
    dolfin_error("AssemblerBase.cpp",
                 "assemble form",
                 "Mesh is not correctly ordered. Consider calling mesh.order()");
  }
}
//-----------------------------------------------------------------------------
std::string AssemblerBase::progress_message(std::size_t rank,
                                            std::string integral_type)
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
