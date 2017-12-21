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

  // Get mesh
  dolfin_assert(a.mesh());
  const Mesh& mesh = *(a.mesh());

  if (A.empty())
  {
    Timer t0("Build sparsity");

    // Create layout for initialising tensor
    std::shared_ptr<TensorLayout> tensor_layout;
    tensor_layout = A.factory().create_layout(mesh.mpi_comm(), a.rank());
    dolfin_assert(tensor_layout);

    // Get dimensions and mapping across processes for each dimension
    std::vector<std::shared_ptr<const IndexMap>> index_maps;
    for (std::size_t i = 0; i < a.rank(); i++)
    {
      dolfin_assert(dofmaps[i]);
      index_maps.push_back(dofmaps[i]->index_map());
    }

    // Initialise tensor layout
    // FIXME: somewhere need to check block sizes are same on both axes
    // NOTE: Jan: that will be done on the backend side; IndexMap will
    //            provide tabulate functions with arbitrary block size;
    //            moreover the functions will tabulate directly using a
    //            correct int type

    tensor_layout->init(index_maps, TensorLayout::Ghosts::UNGHOSTED);

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

    // Insert zeros to dense rows in increasing order of column index
    // to avoid CSR data reallocation when assembling in random order
    // resulting in quadratic complexity; this has to be done before
    // inserting to diagonal below
    if (tensor_layout->sparsity_pattern())
    {
      // Tabulate indices of dense rows
      const std::size_t primary_dim
        = tensor_layout->sparsity_pattern()->primary_dim();
      std::vector<std::size_t> global_dofs;
      dofmaps[primary_dim]->tabulate_global_dofs(global_dofs);

      if (global_dofs.size() > 0)
      {
        // Down cast tensor to matrix
        dolfin_assert(A.rank() == 2);
        GenericMatrix& _matA = as_type<GenericMatrix>(A);

        // Get local row range
        const std::size_t primary_codim = primary_dim == 0 ? 1 : 0;
        const IndexMap& index_map_0 = *dofmaps[primary_dim]->index_map();
        const std::pair<std::size_t, std::size_t> row_range
          = A.local_range(primary_dim);

        // Set zeros in dense rows in order of increasing column index
        const double block = 0.0;
        dolfin::la_index IJ[2];
        for (std::size_t i: global_dofs)
        {
          const std::size_t I = index_map_0.local_to_global(i);
          if (I >= row_range.first && I < row_range.second)
          {
            IJ[primary_dim] = I;
            for (std::size_t J = 0; J < _matA.size(primary_codim); J++)
            {
              IJ[primary_codim] = J;
              _matA.set(&block, 1, &IJ[0], 1, &IJ[1]);
            }
          }
        }

        // Eventually wait with assembly flush for keep_diagonal
        if (!keep_diagonal)
          A.apply("flush");
      }
    }

    // Insert zeros on the diagonal as diagonal entries may be
    // prematurely optimised away by the linear algebra backend when
    // calling GenericMatrix::apply, e.g. PETSc does this then errors
    // when matrices have no diagonal entry inserted.
    if (keep_diagonal && A.rank() == 2)
    {
      // Down cast to GenericMatrix
      GenericMatrix& _matA = as_type<GenericMatrix>(A);

      // Loop over rows and insert 0.0 on the diagonal
      const double block = 0.0;
      const std::pair<std::size_t, std::size_t> row_range = A.local_range(0);
      const std::size_t range = std::min(row_range.second, A.size(1));

      for (std::size_t i = row_range.first; i < range; i++)
      {
        const dolfin::la_index _i = i;
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
                   "interior facet integrals in parallel)",
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

  std::map<CellType::Type, ufc::shape> dolfin_to_ufc_shapes
    = { {CellType::Type::interval, ufc::shape::interval},
        {CellType::Type::triangle, ufc::shape::triangle},
        {CellType::Type::tetrahedron, ufc::shape::tetrahedron},
        {CellType::Type::quadrilateral, ufc::shape::quadrilateral},
        {CellType::Type::hexahedron, ufc::shape::hexahedron} };

  auto cell_type_pair = dolfin_to_ufc_shapes.find(mesh.type().cell_type());
  dolfin_assert(cell_type_pair != dolfin_to_ufc_shapes.end());
  if (coordinate_element->cell_shape() != cell_type_pair->second)
  {
    dolfin_error("AssemblerBase.cpp",
                 "assemble form",
                 "Mesh cell type does not match cell type of UFC form");
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
