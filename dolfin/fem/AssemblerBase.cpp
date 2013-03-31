// Copyright (C) 2007-2011 Anders Logg
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
//
// First added:  2007-01-17
// Last changed: 2013-03-30

#include <boost/scoped_ptr.hpp>
#include <dolfin/common/Timer.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/function/GenericFunction.h>
#include <dolfin/la/GenericMatrix.h>
#include <dolfin/la/GenericTensor.h>
#include <dolfin/la/SparsityPattern.h>
#include <dolfin/la/GenericLinearAlgebraFactory.h>
#include <dolfin/la/TensorLayout.h>
#include <dolfin/log/dolfin_log.h>
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

  check_parameters();

  // Get dof maps
  std::vector<const GenericDofMap*> dofmaps;
  for (std::size_t i = 0; i < a.rank(); ++i)
    dofmaps.push_back(a.function_space(i)->dofmap().get());

  if (reset_sparsity)
  {
    Timer t0("Build sparsity");

    // Create layout for initialising tensor
    boost::shared_ptr<TensorLayout> tensor_layout = A.factory().create_layout(a.rank());
    dolfin_assert(tensor_layout);

    std::vector<std::size_t> global_dimensions(a.rank());
    std::vector<std::pair<std::size_t, std::size_t> > local_range(a.rank());
    std::vector<std::size_t> block_sizes;
    for (std::size_t i = 0; i < a.rank(); i++)
    {
      dolfin_assert(dofmaps[i]);
      global_dimensions[i] = dofmaps[i]->global_dimension();
      local_range[i]       = dofmaps[i]->ownership_range();
      block_sizes.push_back(dofmaps[i]->block_size);
    }

    // Set block size for sparsity graphs
    std::size_t block_size = 1;
    if (a.rank() == 2)
    {
      const std::vector<std::size_t> _bs(a.rank(), dofmaps[0]->block_size);
      block_size = (block_sizes == _bs) ? dofmaps[0]->block_size : 1;
    }

    // Initialise tensor layout
    tensor_layout->init(global_dimensions, block_size, local_range);

    // Build sparsity pattern if required
    if (tensor_layout->sparsity_pattern())
    {
      GenericSparsityPattern& pattern = *tensor_layout->sparsity_pattern();
      SparsityPatternBuilder::build(pattern,
                                a.mesh(), dofmaps,
                                a.ufc_form()->has_cell_integrals(),
                                a.ufc_form()->has_interior_facet_integrals(),
                                a.ufc_form()->has_exterior_facet_integrals(),
                                keep_diagonal);
    }
    t0.stop();

    // Initialize tensor
    Timer t1("Init tensor");
    A.init(*tensor_layout);
    t1.stop();

    // Insert zeros on the diagonal as diagonal entries may be prematurely
    // optimised away by the linear algebra backend when calling
    // GenericMatrix::apply, e.g. PETSc does this then errors when matrices
    // have no diagonal entry inserted.
    if (A.rank() == 2 && keep_diagonal)
    {
      // Down cast to GenericMatrix
      GenericMatrix& _A = A.down_cast<GenericMatrix>();

      // Loop over rows and insert 0.0 on the diagonal
      const double block = 0.0;
      const std::pair<std::size_t, std::size_t> row_range = A.local_range(0);
      const std::size_t range = std::min(row_range.second, A.size(1));
      for (std::size_t i = row_range.first; i < range; i++)
      {
        dolfin::la_index _i = i;
        _A.set(&block, 1, &_i, 1, &_i);
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
                     "Reset of tensor in assembly not requested, but dim %d of tensor does not match form", i);
      }
    }
  }

  if (!add_values)
    A.zero();
}
//-----------------------------------------------------------------------------
void AssemblerBase::check_parameters() const
{
  if (reset_sparsity && add_values)
  {
    dolfin_error("AssemblerBase.cpp",
                 "check parameters",
                 "Can not add values when the sparsity pattern is reset");
  }

  if (!reset_sparsity && keep_diagonal)
  {
    dolfin_error("AssemblerBase.cpp",
                 "check parameters",
                 "Not resetting tensor and keeping diagonal entries are incompatible");
  }
}
//-----------------------------------------------------------------------------
void AssemblerBase::check(const Form& a)
{
  dolfin_assert(a.ufc_form());

  // Check the form
  a.check();

  // Extract mesh and coefficients
  const Mesh& mesh = a.mesh();
  const std::vector<boost::shared_ptr<const GenericFunction> >
    coefficients = a.coefficients();

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

    // auto_ptr deletes its object when it exits its scope
    boost::scoped_ptr<ufc::finite_element> fe(a.ufc_form()->create_finite_element(i + a.rank()));

    // Checks out-commented since they only work for Functions, not Expressions
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

  // Check that the cell dimension matches the mesh dimension
  if (a.rank() + a.ufc_form()->num_coefficients() > 0)
  {
    boost::scoped_ptr<ufc::finite_element> element(a.ufc_form()->create_finite_element(0));
    dolfin_assert(element);
    if (mesh.type().cell_type() == CellType::interval && element->cell_shape() != ufc::interval)
    {
      dolfin_error("AssemblerBase.cpp",
                   "assemble form",
                   "Mesh cell type (intervals) does not match cell type of form");
    }
    if (mesh.type().cell_type() == CellType::triangle && element->cell_shape() != ufc::triangle)
    {
      dolfin_error("AssemblerBase.cpp",
                   "assemble form",
                   "Mesh cell type (triangles) does not match cell type of form");
    }
    if (mesh.type().cell_type() == CellType::tetrahedron && element->cell_shape() != ufc::tetrahedron)
    {
      dolfin_error("AssemblerBase.cpp",
                   "assemble form",
                   "Mesh cell type (tetrahedra) does not match cell type of form");
    }
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
