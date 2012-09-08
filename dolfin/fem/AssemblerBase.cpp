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
// Last changed: 2012-03-02

#include <boost/scoped_ptr.hpp>
#include <dolfin/common/Timer.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/function/GenericFunction.h>
#include <dolfin/la/GenericMatrix.h>
#include <dolfin/la/GenericTensor.h>
#include <dolfin/la/SparsityPattern.h>
#include <dolfin/la/LinearAlgebraFactory.h>
#include <dolfin/log/dolfin_log.h>
#include <dolfin/common/MPI.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/Cell.h>

#include "FiniteElement.h"
#include "Form.h"
#include "GenericDofMap.h"
#include "SparsityPatternBuilder.h"
#include "AssemblerBase.h"

#include <dolfin/la/TensorLayout.h>

using namespace dolfin;

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
  for (uint i = 0; i < coefficients.size(); ++i)
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

    // Checks outcommented since they only work for Functions, not Expressions
    const uint r = coefficients[i]->value_rank();
    const uint fe_r = fe->value_rank();
    if (fe_r != r)
    {
      dolfin_error("AssemblerBase.cpp",
                   "assemble form",
                   "Invalid value rank for coefficient %d (got %d but expecting %d). \
You might have forgotten to specify the value rank correctly in an Expression subclass", i, r, fe_r);
    }

    for (uint j = 0; j < r; ++j)
    {
      const uint dim = coefficients[i]->value_dimension(j);
      const uint fe_dim = fe->value_dimension(j);
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
void AssemblerBase::init_global_tensor(GenericTensor& A, const Form& a,
          const std::vector<std::pair<std::pair<uint, uint>, std::pair<uint, uint> > >& periodic_master_slave_dofs,
          bool reset_sparsity, bool add_values, bool keep_diagonal)
{
  dolfin_assert(a.ufc_form());

  // Check that we should not add values
  if (reset_sparsity && add_values)
  {
    dolfin_error("AssemblerBase.cpp",
                 "assemble form",
                 "Can not add values when the sparsity pattern is reset");
  }

  // Get dof maps
  std::vector<const GenericDofMap*> dofmaps;
  for (uint i = 0; i < a.rank(); ++i)
    dofmaps.push_back(a.function_space(i)->dofmap().get());

  if (reset_sparsity)
  {
    Timer t0("Build sparsity");

    // Create layout for intialising tensor
    boost::shared_ptr<TensorLayout> tensor_layout = A.factory().create_layout(a.rank());
    dolfin_assert(tensor_layout);

    std::vector<uint> global_dimensions(a.rank());
    std::vector<std::pair<uint, uint> > local_range(a.rank());
    for (uint i = 0; i < a.rank(); i++)
    {
      dolfin_assert(dofmaps[i]);
      global_dimensions[i] = dofmaps[i]->global_dimension();
      local_range[i]       = dofmaps[i]->ownership_range();
    }
    tensor_layout->init(global_dimensions, local_range);

    // Build sparsity pattern if required
    if (tensor_layout->sparsity_pattern())
    {
      GenericSparsityPattern& pattern = *tensor_layout->sparsity_pattern();
      SparsityPatternBuilder::build(pattern,
                                a.mesh(), dofmaps, periodic_master_slave_dofs,
                                a.ufc_form()->num_cell_domains(),
                                a.ufc_form()->num_interior_facet_domains(),
                                a.ufc_form()->num_exterior_facet_domains(),
                                keep_diagonal);
    }
    t0.stop();

    // Initialize tensor
    Timer t1("Init tensor");
    A.init(*tensor_layout);
    t1.stop();

    // Insert zeros in the diagonal as diagonal entries may be prematurely
    // optimised away by the linear algebra backend when calling
    // GenericMatrix::apply, e.g. PETSc does this then errors when matrices
    // have no diagonal entry.
    if (A.rank() == 2 && keep_diagonal)
    {
      GenericMatrix& _A = A.down_cast<GenericMatrix>();

      const double block = 0.0;

      const std::pair<uint, uint> row_range = A.local_range(0);
      // Loop over rows
      for (uint i = row_range.first; i < row_range.second; i++)
      {
        // Get global row number
        _A.set(&block, (uint) 1, &i, (uint) 1, &i);

      }
      A.apply("flush");
    }

    // Insert zeros in positions required for periodic boundary
    // conditions. These are applied post-assembly, and may be prematurely
    // optimised away by the linear algebra backend when calling
    // GenericMatrix::apply, e.g. PETSc does this
    if (A.rank() == 2)
    {
      if (tensor_layout->sparsity_pattern())
      {
        const GenericSparsityPattern& pattern = *tensor_layout->sparsity_pattern();
        if (pattern.primary_dim() != 0)
        {
          dolfin_error("AssemblerBase.cpp",
                       "insert zero values in periodic boundary condition positions",
                       "Modifcation of non-zero matrix pattern for periodic boundary conditions is supported row-wise matrices only");
        }

        GenericMatrix& _A = A.down_cast<GenericMatrix>();
        std::vector<std::pair<std::pair<uint, uint>, std::pair<uint, uint> > >::const_iterator dof_pair;
        for (dof_pair = periodic_master_slave_dofs.begin(); dof_pair != periodic_master_slave_dofs.end(); ++dof_pair)
        {
          const uint dofs[2] = {dof_pair->first.first, dof_pair->second.first};

          std::vector<uint> edges;
          for (uint i = 0; i < 2; ++i)
          {
            if (dofs[i] >= pattern.local_range(0).first && dofs[i] < pattern.local_range(0).second)
            {
              pattern.get_edges(dofs[i], edges);
              const std::vector<double> block(edges.size(), 0.0);
              _A.set(&block[0], (uint) 1, &dofs[i], (uint) edges.size(), &edges[0]);
            }
          }
        }
        A.apply("flush");
      }
    }

    // Delete sparsity pattern
    Timer t2("Delete sparsity");
    t2.stop();
  }
  else
  {
    // If tensor is not reset, check that dimensions are correct
    for (uint i = 0; i < a.rank(); ++i)
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
std::string AssemblerBase::progress_message(uint rank,
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
