// Copyright (C) 2007-2009 Anders Logg
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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Garth N. Wells, 2007-2010
// Modified by Ola Skavhaug, 2007-2009
// Modified by Kent-Andre Mardal, 2008
//
// First added:  2007-01-17
// Last changed: 2011-02-21

#include <boost/scoped_ptr.hpp>
#include <dolfin/common/Timer.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/function/GenericFunction.h>
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
#include "AssemblerTools.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void AssemblerTools::check(const Form& a)
{
  // Check the form
  a.check();

  // Extract mesh and coefficients
  const Mesh& mesh = a.mesh();
  const std::vector<boost::shared_ptr<const GenericFunction> > coefficients = a.coefficients();

  // Check that we get the correct number of coefficients
  if (coefficients.size() != a.ufc_form().num_coefficients())
  {
    error("Incorrect number of coefficients for form: %d given but %d required.",
          coefficients.size(), a.ufc_form().num_coefficients());
  }

  // Check that all coefficients have valid value dimensions
  for (uint i = 0; i < coefficients.size(); ++i)
  {
    if (!coefficients[i])
      error("Got NULL Function as coefficient %d.", i);

    // auto_ptr deletes its object when it exits its scope
    boost::scoped_ptr<ufc::finite_element> fe(a.ufc_form().create_finite_element(i + a.rank()));

    // Checks outcommented since they only work for Functions, not Expressions
    const uint r = coefficients[i]->value_rank();
    const uint fe_r = fe->value_rank();
    if (fe_r != r)
    {
      error("Invalid value rank for coefficient %d, got %d but expecting %d. \
Did you forget to specify the value rank correctly in an Expression sub class?", i, r, fe_r);
    }

    for (uint j = 0; j < r; ++j)
    {
      const uint dim = coefficients[i]->value_dimension(j);
      const uint fe_dim = fe->value_dimension(j);
      if (dim != fe_dim)
      {
        error("Invalid value dimension %d for coefficient %d, got %d but expecting %d. \
Did you forget to specify the value dimension correctly in an Expression sub class?", j, i, dim, fe_dim);
      }
    }
  }

  // Check that the cell dimension matches the mesh dimension
  if (a.ufc_form().rank() + a.ufc_form().num_coefficients() > 0)
  {
    boost::scoped_ptr<ufc::finite_element> element(a.ufc_form().create_finite_element(0));
    assert(element);
    if (mesh.type().cell_type() == CellType::interval && element->cell_shape() != ufc::interval)
      error("Mesh cell type (intervals) does not match cell type of form.");
    if (mesh.type().cell_type() == CellType::triangle && element->cell_shape() != ufc::triangle)
      error("Mesh cell type (triangles) does not match cell type of form.");
    if (mesh.type().cell_type() == CellType::tetrahedron && element->cell_shape() != ufc::tetrahedron)
      error("Mesh cell type (tetrahedra) does not match cell type of form.");
  }

  // Check that the mesh is ordered
  if (!mesh.ordered())
    error("Unable to assemble, mesh is not correctly ordered (consider calling mesh.order()).");
}
//-----------------------------------------------------------------------------
void AssemblerTools::init_global_tensor(GenericTensor& A, const Form& a,
                                        bool reset_sparsity, bool add_values)
{
  // Check that we should not add values
  if (reset_sparsity && add_values)
    error("Can not add values when the sparsity pattern is reset");

  // Get dof maps
  std::vector<const GenericDofMap*> dofmaps;
  for (uint i = 0; i < a.rank(); ++i)
    dofmaps.push_back(&(a.function_space(i)->dofmap()));

  if (reset_sparsity)
  {
    // Build sparsity pattern
    Timer t0("Build sparsity");
    boost::scoped_ptr<GenericSparsityPattern> sparsity_pattern(A.factory().create_pattern());
    if (sparsity_pattern)
    {

      // Build sparsity pattern
      SparsityPatternBuilder::build(*sparsity_pattern, a.mesh(), dofmaps,
                                    a.ufc_form().num_cell_domains(),
                                    a.ufc_form().num_interior_facet_domains());
    }
    t0.stop();

    // Initialize tensor
    Timer t1("Init tensor");
    if (sparsity_pattern)
      A.init(*sparsity_pattern);
    else
    {
      // Build data structure for intialising sparsity pattern
      std::vector<uint> global_dimensions(a.rank());
      std::vector<std::pair<uint, uint> > local_range(a.rank());
      std::vector<const boost::unordered_map<uint, uint>* > off_process_owner(a.rank());
      for (uint i = 0; i < a.rank(); i++)
      {
        assert(dofmaps[i]);
        global_dimensions[i] = dofmaps[i]->global_dimension();
        local_range[i]       = dofmaps[i]->ownership_range();
        off_process_owner[i] = &(dofmaps[i]->off_process_owner());
      }

      // Create and build sparsity pattern
      SparsityPattern _sparsity_pattern;
      _sparsity_pattern.init(global_dimensions, local_range, off_process_owner);
      A.init(_sparsity_pattern);
      A.zero();
    }
    t1.stop();

    // Delete sparsity pattern
    Timer t2("Delete sparsity");
    t2.stop();
  }
  else
  {
    // If tensor is not reset, check that dimensions are correct
    for (uint i = 0; i < a.rank(); ++i)
    {
      if (A.size(i) != dofmaps[i]-> global_dimension())
        error("Reset of tensor in assembly not requested, but dim %d of tesnor does not match form.", i);
    }
  }

  if (!add_values)
    A.zero();
}
//-----------------------------------------------------------------------------
std::string AssemblerTools::progress_message(uint rank,
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
