// Copyright (C) 2007-2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2007-2009
// Modified by Ola Skavhaug, 2007-2009
// Modified by Kent-Andre Mardal, 2008
//
// First added:  2007-01-17
// Last changed: 2009-12-15

#include <dolfin/log/dolfin_log.h>
#include <dolfin/common/Timer.h>
#include <dolfin/la/GenericTensor.h>
#include <dolfin/la/SparsityPattern.h>
#include <dolfin/la/LinearAlgebraFactory.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/function/GenericFunction.h>
#include "DofMap.h"
#include "Form.h"
#include "UFC.h"
#include "AssemblerTools.h"
#include "SparsityPatternBuilder.h"
#include "DirichletBC.h"
#include "FiniteElement.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void AssemblerTools::check(const Form& a)
{
  // Check the form
  a.check();

  // Extract mesh and coefficients
  const Mesh& mesh = a.mesh();
  const std::vector<const GenericFunction*> coefficients = a.coefficients();

  // Check that we get the correct number of coefficients
  if (coefficients.size() != a.ufc_form().num_coefficients())
    error("Incorrect number of coefficients for form: %d given but %d required.",
          coefficients.size(), a.ufc_form().num_coefficients());

  // Check that all coefficients have valid value dimensions
  for (uint i = 0; i < coefficients.size(); ++i)
  {
    if(!coefficients[i])
      error("Got NULL Function as coefficient %d.", i);

    // auto_ptr deletes its object when it exits its scope
    std::auto_ptr<ufc::finite_element> fe(a.ufc_form().create_finite_element(i + a.rank()));

    // Checks outcommented since they only work for Functions, not Expressions
    const uint r = coefficients[i]->value_rank();
    const uint fe_r = fe->value_rank();
    if (fe_r != r)
      error("Invalid value rank for coefficient %d, got %d but expecting %d. \
Did you forget to specify the value rank correctly in an Expression sub class?", i, r, fe_r);

    for (uint j = 0; j < r; ++j)
    {
      uint dim = coefficients[i]->value_dimension(j);
      uint fe_dim = fe->value_dimension(j);
      if (dim != fe_dim)
        error("Invalid value dimension %d for coefficient %d, got %d but expecting %d. \
Did you forget to specify the value dimension correctly in an Expression sub class?", j, i, dim, fe_dim);
    }
  }

  // Check that the cell dimension matches the mesh dimension
  if (a.ufc_form().rank() + a.ufc_form().num_coefficients() > 0)
  {
    ufc::finite_element* element = a.ufc_form().create_finite_element(0);
    assert(element);
    if (mesh.type().cell_type() == CellType::interval && element->cell_shape() != ufc::interval)
      error("Mesh cell type (intervals) does not match cell type of form.");
    if (mesh.type().cell_type() == CellType::triangle && element->cell_shape() != ufc::triangle)
      error("Mesh cell type (triangles) does not match cell type of form.");
    if (mesh.type().cell_type() == CellType::tetrahedron && element->cell_shape() != ufc::tetrahedron)
      error("Mesh cell type (tetrahedra) does not match cell type of form.");
    delete element;
  }

  // Check that the mesh is ordered
  if (!mesh.ordered())
    error("Unable to assemble, mesh is not correctly ordered (consider calling mesh.order()).");
}
//-----------------------------------------------------------------------------
void AssemblerTools::init_global_tensor(GenericTensor& A,
                                        const Form& a,
                                        UFC& ufc,
                                        bool reset_sparsity,
                                        bool add_values)
{
  // Check that we should not add values
  if (reset_sparsity && add_values)
    error("Can not add values when the sparsity pattern is reset");

  if (reset_sparsity)
  {
    // Build sparsity pattern
    Timer t0("Build sparsity");
    GenericSparsityPattern* sparsity_pattern = A.factory().create_pattern();
    if (sparsity_pattern)
    {
      // Get dof maps
      std::vector<const DofMap*> dof_maps(0);
      for (uint i = 0; i < ufc.form.rank(); ++i)
        dof_maps.push_back(&(a.function_space(i)->dofmap()));

      // Build sparsity pattern
      SparsityPatternBuilder::build(*sparsity_pattern, a.mesh(), dof_maps,
                                    a.ufc_form().num_cell_integrals(),
                                    a.ufc_form().num_interior_facet_integrals());
    }
    t0.stop();

    // Initialize tensor
    Timer t1("Init tensor");
    if (sparsity_pattern)
      A.init(*sparsity_pattern);
    else
    {
      A.resize(ufc.form.rank(), ufc.global_dimensions);
      A.zero();
    }
    t1.stop();

    // Delete sparsity pattern
    Timer t2("Delete sparsity");
    delete sparsity_pattern;
    t2.stop();
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
