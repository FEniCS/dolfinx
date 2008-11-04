// Copyright (C) 2003-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2005-2008.
// Modified by Martin Sandve Alnes, 2008.
//
// First added:  2003-11-28
// Last changed: 2008-11-04

#include <dolfin/log/log.h>
#include <dolfin/common/Array.h>
#include <dolfin/common/NoDeleter.h>
#include <dolfin/io/File.h>
#include <dolfin/la/GenericVector.h>
#include <dolfin/la/DefaultFactory.h>
#include <dolfin/fem/FiniteElement.h>
#include <dolfin/fem/DofMap.h>
#include "Data.h"
#include "UFCFunction.h"
#include "FunctionSpace.h"
#include "SubFunction.h"
#include "Function.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
Function::Function()
  : _function_space(static_cast<FunctionSpace*>(0)),
    _vector(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Function::Function(const FunctionSpace& V)
  : _function_space(&V, NoDeleter<const FunctionSpace>()),
    _vector(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Function::Function(std::tr1::shared_ptr<const FunctionSpace> V)
  : _function_space(V),
    _vector(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Function::Function(std::string filename)
  : _function_space(static_cast<FunctionSpace*>(0)),
    _vector(0)
{
  File file(filename);
  file >> *this;
}
//-----------------------------------------------------------------------------
Function::Function(const SubFunction& v)
  : _function_space(v.v.function_space().extract_sub_space(v.component)),
    _vector(0)
{
  // Initialize vector
  init();

  // Copy subset of coefficients
  const uint n = _vector->size();
  const uint offset = function_space().dofmap().offset();
  uint* rows = new uint[n];
  double* values = new double[n];
  for (uint i = 0; i < n; i++)
    rows[i] = offset + i;
  v.v.vector().get(values, n, rows);
  _vector->set(values);
  _vector->apply();
  
  // Clean up
  delete [] rows;
  delete [] values;
}
//-----------------------------------------------------------------------------
Function::Function(const Function& v)
  : _function_space(v._function_space),
    _vector(0)
{
  *this = v;
}
//-----------------------------------------------------------------------------
Function::~Function()
{
  delete _vector;
}
//-----------------------------------------------------------------------------
const Function& Function::operator= (const Function& v)
{
  // Note 1: function spaces must be the same
  // Note 2: vector needs special handling

  // Check that function has a function spaces
  if (!has_function_space())
    error("Unable to assign to function, function does not have a function space.");

  // Check that function spaces are the same
  if (!v.in(function_space()))
    error("Unable to assign to function, not in the same function space.");

  // Check that vector exists
  if (!v._vector)
    error("Unable to assign to function, missing coefficients (user-defined function).");
  
  // Assign vector
  init();
  dolfin_assert(_vector);
  *_vector = *v._vector;

  return *this;
}
//-----------------------------------------------------------------------------
SubFunction Function::operator[] (uint i)
{
  // Check that vector exists
  if (!_vector)
    error("Unable to extract sub function, missing coefficients (user-defined function).");

  SubFunction sub_function(*this, i);
  return sub_function;
}
//-----------------------------------------------------------------------------
const FunctionSpace& Function::function_space() const
{
  dolfin_assert(_function_space);
  return *_function_space;
}
//-----------------------------------------------------------------------------
GenericVector& Function::vector()
{
  // Initialize vector of dofs if not initialized
  if (!_vector)
    init();

  dolfin_assert(_vector);
  return *_vector;
}
//-----------------------------------------------------------------------------
const GenericVector& Function::vector() const
{
  // Check if vector of dofs has been initialized
  if (!_vector)
    error("Requesting vector of degrees of freedom for function, but vector has not been initialized.");

  dolfin_assert(_vector);
  return *_vector;
}
//-----------------------------------------------------------------------------
bool Function::has_function_space() const
{
  return _function_space.get();
}
//-----------------------------------------------------------------------------
bool Function::in(const FunctionSpace& V) const
{
  // Function is in any space if V is not defined
  return !_function_space || _function_space.get() == &V;
}
//-----------------------------------------------------------------------------
void Function::eval(double* values, const Data& data) const
{
  dolfin_assert(values);
  dolfin_assert(data.x);

  // Use vector of dofs if available
  if (_vector)
  {
    dolfin_assert(_function_space);
    _function_space->eval(values, data.x, *this);
    return;
  }

  // Missing eval() function if we get here
  error("Missing eval() for user-defined function (must be overloaded).");
}
//-----------------------------------------------------------------------------
void Function::interpolate(double* coefficients,
                           const ufc::cell& ufc_cell,
                           int local_facet) const
{
  dolfin_assert(coefficients);
  dolfin_assert(_function_space);
  interpolate(coefficients, *_function_space, ufc_cell, local_facet);
}
//-----------------------------------------------------------------------------
void Function::interpolate(double* coefficients,
                           const FunctionSpace& V,
                           const ufc::cell& ufc_cell,
                           int local_facet) const
{
  dolfin_assert(coefficients);

  // Check that function space matches
  if (!in(V))
    error("Unable to interpolate function, incorrect function space.");

  // Either pick values or evaluate dof functionals
  if (_vector)
  {
    // Get dofmap
    const DofMap& dofmap = V.dofmap();

    // Tabulate dofs
    uint* dofs = new uint[dofmap.local_dimension()];
    dofmap.tabulate_dofs(dofs, ufc_cell);
    
    // Pick values from global vector
    _vector->get(coefficients, dofmap.local_dimension(), dofs);
    delete [] dofs;
  }
  else
  {
    // Create data
    const Cell cell(V.mesh(), ufc_cell.entity_indices[V.mesh().topology().dim()][0]);
    Data data(cell, local_facet);

    // Create UFC wrapper for this function
    UFCFunction v(*this, data);

    // Get element
    const FiniteElement& element = V.element();

    // Evaluate each dof to get coefficients for nodal basis expansion
    for (uint i = 0; i < element.space_dimension(); i++)
      coefficients[i] = element.evaluate_dof(i, v, ufc_cell);
  }
}
//-----------------------------------------------------------------------------
void Function::interpolate(GenericVector& coefficients,
                           const FunctionSpace& V) const
{
  V.interpolate(coefficients, *this);
}
//-----------------------------------------------------------------------------
void Function::interpolate(double* vertex_values) const
{
  dolfin_assert(vertex_values);
  dolfin_assert(_function_space);
  _function_space->interpolate(vertex_values, *this);
}
//-----------------------------------------------------------------------------
void Function::init()
{
  // Get size
  dolfin_assert(_function_space);
  const uint N = _function_space->dofmap().global_dimension();

  // Create vector of dofs
  if (!_vector)
  {
    DefaultFactory factory;
    _vector = factory.create_vector();
  }

  // Initialize vector of dofs
  dolfin_assert(_vector);
  _vector->resize(N);
  _vector->zero();
}
//-----------------------------------------------------------------------------
