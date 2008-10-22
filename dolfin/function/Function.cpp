// Copyright (C) 2003-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2005-2008.
// Modified by Martin Sandve Alnes, 2008.
//
// First added:  2003-11-28
// Last changed: 2008-10-14

#include <dolfin/log/log.h>
#include <dolfin/common/NoDeleter.h>
#include <dolfin/io/File.h>
#include <dolfin/la/GenericVector.h>
#include <dolfin/la/DefaultFactory.h>
#include <dolfin/fem/FiniteElement.h>
#include <dolfin/fem/DofMap.h>
#include "FunctionSpace.h"
#include "SubFunction.h"
#include "Function.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
Function::Function(const FunctionSpace& V)
  : _function_space(&V, NoDeleter<const FunctionSpace>()),
    _vector(0),
    _time(0), _cell(0), _facet(-1)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Function::Function(const std::tr1::shared_ptr<FunctionSpace> V)
  : _function_space(V),
    _vector(0),
    _time(0), _cell(0), _facet(-1)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Function::Function(const std::string filename)
  : _function_space(static_cast<FunctionSpace*>(0)),
    _vector(0),
    _time(0), _cell(0), _facet(-1)
{
  File file(filename);
  file >> *this;
}
//-----------------------------------------------------------------------------
Function::Function(const SubFunction& v)
  : _function_space(v.v.function_space().extract_sub_space(Array<uint>(v.i))),
    _vector(0),
    _time(0), _cell(0), _facet(-1)
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
  : _function_space(v._function_space), _vector(0),
    _time(0), _cell(0), _facet(-1)
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
  // Note 3: don't assign cell and facet

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

  // Assign time
  _time = v._time;

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
const FiniteElement& Function::element() const
{
  dolfin_assert(_function_space);
  return _function_space->element();
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
double Function::time() const
{
  return _time;
}
//-----------------------------------------------------------------------------
bool Function::in(const FunctionSpace& V) const
{
  return &function_space() == &V;
}
//-----------------------------------------------------------------------------
void Function::eval(double* values, const double* x) const
{
  dolfin_assert(values);
  dolfin_assert(x);
  dolfin_assert(_function_space);

  // Use vector of dofs if available
  if (_vector)
  {
    _function_space->eval(values, x, *this);
    return;
  }

  // Use scalar eval() if available
  if (_function_space->element().value_rank() == 0)
  {
    values[0] = eval(x);
    return;
  }

  // Use time-dependent eval if available
  eval(values, x, _time);
}
//-----------------------------------------------------------------------------
void Function::eval(double* values, const double* x, double t) const
{
  // Missing eval() function if we get here
  error("Missing eval() for user-defined function (must be overloaded).");
}
//-----------------------------------------------------------------------------
double Function::eval(const double* x) const
{
  // Use time-dependent eval if available
  return eval(x, _time);
}
//-----------------------------------------------------------------------------
double Function::eval(const double* x, double t) const
{
  // Missing eval() function if we get here
  error("Missing eval() for scalar user-defined function (must be overloaded).");
  return 0.0;
}
//-----------------------------------------------------------------------------
void Function::eval(simple_array<double>& values, const simple_array<double>& x) const
{
  eval(values.data, x.data);
}
//-----------------------------------------------------------------------------
void Function::evaluate(double* values, const double* coordinates, const ufc::cell& cell) const
{
  error("Not implemented, need to think about how to handle it when not user-defined.");
}
//-----------------------------------------------------------------------------
void Function::interpolate(GenericVector& coefficients, const FunctionSpace& V) const
{
  V.interpolate(coefficients, *this);
}
//-----------------------------------------------------------------------------
void Function::interpolate(double* coefficients, const ufc::cell& ufc_cell) const
{
  dolfin_assert(coefficients);
  dolfin_assert(_function_space);

  // Either pick values or evaluate dof functionals
  if (_vector)
  {
    // Get dofmap
    const DofMap& dofmap = _function_space->dofmap();

    // Tabulate dofs
    uint* dofs = new uint[dofmap.local_dimension()];
    dofmap.tabulate_dofs(dofs, ufc_cell);
    
    // Pick values from global vector
    _vector->get(coefficients, dofmap.local_dimension(), dofs);
    delete [] dofs;
  }
  else
  {
    // Get element
    const FiniteElement& element = _function_space->element();

    // Evaluate each dof to get coefficients for nodal basis expansion
    for (uint i = 0; i < element.space_dimension(); i++)
      coefficients[i] = element.evaluate_dof(i, *this, ufc_cell);
  }
}
//-----------------------------------------------------------------------------
void Function::interpolate(double* coefficients, const FunctionSpace& V,
                           const ufc::cell& ufc_cell) const
{
  dolfin_assert(coefficients);
  dolfin_assert(_function_space);
  
  error("Function::interpolate not yet programmed.");
}
//-----------------------------------------------------------------------------
void Function::interpolate(double* vertex_values) const
{
  dolfin_assert(vertex_values);
  dolfin_assert(_function_space);
  _function_space->interpolate(vertex_values, *this);
}
//-----------------------------------------------------------------------------
const Cell& Function::cell() const
{
  if (!_cell)
    error("Current cell is unknown.");

  return *_cell;
}
//-----------------------------------------------------------------------------
dolfin::uint Function::facet() const
{
  if (_facet < 0)
    error("Current facet is unknown.");

  return static_cast<uint>(_facet);
}
//-----------------------------------------------------------------------------
Point Function::normal() const
{
  return cell().normal(facet());
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
