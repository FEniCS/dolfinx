// Copyright (C) 2003-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2005-2008.
// Modified by Martin Sandve Alnes, 2008.
//
// First added:  2003-11-28
// Last changed: 2008-09-11

#include <dolfin/log/log.h>
#include <dolfin/common/NoDeleter.h>
#include <dolfin/la/GenericVector.h>
#include <dolfin/la/DefaultFactory.h>
#include <dolfin/fem/FiniteElement.h>
#include <dolfin/fem/DofMap.h>
#include "FunctionSpace.h"
#include "NewFunction.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
NewFunction::NewFunction(const FunctionSpace& V)
  : _function_space(&V, NoDeleter<const FunctionSpace>()), _vector(0),
    _cell(0), _facet(-1), _time(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
NewFunction::NewFunction(const std::tr1::shared_ptr<FunctionSpace> V)
  : _function_space(V), _vector(0),
    _cell(0), _facet(-1), _time(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
NewFunction::NewFunction(const std::string filename)
  : _function_space(static_cast<FunctionSpace*>(0)), _vector(0),
    _cell(0), _facet(-1), _time(0)
{
  error("Not implemented.");
}
//-----------------------------------------------------------------------------
NewFunction::NewFunction(const NewFunction& v)
  : _function_space(static_cast<FunctionSpace*>(0)), _vector(0),
    _cell(0), _facet(-1), _time(0)
{
  error("Not implemented.");
}
//-----------------------------------------------------------------------------
NewFunction::~NewFunction()
{
  delete _vector;
}
//-----------------------------------------------------------------------------
const NewFunction& NewFunction::operator= (const NewFunction& v)
{
  // FIXME: Need to check pointers here and check if _U is nonzero
  //*_V = *v._V;
  //*_U = *u._V;
  
  return *this;
}
//-----------------------------------------------------------------------------
const FunctionSpace& NewFunction::function_space() const
{
  dolfin_assert(_function_space);
  return *_function_space;
}
//-----------------------------------------------------------------------------
GenericVector& NewFunction::vector()
{
  // Initialize vector of dofs if not initialized
  if (!_vector)
    init();

  dolfin_assert(_vector);
  return *_vector;
}
//-----------------------------------------------------------------------------
const GenericVector& NewFunction::vector() const
{
  // Check if vector of dofs has been initialized
  if (!_vector)
    error("Requesting vector of degrees of freedom for function, but vector has not been initialized.");
  
  dolfin_assert(_vector);
  return *_vector;
}
//-----------------------------------------------------------------------------
double NewFunction::time() const
{
  return _time;
}
//-----------------------------------------------------------------------------
void NewFunction::eval(double* values, const double* x) const
{
  dolfin_assert(values);
  dolfin_assert(x);
  dolfin_assert(_function_space);

  // Use vector of dofs if available
  if (_vector)
  {
    _function_space->eval(values, x, *_vector);
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
void NewFunction::eval(double* values, const double* x, double t) const
{
  // Missing eval() function if we get here
  error("Missing eval() for user-defined function (must be overloaded).");
}
//-----------------------------------------------------------------------------
double NewFunction::eval(const double* x) const
{
  // Use time-dependent eval if available
  return eval(x, _time);
}
//-----------------------------------------------------------------------------
double NewFunction::eval(const double* x, double t) const
{
  // Missing eval() function if we get here
  error("Missing eval() for scalar user-defined function (must be overloaded).");
  return 0.0;
}
//-----------------------------------------------------------------------------
void NewFunction::eval(simple_array<double>& values, const simple_array<double>& x) const
{
  eval(values.data, x.data);
}
//-----------------------------------------------------------------------------
const Cell& NewFunction::cell() const
{
  if (!_cell)
    error("Current cell is unknown.");

  return *_cell;
}
//-----------------------------------------------------------------------------
dolfin::uint NewFunction::facet() const
{
  if (_facet < 0)
    error("Current facet is unknown.");

  return static_cast<uint>(_facet);
}
//-----------------------------------------------------------------------------
Point NewFunction::normal() const
{
  return cell().normal(facet());
}
//-----------------------------------------------------------------------------
void NewFunction::init()
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
