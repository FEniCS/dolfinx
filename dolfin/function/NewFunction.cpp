// Copyright (C) 2003-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2005-2008.
// Modified by Martin Sandve Alnes, 2008.
//
// First added:  2003-11-28
// Last changed: 2008-10-12

#include <dolfin/log/log.h>
#include <dolfin/common/NoDeleter.h>
#include <dolfin/io/File.h>
#include <dolfin/la/GenericVector.h>
#include <dolfin/la/DefaultFactory.h>
#include <dolfin/fem/FiniteElement.h>
#include <dolfin/fem/DofMap.h>
#include "FunctionSpace.h"
#include "NewSubFunction.h"
#include "NewFunction.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
NewFunction::NewFunction(const FunctionSpace& V)
  : _function_space(&V, NoDeleter<const FunctionSpace>()), _vector(0),
    _time(0), _cell(0), _facet(-1)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
NewFunction::NewFunction(const std::tr1::shared_ptr<FunctionSpace> V)
  : _function_space(V), _vector(0),
    _time(0), _cell(0), _facet(-1)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
NewFunction::NewFunction(const std::string filename)
  : _function_space(static_cast<FunctionSpace*>(0)), _vector(0),
    _time(0), _cell(0), _facet(-1)
{
  // FIXME: Uncomment when renamed to Function
  //File file(filename);
  //file >> *this;
}
//-----------------------------------------------------------------------------
NewFunction::NewFunction(const NewSubFunction& v)
  : _function_space(static_cast<FunctionSpace*>(0)), _vector(0),
    _time(0), _cell(0), _facet(-1)
{
  *this = v;
}
//-----------------------------------------------------------------------------
NewFunction::NewFunction(const NewFunction& v)
  : _function_space(v._function_space), _vector(0),
    _time(0), _cell(0), _facet(-1)
{
  *this = v;
}
//-----------------------------------------------------------------------------
NewFunction::~NewFunction()
{
  delete _vector;
}
//-----------------------------------------------------------------------------
const NewFunction& NewFunction::operator= (const NewFunction& v)
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
const NewFunction& NewFunction::operator= (const NewSubFunction& v)
{
  // Check that vector exists
  //if (!v._vector)
  //  error("Unable to assign to function, missing coefficients (user-defined function).");

  /*
  // Create sub system
  SubSystem sub_system(sub_function.i);

  // Extract sub element
  std::tr1::shared_ptr<FiniteElement> _element(sub_system.extractFiniteElement(*(sub_function.f->finite_element)));
  finite_element.swap(_element);

  // Initialize scratch space
  scratch = new Scratch(*finite_element);

  // Extract sub dof map and offset
  uint offset = 0;
  std::tr1::shared_ptr<DofMap> _dof_map(sub_function.f->dof_map->extractDofMap(sub_system.array(), offset));
  dof_map.swap(_dof_map);

  // Create vector of dofs and copy values
  init();
  double* values   = new double[n];
  uint* get_rows = new uint[n];
  uint* set_rows = new uint[n];
  for (uint i = 0; i < n; i++)
  {
    get_rows[i] = offset + i;
    set_rows[i] = i;
  }
  sub_function.f->x->get(values, n, get_rows);
  x->set(values, n, set_rows);
  x->apply();

  delete [] values;
  delete [] get_rows;
  delete [] set_rows;
  */
  
  return *this;
}
//-----------------------------------------------------------------------------
NewSubFunction NewFunction::operator[] (uint i)
{
  // Check that vector exists
  if (!_vector)
    error("Unable to extract sub function, missing coefficients (user-defined function).");

  NewSubFunction sub_function(*this, i);
  return sub_function;
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
bool NewFunction::in(const FunctionSpace& V) const
{
  return &function_space() == &V;
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
void NewFunction::interpolate(GenericVector& coefficients, const FunctionSpace& V) const
{
  V.interpolate(coefficients, *this);
}
//-----------------------------------------------------------------------------
void NewFunction::interpolate(double* coefficients, const ufc::cell& ufc_cell) const
{
  dolfin_assert(coefficients);
  dolfin_assert(_function_space);
  _function_space->interpolate(coefficients, ufc_cell, *this);
}
//-----------------------------------------------------------------------------
void NewFunction::interpolate(double* vertex_values) const
{
  dolfin_assert(vertex_values);
  dolfin_assert(_function_space);
  _function_space->interpolate(vertex_values, *this);
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
