// Copyright (C) 2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells 2005-2007.
//
// First added:  2003-11-28
// Last changed: 2007-08-27
//
// The class Function serves as the envelope class and holds a pointer
// to a letter class that is a subclass of GenericFunction. All the
// functionality is handled by the specific implementation (subclass).

#include <dolfin/File.h>
#include <dolfin/UserFunction.h>
#include <dolfin/ConstantFunction.h>
#include <dolfin/DiscreteFunction.h>
#include <dolfin/Function.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Function::Function()
  : Variable("u", "empty function"),
    f(0), _type(empty), _cell(0), _facet(-1)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Function::Function(Mesh& mesh)
  : Variable("u", "user-defined function"),
    f(0), _type(user), _cell(0), _facet(-1)
{
  f = new UserFunction(mesh, this);
}
//-----------------------------------------------------------------------------
Function::Function(Mesh& mesh, real value)
  : Variable("u", "constant function"),
    f(0), _type(constant), _cell(0), _facet(-1)
{
  f = new ConstantFunction(mesh, value);
}
//-----------------------------------------------------------------------------
Function::Function(Mesh& mesh, Vector& x, Form& form, uint i)
  : Variable("u", "discrete function"),
    f(0), _type(discrete), _cell(0), _facet(-1)
{
  f = new DiscreteFunction(mesh, x, form, i);
}
//-----------------------------------------------------------------------------
Function::Function(Mesh& mesh, DofMap& dof_map, Vector& x, const ufc::form& form, uint i)
  : Variable("u", "discrete function"),
    f(0), _type(discrete), _cell(0), _facet(-1)
{
  f = new DiscreteFunction(mesh, dof_map, x, form, i);
}
//-----------------------------------------------------------------------------
Function::Function(const std::string filename)
  : Variable("u", "discrete function from data file"),
    f(0), _type(empty), _cell(0), _facet(-1)
{
  File file(filename);
  file >> *this;
}
//-----------------------------------------------------------------------------
Function::Function(SubFunction f)
  : Variable("u", "discrete function"),
    f(0), _type(discrete), _cell(0), _facet(-1)
{
  cout << "Extracting sub function." << endl;
  this->f = new DiscreteFunction(f);
}
//-----------------------------------------------------------------------------
Function::Function(const Function& f) : f(0), _type(f.type()), _cell(0), _facet(-1)
{
  if( f.type() == discrete ) 
  {
    this->f = new DiscreteFunction(*static_cast<DiscreteFunction*>(f.f));
    rename("x", "discrete function");
  }
  else if( f.type() == constant )
  {
    this->f = new ConstantFunction(*static_cast<ConstantFunction*>(f.f));
    rename("x", "constant function");
  }
  else
    error("Copy constructor works for discrete and constant functions only (so far).");
}
//-----------------------------------------------------------------------------
Function::~Function()
{
  if (f)
    delete f;
}
//-----------------------------------------------------------------------------
void Function::init(Mesh& mesh, Vector& x, Form& form, uint i)
{
  if (f)
    delete f;

  f = new DiscreteFunction(mesh, x, form, i);
  
  rename("u", "discrete function");
  _type = discrete;
}
//-----------------------------------------------------------------------------
Function::Type Function::type() const
{
  return _type;
}
//-----------------------------------------------------------------------------
dolfin::uint Function::rank() const
{
  if (!f)
    error("Function contains no data.");

  return f->rank();
}
//-----------------------------------------------------------------------------
dolfin::uint Function::dim(unsigned int i) const
{
  if (!f)
    error("Function contains no data.");

  return f->dim(i);
}
//-----------------------------------------------------------------------------
Mesh& Function::mesh() const
{
  if (!f)
    error("Function contains no data.");

  return f->mesh;
}
//-----------------------------------------------------------------------------
Vector& Function::vector() const
{
  if (!f)
    error("Function contains no data.");

  if (_type != discrete)
    error("A vector can only be extracted from discrete functions.");

  return (static_cast<DiscreteFunction*>(f))->vector();
}
//-----------------------------------------------------------------------------
dolfin::uint Function::numSubFunctions() const
{
  if (_type != discrete)
    error("Only discrete functions have sub functions.");

  return static_cast<DiscreteFunction*>(f)->numSubFunctions();
}
//-----------------------------------------------------------------------------
SubFunction Function::operator[] (uint i)
{
  if (_type != discrete)
    error("Sub functions can only be extracted from discrete functions.");

  SubFunction sub_function(static_cast<DiscreteFunction*>(f), i);
  return sub_function;
}
//-----------------------------------------------------------------------------
const Function& Function::operator= (Function& f)
{
  // FIXME: Handle other assignments
  if (f._type != discrete)
    error("Can only handle assignment from discrete functions (for now).");
  
  // Either create or copy discrete function
  if (_type == discrete)
  {
    *static_cast<DiscreteFunction*>(this->f) = *static_cast<DiscreteFunction*>(f.f);
  }
  else
  {
    delete this->f;
    this->f = new DiscreteFunction(*static_cast<DiscreteFunction*>(f.f));
    _type = discrete;
    rename(name(), "discrete function");
  }
  return *this;
}
//-----------------------------------------------------------------------------
const Function& Function::operator= (SubFunction sub_function)
{
  if (f)
    delete f;

  f = new DiscreteFunction(sub_function);
  
  rename("u", "discrete function");
  _type = discrete;

  return *this;
}
//-----------------------------------------------------------------------------
void Function::interpolate(real* values)
{
  if (!f)
    error("Function contains no data.");

  f->interpolate(values);
}
//-----------------------------------------------------------------------------
void Function::interpolate(real* coefficients,
                           const ufc::cell& ufc_cell,
                           const ufc::finite_element& finite_element,
                           Cell& cell, int facet)
{
  if (!f)
    error("Function contains no data.");

  // Make current cell and facet are available to user-defined function
  _cell = &cell;
  _facet = facet;

  // Interpolate function
  f->interpolate(coefficients, ufc_cell, finite_element);

  // Make cell and facet unavailable
  _cell = 0;
  _facet = -1;
}
//-----------------------------------------------------------------------------
void Function::eval(real* values, const real* x) const
{
  // Try scalar function if not overloaded
  values[0] = eval(x);
}
//-----------------------------------------------------------------------------
dolfin::real Function::eval(const real* x) const
{
  error("Missing eval() for user-defined function (must be overloaded).");
  return 0.0;
}
//-----------------------------------------------------------------------------
const Cell& Function::cell() const
{
   if (!_cell)
    error("Current cell is unknown (only available during assembly).");
  
  return *_cell;
}
//-----------------------------------------------------------------------------
int Function::facet() const
{
  return _facet;
}
//-----------------------------------------------------------------------------
