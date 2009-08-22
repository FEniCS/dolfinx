// Copyright (C) 2003-2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2005-2009.
// Modified by Martin Sandve Alnes, 2008.
//
// First added:  2003-11-28
// Last changed: 2009-08-17

#include <algorithm>
#include <boost/assign/list_of.hpp>
#include <dolfin/log/log.h>
#include <dolfin/common/NoDeleter.h>
#include <dolfin/io/File.h>
#include <dolfin/la/GenericVector.h>
#include <dolfin/la/DefaultFactory.h>
#include <dolfin/fem/FiniteElement.h>
#include <dolfin/fem/DofMap.h>
#include <dolfin/fem/UFC.h>
#include "Data.h"
#include "UFCFunction.h"
#include "FunctionSpace.h"
#include "Function.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
Function::Function()
  :  Variable("v", "unnamed function"),
     _function_space(static_cast<FunctionSpace*>(0)),
     _vector(static_cast<GenericVector*>(0)),
     _off_process_vector(static_cast<GenericVector*>(0))
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Function::Function(const FunctionSpace& V)
  : Variable("v", "unnamed function"),
    _function_space(reference_to_no_delete_pointer(V)),
    _vector(static_cast<GenericVector*>(0)),
     _off_process_vector(static_cast<GenericVector*>(0))
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Function::Function(boost::shared_ptr<const FunctionSpace> V)
  : Variable("v", "unnamed function"),
    _function_space(V),
    _vector(static_cast<GenericVector*>(0)),
     _off_process_vector(static_cast<GenericVector*>(0))
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Function::Function(boost::shared_ptr<const FunctionSpace> V,
                   GenericVector& x)
  : Variable("v", "unnamed function"),
    _function_space(V),
    _vector(reference_to_no_delete_pointer(x)),
    _off_process_vector(static_cast<GenericVector*>(0))
{
  assert(V->dofmap().global_dimension() == x.size());
}
//-----------------------------------------------------------------------------
Function::Function(boost::shared_ptr<const FunctionSpace> V,
                   boost::shared_ptr<GenericVector> x)
  : Variable("v", "unnamed function"),
    _function_space(V),
    _vector(x),
    _off_process_vector(static_cast<GenericVector*>(0))
{
  assert(V->dofmap().global_dimension() <= x->size());
}
//-----------------------------------------------------------------------------
Function::Function(const FunctionSpace& V, GenericVector& x)
  : Variable("v", "unnamed function"),
    _function_space(reference_to_no_delete_pointer(V)),
    _vector(reference_to_no_delete_pointer(x)),
    _off_process_vector(static_cast<GenericVector*>(0))
{
  assert(V.dofmap().global_dimension() == x.size());
}
//-----------------------------------------------------------------------------
Function::Function(const FunctionSpace& V, std::string filename)
  : Variable("v", "unnamed function"),
    _function_space(reference_to_no_delete_pointer(V)),
    _vector(static_cast<GenericVector*>(0)),
    _off_process_vector(static_cast<GenericVector*>(0))
{
  // Initialize vector
  init();

  // Read vector from file
  File file(filename);
  file >> *_vector;

  // Check size of vector
  if (_vector->size() != _function_space->dim())
    error("Unable to read Function from file, number of degrees of freedom (%d) does not match dimension of function space (%d).", _vector->size(), _function_space->dim());
}
//-----------------------------------------------------------------------------
Function::Function(boost::shared_ptr<const FunctionSpace> V, std::string filename)
  : Variable("v", "unnamed function"),
    _function_space(V),
    _vector(static_cast<GenericVector*>(0)),
    _off_process_vector(static_cast<GenericVector*>(0))
{
  // Initialize vector
  init();

  // Read vector from file
  File file(filename);
  file >> *_vector;

  // Check size of vector
  if (_vector->size() != _function_space->dim())
    error("Unable to read Function from file, number of degrees of freedom (%d) does not match dimension of function space (%d).", _vector->size(), _function_space->dim());
}
//-----------------------------------------------------------------------------
Function::Function(const Function& v)
  : Variable("v", "unnamed function"),
    _function_space(static_cast<FunctionSpace*>(0)),
    _vector(static_cast<GenericVector*>(0)),
    _off_process_vector(static_cast<GenericVector*>(0))
{
  *this = v;
}
//-----------------------------------------------------------------------------
Function::~Function()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
const Function& Function::operator= (const Function& v)
{
  // Check for function space and vector
  if (!v.has_function_space())
    error("Cannot copy Functions which do not have a FunctionSpace.");

  cout << "Copy functions space in assignment" << endl;
  // Copy function space
  _function_space = v._function_space;

  // Initialize vector
  init();

  // FIXME: For hard copies of subfunctions, we need to implement extraction 
  // and 'reset' of dof maps, and copying of parts of the vector.
  bool discrete_subfunction = false;
  if (v.has_vector())
  {
    if (v._vector->size() != v._function_space->dofmap().global_dimension())
    {
      warning("Proper copying of sub-Functions not yet implemented. You will receive deep copy, but with a copy of the longer vector of the original function.");
      _vector->resize(v._vector->size());
      discrete_subfunction = true;
    }
  }

  // Copy values or interpolate
  if (v.has_vector())
  {
    assert(_vector->size() == v._vector->size());
    *_vector = *v._vector;
  }
  else
  {
    info("Assignment from user-defined function, interpolating.");
    function_space().interpolate(*_vector, v);
  }

  return *this;
}
//-----------------------------------------------------------------------------
Function& Function::operator[] (uint i)
{
  // Check that vector exists
  if (!_vector)
    error("Unable to extract sub function, missing coefficients (user-defined function).");

  // Check if sub-Function is in the cache, otherwise create and add to cache
  boost::ptr_map<uint, Function>::iterator sub_function = sub_functions.find(i);
  if (sub_function != sub_functions.end())
    return *(sub_function->second);
  else
  {
    // Extract function subspace
    std::vector<uint> component = boost::assign::list_of(i);
    boost::shared_ptr<const FunctionSpace> sub_space(this->function_space().extract_sub_space(component));

    // Insert sub-Function into map and return reference
    sub_functions.insert(i, new Function(sub_space, this->_vector));
    return *(sub_functions.find(i)->second);
  }
}
//-----------------------------------------------------------------------------
const FunctionSpace& Function::function_space() const
{
  assert(_function_space);
  return *_function_space;
}
//-----------------------------------------------------------------------------
boost::shared_ptr<const FunctionSpace> Function::function_space_ptr() const
{
  return _function_space;
}
//-----------------------------------------------------------------------------
GenericVector& Function::vector()
{
  // Initialize vector of dofs if not initialized
  if (!_vector)
    init();

  assert(_vector);

  if (_vector->size() != _function_space->dofmap().global_dimension())
    warning("You are extracting the vector from a sub-Function. You will receive the longer vector of the original function.");

  return *_vector;
}
//-----------------------------------------------------------------------------
const GenericVector& Function::vector() const
{
  // Check if vector of dofs has been initialized
  if (!_vector)
    error("Requesting vector of degrees of freedom for function, but vector has not been initialized.");

  assert(_vector);
  return *_vector;
}
//-----------------------------------------------------------------------------
bool Function::has_function_space() const
{
  return _function_space.get();
}
//-----------------------------------------------------------------------------
bool Function::has_vector() const
{
  return _vector.get();
}
//-----------------------------------------------------------------------------
bool Function::in(const FunctionSpace& V) const
{
  // Function is in any space if V is not defined
  return !_function_space || _function_space.get() == &V;
}
//-----------------------------------------------------------------------------
dolfin::uint Function::geometric_dimension() const
{
  assert(_function_space);
  return _function_space->mesh().geometry().dim();
}
//-----------------------------------------------------------------------------
void Function::eval(double* values, const double* x) const
{
  assert(values);
  assert(x);

  // Use vector of dofs if available
  if (_vector)
  {
    assert(_function_space);
    _function_space->eval(values, x, *this);
    return;
  }

  // Missing eval() function if we get here
  error("Missing eval() for user-defined function (must be overloaded).");
}
//-----------------------------------------------------------------------------
void Function::eval(double* values, const Data& data) const
{
  assert(values);
  assert(data.x);

  // Use vector of dofs if available
  if (_vector)
  {
    assert(_function_space);
    _function_space->eval(values, data.x, *this);
    return;
  }

  // Try simple eval function
  eval(values, data.x);
}
//-----------------------------------------------------------------------------
void Function::eval(double* values, const double* x, const ufc::cell& ufc_cell, 
                    uint cell_index) const
{
  assert(_function_space);
  _function_space->eval(values, x, *this, ufc_cell, cell_index);
}
//-----------------------------------------------------------------------------
void Function::interpolate(double* coefficients,
                           const ufc::cell& ufc_cell,
                           uint cell_index,
                           int local_facet) const
{
  assert(coefficients);
  assert(_function_space);
  interpolate(coefficients, *_function_space, ufc_cell, cell_index, local_facet);
}
//-----------------------------------------------------------------------------
void Function::interpolate(double* coefficients,
                           const FunctionSpace& V,
                           const ufc::cell& ufc_cell,
                           uint cell_index,
                           int local_facet) const
{
  assert(coefficients);

  // Either pick values or evaluate dof functionals
  if (in(V) && _vector)
  {
    // Get dofmap
    const DofMap& dofmap = V.dofmap();

    // Tabulate dofs
    uint* dofs = new uint[dofmap.local_dimension(ufc_cell)];
    dofmap.tabulate_dofs(dofs, ufc_cell, cell_index);

    // Pick values from vector(s)
    get(coefficients, dofmap.local_dimension(ufc_cell), dofs);

    // Clean up
    delete [] dofs;
  }
  else
  {
    // Create data
    const Cell cell(V.mesh(), cell_index);
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
void Function::interpolate(const Function& v)
{
  assert(has_function_space());
  assert(v.has_function_space());
  function_space().interpolate(this->vector(), v, "non-matching");
}
//-----------------------------------------------------------------------------
void Function::interpolate_vertex_values(double* vertex_values) const
{
  assert(vertex_values);
  assert(_function_space);
  _function_space->interpolate_vertex_values(vertex_values, *this);
}
//-----------------------------------------------------------------------------
void Function::interpolate()
{
  // Check that function is not already discrete
  if (has_vector())
    error("Unable to interpolate function, already interpolated (has a vector).");

  // Check that we have a function space
  if (!has_function_space())
    error("Unable to interpolate function, missing function space.");

  // Interpolate to vector
  DefaultFactory factory;
  boost::shared_ptr<GenericVector> coefficients(factory.create_vector());
  function_space().interpolate(*coefficients, *this);

  // Set values
  init();
  *_vector = *coefficients;
}
//-----------------------------------------------------------------------------
void Function::compute_off_process_dofs() const
{
  // Clear data
  _off_process_dofs.clear();
  global_to_local.clear();

  // Get mesh
  assert(_function_space);
  const Mesh& mesh = _function_space->mesh();

  // Storage for each cell dofs
  const DofMap& dofmap = _function_space->dofmap();
  const uint num_dofs_per_cell = _function_space->element().space_dimension();
  const uint num_dofs_global = vector().size();
  uint* dofs = new uint[num_dofs_per_cell];

  // Iterate over mesh and check which dofs are needed
  UFCCell ufc_cell(mesh);
  uint i = 0;
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    // Update to current cell
    ufc_cell.update(*cell);

    // Tabulate dofs on cell
    dofmap.tabulate_dofs(dofs, ufc_cell, cell->index());

    for (uint d = 0; d < num_dofs_per_cell; ++d)
    {
      const uint dof = dofs[d];
      const uint index_owner = MPI::index_owner(dof, num_dofs_global);
      if (index_owner != MPI::process_number())
      {
        if (std::find(_off_process_dofs.begin(), _off_process_dofs.end(), dof) == _off_process_dofs.end())
        {
          _off_process_dofs.push_back(dof);
          global_to_local[dof] = i++;
        }
      }
    }
  }

  delete [] dofs;
}
//-----------------------------------------------------------------------------
void Function::init()
{
  // Get size
  assert(_function_space);
  const uint N = _function_space->dofmap().global_dimension();

  // Create vector of dofs
  if (!_vector)
  {
    DefaultFactory factory;
    _vector.reset(factory.create_vector());
  }

  // Initialize vector of dofs
  assert(_vector);
  _vector->resize(N);
  _vector->zero();
}
//-----------------------------------------------------------------------------
void Function::get(double* block, uint m, const uint* rows) const
{
  // Get local ownership range
  const std::pair<uint, uint> range = _vector->local_range();

  if (range.first == 0 && range.second == _vector->size())
    _vector->get(block, m, rows);
  else
  {
    if (!_off_process_vector.get())
      error("Function has not been prepared with off-process data. Did you forget to call Function::gather()?");

    // FIXME: Perform some more sanity checks

    // Build lists of local and nonlocal coefficients
    uint n_local = 0;
    uint n_nonlocal = 0;
    for (uint i = 0; i < m; ++i)
    {
      if (rows[i] >= range.first && rows[i] < range.second)
      {
        scratch.local_index[n_local]  = i;
        scratch.local_rows[n_local++] = rows[i];
     }
      else
      {
        scratch.nonlocal_index[n_nonlocal]  = i;
        scratch.nonlocal_rows[n_nonlocal++] = global_to_local[rows[i]];
      }
    }

    // Get local coefficients
    _vector->get_local(scratch.local_block, n_local, scratch.local_rows);

    // Get off process coefficients
    _off_process_vector->get_local(scratch.nonlocal_block, n_nonlocal, scratch.nonlocal_rows);

    // Copy result into block
    for (uint i = 0; i < n_local; ++i)
      block[scratch.local_index[i]] = scratch.local_block[i];
    for (uint i = 0; i < n_nonlocal; ++i)
      block[scratch.nonlocal_index[i]] = scratch.nonlocal_block[i];
  }
}
//-----------------------------------------------------------------------------
void Function::gather() const
{
  // Gather off-process coefficients if running in parallel and function has a vector
  if (MPI::num_processes() > 1 && has_vector())
  {
    assert(_function_space);

    // Initialise scratch space
    scratch.init(_function_space->dofmap().max_local_dimension());

    // Compute lists of off-process dofs
    compute_off_process_dofs();

    // Create off process vector if it doesn't exist
    if (!_off_process_vector.get())
      _off_process_vector.reset(_vector->factory().create_local_vector());

    // Gather off process coefficients
    _vector->gather(*_off_process_vector, _off_process_dofs);
  }
}
//-----------------------------------------------------------------------------
