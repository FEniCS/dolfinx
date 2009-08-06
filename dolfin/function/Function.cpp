// Copyright (C) 2003-2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2005-2009.
// Modified by Martin Sandve Alnes, 2008.
//
// First added:  2003-11-28
// Last changed: 2009-06-22

#include <algorithm>
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
#include "SubFunction.h"
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
  assert(V->dofmap().global_dimension() == x->size());
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
Function::Function(const SubFunction& v)
  : Variable("v", "unnamed function"),
    _function_space(v.v.function_space().extract_sub_space(v.component)),
    _vector(static_cast<GenericVector*>(0)),
    _off_process_vector(static_cast<GenericVector*>(0))
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

  // Copy function space
  _function_space = v._function_space;

  // Initialize vector and copy values
  init();

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
SubFunction Function::operator[] (uint i) const
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
void Function::eval(double* values, const double* x, const ufc::cell& ufc_cell, uint cell_index) const
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
 
    /*
    if (MPI::num_processes() == 1)
    {
      // Pick values from global vector
      _vector->get(coefficients, dofmap.local_dimension(ufc_cell), dofs);
    }
    else
    {
      for (uint i = 0; i < dofmap.local_dimension(ufc_cell); ++i)
        coefficients[i] = dof_values[dofs[i]];
    }
    */
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
void Function::collect_global_dof_values() const
{
  // This function collects the global dof values for all dofs located on
  // the local mesh. These dofs may be stored in a portion of the vector
  // on another process. We build the map in two steps. First, we compute
  // which dofs are needed and send requests to the processes that own the
  // dofs. Then all processes send the requested values back.

  // Clear map
  dof_values.clear();

  std::map<uint, uint> dof_owner;

  //std::map<uint, uint> global_to_local;

  // Get mesh
  assert(_function_space);
  const Mesh& mesh = _function_space->mesh();

  // Storage for each cell dofs
  const DofMap& dofmap = _function_space->dofmap();
  const uint num_dofs_per_cell = _function_space->element().space_dimension();
  const uint num_dofs_global = vector().size();
  uint* dofs = new uint[num_dofs_per_cell];
  for (uint i = 0; i < num_dofs_per_cell; i++)
    dofs[i] = 0;

  // Iterate over mesh and check which dofs are needed
  UFCCell ufc_cell(mesh);
  //uint ii = 0;
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
      
      if (index_owner == MPI::process_number())
      {
        if (dof_values.find(dof) == dof_values.end())
          //dof_values[dof] = (*_vector)[dof - dof_offset];
          dof_values[dof] = (*_vector)[dof];
      }
      else
      {
        // Put unique new dof in communication map
        if (dof_owner.find(dof) == dof_owner.end())
          dof_owner[dof] = index_owner;
      }
    }
  }

  // Request dofs from other processes
  std::vector<uint> req_dofs;
  std::vector<uint> req_procs;

  std::map<uint, std::vector<uint> > proc_dofs;

  for (std::map<uint, uint>::const_iterator it = dof_owner.begin(); it != dof_owner.end(); ++it)
  {
    req_dofs.push_back(it->first);
    req_procs.push_back(it->second);
    proc_dofs[it->second].push_back(it->first);
  }

  MPI::distribute(req_dofs, req_procs);

  std::vector<double> send_dof_values;

  // Collect dofs belonging to other  processes
  for (uint i = 0; i < req_dofs.size(); ++i)
    //send_dof_values.push_back((*_vector)[req_dofs[i] - dof_offset]);
    send_dof_values.push_back((*_vector)[req_dofs[i]]);

  // Send and receive dofs from other processes
  MPI::distribute(send_dof_values, req_procs);

  for (uint i = 0; i < send_dof_values.size(); ++i)
  {
    std::vector<uint>& dof_vec = proc_dofs[req_procs[i]];
    const uint dof = dof_vec.front();
    dof_vec.erase(dof_vec.begin());
    dof_values[dof] = send_dof_values[i];
  }
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
  for (uint i = 0; i < num_dofs_per_cell; i++)
    dofs[i] = 0;

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
    boost::shared_ptr<GenericVector> _vec(factory.create_vector());
    _vector = _vec;
  }

  // Initialize vector of dofs
  assert(_vector);
  _vector->resize(N);
  _vector->zero();
}
//-----------------------------------------------------------------------------
void Function::get(double* block, uint m, const uint* rows) const
{
  if (dolfin::MPI::num_processes() == 1)
    _vector->get(block, m, rows);
  else
  {
    if (!_off_process_vector.get())
      error("Function has not been prepared with off-process data. Did you forget to call Function::update()?");

    // FIXME: Allocate scratch data elsewhere to allow re-use.
    uint* local_rows       = scratch.local_rows;
    uint* nonlocal_rows    = scratch.nonlocal_rows;
    double* local_block    = scratch.local_block;
    double* nonlocal_block = scratch.nonlocal_block;
    uint* local_index      = scratch.local_index;
    uint* nonlocal_index   = scratch.nonlocal_index;

    for (uint i = 0; i < m; ++i)
      block[i] = 0.0;

    // Get local ownership range
    std::pair<uint, uint> range = _vector->local_range();

    // Build lists of local and nonlocal coefficients
    uint n_local = 0;
    uint n_nonlocal = 0;
    for (uint i = 0; i < m; ++i)
    {
      if (rows[i] >= range.first && rows[i] < range.second)
      {
        local_index[n_local]  = i;
        local_rows[n_local++] = rows[i];
     }
      else 
      {
        nonlocal_index[n_nonlocal]  = i;
        nonlocal_rows[n_nonlocal++] = global_to_local[rows[i]];
      }
    }

    // Get local coefficients
    _vector->get(local_block, n_local, local_rows);

    // Get off process coefficients
    _off_process_vector->get(nonlocal_block, n_nonlocal, nonlocal_rows);

    // Copy result into block
    for (uint i = 0; i < n_local; ++i)
      block[local_index[i]] = local_block[i];      
    for (uint i = 0; i < n_nonlocal; ++i)
      block[nonlocal_index[i]] = nonlocal_block[i];      
  }
}
//-----------------------------------------------------------------------------
void Function::update()
{
  // Gather off-process coefficients if running in parallel
  if (MPI::num_processes() > 1 || has_vector())
  {
    assert(_function_space);

    // Initialise scratch space
    scratch.init(_function_space->dofmap().max_local_dimension());

    // Compute lists of off-process dofs
    compute_off_process_dofs();

    // FIXME: Add gather to GenericVector
    // Gather off-process dofs


    // Create off process vector if it doesn't exist
    if (!_off_process_vector.get())
    {
      boost::shared_ptr<GenericVector> _tmp(_vector->factory().create_local_vector());
      _off_process_vector = _tmp;
    }  

    // Gather off process coefficients
    _vector->gather(*_off_process_vector, _off_process_dofs);
  }
}
//-----------------------------------------------------------------------------
