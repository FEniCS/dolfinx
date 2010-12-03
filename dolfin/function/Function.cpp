// Copyright (C) 2003-2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2005-2009.
// Modified by Martin Sandve Alnes, 2008.
// Modified by Andre Massing, 2009.
//
// First added:  2003-11-28
// Last changed: 2010-04-29

#include <algorithm>
#include <boost/assign/list_of.hpp>
#include <boost/scoped_array.hpp>
#include <dolfin/log/log.h>
#include <dolfin/common/utils.h>
#include <dolfin/common/Array.h>
#include <dolfin/common/NoDeleter.h>
#include <dolfin/parameter/GlobalParameters.h>
#include <dolfin/io/File.h>
#include <dolfin/la/GenericVector.h>
#include <dolfin/la/DefaultFactory.h>
#include <dolfin/fem/FiniteElement.h>
#include <dolfin/fem/GenericDofMap.h>
#include <dolfin/fem/DirichletBC.h>
#include <dolfin/fem/UFC.h>
#include <dolfin/mesh/Vertex.h>
#include <dolfin/adaptivity/Extrapolation.h>
#include "Data.h"
#include "Expression.h"
#include "FunctionSpace.h"
#include "Function.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
Function::Function(const FunctionSpace& V)
  : _function_space(reference_to_no_delete_pointer(V)),
    allow_extrapolation(dolfin::parameters["allow_extrapolation"])
{
  // Initialize vector
  init_vector();
}
//-----------------------------------------------------------------------------
Function::Function(boost::shared_ptr<const FunctionSpace> V)
  : _function_space(V),
    allow_extrapolation(dolfin::parameters["allow_extrapolation"])
{
  // Initialize vector
  init_vector();
}
//-----------------------------------------------------------------------------
Function::Function(const FunctionSpace& V, GenericVector& x)
  : _function_space(reference_to_no_delete_pointer(V)),
    _vector(reference_to_no_delete_pointer(x)),
    allow_extrapolation(dolfin::parameters["allow_extrapolation"])
{
  // Assertion uses '<=' to deal with sub-functions
  assert(V.dofmap().global_dimension() <= x.size());
}
//-----------------------------------------------------------------------------
Function::Function(boost::shared_ptr<const FunctionSpace> V,
                   boost::shared_ptr<GenericVector> x)
  : _function_space(V),
    _vector(x),
    allow_extrapolation(dolfin::parameters["allow_extrapolation"])
{
  // Assertion uses '<=' to deal with sub-functions
  assert(V->dofmap().global_dimension() <= x->size());
}
//-----------------------------------------------------------------------------
Function::Function(boost::shared_ptr<const FunctionSpace> V,
                   GenericVector& x)
  : _function_space(V),
    _vector(reference_to_no_delete_pointer(x)),
    allow_extrapolation(dolfin::parameters["allow_extrapolation"])
{
  // Assertion uses '<=' to deal with sub-functions
  assert(V->dofmap().global_dimension() <= x.size());
}
//-----------------------------------------------------------------------------
Function::Function(const FunctionSpace& V, std::string filename)
  : _function_space(reference_to_no_delete_pointer(V)),
    allow_extrapolation(dolfin::parameters["allow_extrapolation"])
{
  // Initialize vector
  init_vector();

  // Read vector from file
  File file(filename);
  file >> *_vector;

  // Check size of vector
  if (_vector->size() != _function_space->dim())
    error("Unable to read Function from file, number of degrees of freedom (%d) does not match dimension of function space (%d).", _vector->size(), _function_space->dim());
}
//-----------------------------------------------------------------------------
Function::Function(boost::shared_ptr<const FunctionSpace> V,
                   std::string filename)
  : _function_space(V),
    allow_extrapolation(dolfin::parameters["allow_extrapolation"])
{
  // Create vector
  DefaultFactory factory;
  _vector.reset(factory.create_vector());

  // Initialize vector
  init_vector();

  // Read vector from file
  File file(filename);
  file >> *_vector;

  // Check size of vector
  if (_vector->size() != _function_space->dim())
    error("Unable to read Function from file, number of degrees of freedom (%d) does not match dimension of function space (%d).", _vector->size(), _function_space->dim());
}
//-----------------------------------------------------------------------------
Function::Function(const Function& v)
  : allow_extrapolation(dolfin::parameters["allow_extrapolation"])
{
  // Assign data
  *this = v;
}
//-----------------------------------------------------------------------------
Function::Function(const Function& v, uint i)
  : allow_extrapolation(dolfin::parameters["allow_extrapolation"])
{
  // Copy function space pointer
  this->_function_space = v[i]._function_space;

  // Copy vector pointer
  this->_vector = v[i]._vector;
}
//-----------------------------------------------------------------------------
Function::~Function()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
const Function& Function::operator= (const Function& v)
{
  assert(v._vector);

  // Make a copy of all the data, or if v is a sub-function, then we collapse
  // the dof map and copy only the relevant entries from the vector of v.
  if (v._vector->size() == v._function_space->dim())
  {
    // Copy function space
    _function_space = v._function_space;

    // Copy vector
    _vector.reset(v._vector->copy());
  }
  else
  {
    // Create collapsed dof map
    std::map<uint, uint> collapsed_map;
    boost::shared_ptr<GenericDofMap> collapsed_dof_map(v._function_space->dofmap().collapse(collapsed_map, v._function_space->mesh()));

    // Create new FunctionsSpapce
    _function_space = v._function_space->collapse_sub_space(collapsed_dof_map);

    // FIXME: This assertion doesn't work in parallel
    //assert(collapsed_map.size() == _function_space->dofmap().global_dimension());

    // Create new vector (global)
    _vector.reset(v.vector().factory().create_vector());
    _vector->resize( collapsed_dof_map->global_dimension() );

    // Get row indices of original and new vectors
    std::map<uint, uint>::const_iterator entry;
    std::vector<uint> new_rows(collapsed_map.size());
    std::vector<uint> old_rows(collapsed_map.size());
    uint i = 0;
    for (entry = collapsed_map.begin(); entry != collapsed_map.end(); ++entry)
    {
      new_rows[i]   = entry->first;
      old_rows[i++] = entry->second;
    }

    // Get old values and set new values
    std::vector<double> values(collapsed_map.size());
    v.vector().get(&values[0], collapsed_map.size(), &old_rows[0]);
    this->_vector->set(&values[0], collapsed_map.size(), &new_rows[0]);
    this->_vector->apply("insert");
  }

  return *this;
}
//-----------------------------------------------------------------------------
const Function& Function::operator= (const Expression& v)
{
  interpolate(v);
  return *this;
}
//-----------------------------------------------------------------------------
Function& Function::operator[] (uint i) const
{
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
  assert(_function_space);
  return _function_space;
}
//-----------------------------------------------------------------------------
GenericVector& Function::vector()
{
  // Check that this is not a sub function.
  if (_vector->size() != _function_space->dofmap().global_dimension())
  {
    cout << "Size of vector: " << _vector->size() << endl;
    cout << "Size of function space: " << _function_space->dofmap().global_dimension() << endl;
    error("You are attempting to access a non-const vector from a sub-Function.");
  }
  return *_vector;
}
//-----------------------------------------------------------------------------
const GenericVector& Function::vector() const
{
  assert(_vector);
  return *_vector;
}
//-----------------------------------------------------------------------------
bool Function::in(const FunctionSpace& V) const
{
  assert(_function_space);
  return _function_space.get() == &V;
}
//-----------------------------------------------------------------------------
dolfin::uint Function::geometric_dimension() const
{
  assert(_function_space);
  return _function_space->mesh().geometry().dim();
}
//-----------------------------------------------------------------------------
void Function::eval(Array<double>& values, const Array<double>& x) const
{
  assert(_function_space);

  // Find the cell that contains x
  const double* _x = x.data().get();
  Point point(_function_space->mesh().geometry().dim(), _x);
  int id = _function_space->mesh().any_intersected_entity(point);

  // If not found, use the closest cell
  if (id == -1)
  {
    if (allow_extrapolation)
    {
      id = _function_space->mesh().closest_cell(point);
      cout << "Extrapolating function value at x = " << point << " (not inside domain)." << endl;
    }
    else
    {
      cout << "Evaluating at x = " << point << endl;
      error("Unable to evaluate function at given point (not inside domain). Set option 'allow_extrapolation' to allow extrapolation'.");
    }
  }


  Cell cell(_function_space->mesh(), id);
  UFCCell ufc_cell(cell);

  // Evaluate function
  eval(values, x, cell, ufc_cell);
}
//-----------------------------------------------------------------------------
void Function::eval(Array<double>& values,
                    const Array<double>& x,
                    const Cell& dolfin_cell,
                    const ufc::cell& ufc_cell) const
{
  const FiniteElement& element = _function_space->element();

  // FIXME: Rather than computing num_tensor_entries, we could just use values.size()

  // Compute in tensor (one for scalar function, . . .)
  uint num_tensor_entries = 1;
  for (uint i = 0; i < element.value_rank(); i++)
    num_tensor_entries *= element.value_dimension(i);

  assert(values.size() == num_tensor_entries);

  // Create array for basis
  boost::scoped_array<double> basis(new double[num_tensor_entries]);

  // Create array for expansion coefficients
  boost::scoped_array<double> coefficients(new double[element.space_dimension()]);

  // Restrict function to cell
  restrict(coefficients.get(), element, dolfin_cell, ufc_cell, -1);

  // Initialise values
  for (uint j = 0; j < num_tensor_entries; ++j)
    values[j] = 0.0;

  // Compute linear combination
  for (uint i = 0; i < element.space_dimension(); ++i)
  {
    element.evaluate_basis(i, basis.get(), &x[0], ufc_cell);
    for (uint j = 0; j < num_tensor_entries; ++j)
      values[j] += coefficients[i]*basis[j];
  }
}
//-----------------------------------------------------------------------------
void Function::interpolate(const GenericFunction& v)
{
  // Gather off-process dofs
  v.gather();

  init_vector();
  function_space().interpolate(*_vector, v);
}
//-----------------------------------------------------------------------------
void Function::extrapolate(const Function& v)
{
  Extrapolation::extrapolate(*this, v);
}
//-----------------------------------------------------------------------------
dolfin::uint Function::value_rank() const
{
  return _function_space->element().value_rank();
}
//-----------------------------------------------------------------------------
dolfin::uint Function::value_dimension(uint i) const
{
  return _function_space->element().value_dimension(i);
}
//-----------------------------------------------------------------------------
void Function::eval(Array<double>& values, const Data& data) const
{
  //assert(values);
  assert(_function_space);

  // Check if UFC cell if available and cell matches
  if (data._dolfin_cell && _function_space->has_cell(*data._dolfin_cell))
  {
    // Efficient evaluation on given cell
    assert(data._ufc_cell);
    eval(values, data.x, *data._dolfin_cell, *data._ufc_cell);
  }
  else
  {
    // Redirect to point-based evaluation
    eval(values, data.x);
  }
}
//-----------------------------------------------------------------------------
void Function::restrict(double* w,
                        const FiniteElement& element,
                        const Cell& dolfin_cell,
                        const ufc::cell& ufc_cell,
                        int local_facet) const
{
  assert(w);
  assert(_function_space);

  // Check if we are restricting to an element of this function space
  if (_function_space->has_element(element) && _function_space->has_cell(dolfin_cell))
  {
    // Get dofmap for cell
    const std::vector<uint>& dofs = _function_space->dofmap().cell_dofs(dolfin_cell.index());

    // FIXME: The parallel check in Function::get is a bit expensive
    // Pick values from vector(s)
    get(w, dofs.size(), &dofs[0]);
  }
  else
  {
    // Restrict as UFC function (by calling eval)
    restrict_as_ufc_function(w, element, dolfin_cell, ufc_cell, local_facet);
  }
}
//-----------------------------------------------------------------------------
void Function::compute_vertex_values(Array<double>& vertex_values,
                                     const Mesh& mesh) const
{
  assert(&mesh == &_function_space->mesh());

  // Gather off-process dofs
  gather();

  // Get finite element
  const FiniteElement& element = _function_space->element();

  // Local data for interpolation on each cell
  const uint num_cell_vertices = mesh.type().num_vertices(mesh.topology().dim());

  // Compute in tensor (one for scalar function, . . .)
  uint num_tensor_entries = 1;
  for (uint i = 0; i < element.value_rank(); i++)
    num_tensor_entries *= element.value_dimension(i);

  // Create array to hold cell vertex values
  boost::scoped_array<double> cell_vertex_values(new double[num_tensor_entries*num_cell_vertices]);

  // Create array for expansion coefficients
  boost::scoped_array<double> coefficients(new double[element.space_dimension()]);

  // Interpolate vertex values on each cell (using last computed value if not
  // continuous, e.g. discontinuous Galerkin methods)
  UFCCell ufc_cell(mesh);
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    // Update to current cell
    ufc_cell.update(*cell);

    // Pick values from global vector
    restrict(coefficients.get(), element, *cell, ufc_cell, -1);

    // Interpolate values at the vertices
    element.interpolate_vertex_values(cell_vertex_values.get(), coefficients.get(), ufc_cell);

    // Copy values to array of vertex values
    for (VertexIterator vertex(*cell); !vertex.end(); ++vertex)
    {
      for (uint i = 0; i < num_tensor_entries; ++i)
      {
        const uint local_index  = vertex.pos()*num_tensor_entries + i;
        const uint global_index = i*mesh.num_vertices() + vertex->index();
        vertex_values[global_index] = cell_vertex_values[local_index];
      }
    }
  }
}
//-----------------------------------------------------------------------------
void Function::gather() const
{
  // Gather off-process coefficients if running in parallel and function has a vector
  if (MPI::num_processes() > 1)
  {
    assert(_function_space);

    // Initialise scratch space
    gather_scratch.init(_function_space->dofmap().max_local_dimension());

    // Compute lists of off-process dofs
    compute_off_process_dofs();

    // Create off process vector if it doesn't exist
    if (!_off_process_vector.get())
      _off_process_vector.reset(_vector->factory().create_local_vector());

    // Gather off process coefficients
    const Array<uint> wrapped_off_process_dofs(_off_process_dofs.size(),
                                               &_off_process_dofs[0]);
    _vector->gather(*_off_process_vector, wrapped_off_process_dofs);
  }
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

  // Get dof map
  const GenericDofMap& dofmap = _function_space->dofmap();

  // Dofs per cell and total dofs
  const uint num_dofs_per_cell = _function_space->element().space_dimension();
  const uint num_dofs_global = vector().size();

  // Iterate over mesh and check which dofs are needed
  UFCCell ufc_cell(mesh);
  uint i = 0;
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    // Update to current cell
    ufc_cell.update(*cell);

    // Get dofs on cell
    const std::vector<uint>& dofs = dofmap.cell_dofs(cell->index());

    for (uint d = 0; d < num_dofs_per_cell; ++d)
    {
      const uint dof = dofs[d];
      const uint index_owner = MPI::index_owner(dof, num_dofs_global);
      if (index_owner != MPI::process_number())
      {
        // FIXME: Could we use dolfin::Set here?
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
void Function::init_vector()
{
  // Get size
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

  // Get local values when running in serial or collect values in parallel
  if (range.first == 0 && range.second == _vector->size())
    _vector->get_local(block, m, rows);
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
        gather_scratch.local_index[n_local]  = i;
        gather_scratch.local_rows[n_local++] = rows[i];
      }
      else
      {
        gather_scratch.nonlocal_index[n_nonlocal]  = i;
        gather_scratch.nonlocal_rows[n_nonlocal++] = global_to_local[rows[i]];
      }
    }

    // Get local coefficients
    _vector->get_local(gather_scratch.local_block, n_local, gather_scratch.local_rows);

    // Get off process coefficients
    _off_process_vector->get_local(gather_scratch.nonlocal_block, n_nonlocal, gather_scratch.nonlocal_rows);

    // Copy result into block
    for (uint i = 0; i < n_local; ++i)
      block[gather_scratch.local_index[i]] = gather_scratch.local_block[i];
    for (uint i = 0; i < n_nonlocal; ++i)
      block[gather_scratch.nonlocal_index[i]] = gather_scratch.nonlocal_block[i];
  }
}
//-----------------------------------------------------------------------------
