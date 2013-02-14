// Copyright (C) 2003-2012 Anders Logg
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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Garth N. Wells 2005-2010
// Modified by Martin Sandve Alnes 2008
// Modified by Andre Massing 2009
//
// First added:  2003-11-28
// Last changed: 2012-10-25

#include <algorithm>
#include <map>
#include <utility>
#include <vector>
#include <boost/assign/list_of.hpp>

#include <dolfin/adaptivity/Extrapolation.h>
#include <dolfin/common/utils.h>
#include <dolfin/common/Timer.h>
#include <dolfin/common/Array.h>
#include <dolfin/fem/FiniteElement.h>
#include <dolfin/fem/GenericDofMap.h>
#include <dolfin/fem/DirichletBC.h>
#include <dolfin/fem/UFC.h>
#include <dolfin/io/File.h>
#include <dolfin/io/XMLFile.h>
#include <dolfin/la/GenericVector.h>
#include <dolfin/la/DefaultFactory.h>
#include <dolfin/log/log.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/Vertex.h>
#include <dolfin/mesh/Point.h>
#include <dolfin/mesh/Restriction.h>
#include <dolfin/parameter/GlobalParameters.h>
#include "Expression.h"
#include "FunctionSpace.h"
#include "Function.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
Function::Function(const FunctionSpace& V)
  : Hierarchical<Function>(*this),
    _function_space(reference_to_no_delete_pointer(V)),
    allow_extrapolation(dolfin::parameters["allow_extrapolation"])
{
  // Check that we don't have a subspace
  if (!V.component().empty())
  {
    dolfin_error("Function.cpp",
                 "create function",
                 "Cannot be created from subspace. Consider collapsing the function space");
  }

  // Initialize vector
  init_vector();
}
//-----------------------------------------------------------------------------
Function::Function(boost::shared_ptr<const FunctionSpace> V)
  : Hierarchical<Function>(*this), _function_space(V),
    allow_extrapolation(dolfin::parameters["allow_extrapolation"])
{
  // Check that we don't have a subspace
  if (!V->component().empty())
  {
    dolfin_error("Function.cpp",
                 "create function",
                 "Cannot be created from subspace. Consider collapsing the function space");
  }

  // Initialize vector
  init_vector();
}
//-----------------------------------------------------------------------------
Function::Function(boost::shared_ptr<const FunctionSpace> V,
                   boost::shared_ptr<GenericVector> x)
  : Hierarchical<Function>(*this), _function_space(V), _vector(x),
    allow_extrapolation(dolfin::parameters["allow_extrapolation"])
{
  // We do not check for a subspace since this constructor is used for creating
  // subfunctions

  // Assertion uses '<=' to deal with sub-functions
  dolfin_assert(V->dofmap());
  dolfin_assert(V->dofmap()->global_dimension() <= x->size());
}
//-----------------------------------------------------------------------------
Function::Function(const FunctionSpace& V, std::string filename)
  : Hierarchical<Function>(*this),
    _function_space(reference_to_no_delete_pointer(V)),
    allow_extrapolation(dolfin::parameters["allow_extrapolation"])
{
  // Check that we don't have a subspace
  if (!V.component().empty())
  {
    dolfin_error("Function.cpp",
                 "create function",
                 "Cannot be created from subspace. Consider collapsing the function space");
  }

  // Initialize vector
  init_vector();

  // Check size of vector
  if (_vector->size() != _function_space->dim())
  {
    dolfin_error("Function.cpp",
                 "read function from file",
                 "The number of degrees of freedom (%d) does not match dimension of function space (%d)",
                 _vector->size(), _function_space->dim());
  }

  // Read function data from file
  File file(filename);
  file >> *this;
}
//-----------------------------------------------------------------------------
Function::Function(boost::shared_ptr<const FunctionSpace> V,
                   std::string filename)
  : Hierarchical<Function>(*this), _function_space(V),
    allow_extrapolation(dolfin::parameters["allow_extrapolation"])
{
  // Check that we don't have a subspace
  if (!V->component().empty())
  {
    dolfin_error("Function.cpp",
                 "create function",
                 "Cannot be created from subspace. Consider collapsing the function space");
  }

  // Initialize vector
  init_vector();

  // Check size of vector
  if (_vector->size() != _function_space->dim())
  {
    dolfin_error("Function.cpp",
                 "read function from file",
                 "The number of degrees of freedom (%d) does not match dimension of function space (%d)",
                 _vector->size(), _function_space->dim());
  }

  // Read function data from file
  File file(filename);
  file >> *this;
}
//-----------------------------------------------------------------------------
Function::Function(const Function& v)
  : Hierarchical<Function>(*this),
    allow_extrapolation(dolfin::parameters["allow_extrapolation"])
{
  // Assign data
  *this = v;
}
//-----------------------------------------------------------------------------
Function::Function(const Function& v, std::size_t i)
  : Hierarchical<Function>(*this),
    allow_extrapolation(dolfin::parameters["allow_extrapolation"])
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
  dolfin_assert(v._vector);

  // Make a copy of all the data, or if v is a sub-function, then we collapse
  // the dof map and copy only the relevant entries from the vector of v.
  if (v._vector->size() == v._function_space->dim())
  {
    // Copy function space
    _function_space = v._function_space;

    // Copy vector
    _vector = v._vector->copy();

    // Clear subfunction cache
    sub_functions.clear();
  }
  else
  {
    // Create new collapsed FunctionSpace
    boost::unordered_map<std::size_t, std::size_t> collapsed_map;
    _function_space = v._function_space->collapse(collapsed_map);

    // FIXME: This dolfin_assertion doesn't work in parallel
    //dolfin_assert(collapsed_map.size() == _function_space->dofmap()->global_dimension());
    //dolfin_assert(collapsed_map.size() == _function_space->dofmap()->local_dimension());

    // Get row indices of original and new vectors
    boost::unordered_map<std::size_t, std::size_t>::const_iterator entry;
    std::vector<dolfin::la_index> new_rows(collapsed_map.size());
    std::vector<dolfin::la_index> old_rows(collapsed_map.size());
    std::size_t i = 0;
    for (entry = collapsed_map.begin(); entry != collapsed_map.end(); ++entry)
    {
      new_rows[i]   = entry->first;
      old_rows[i++] = entry->second;
    }

    // Gather values into a vector
    std::vector<double> gathered_values;
    dolfin_assert(v.vector());
    v.vector()->gather(gathered_values, old_rows);

    // Initial new vector (global)
    init_vector();
    dolfin_assert(_function_space->dofmap());
    dolfin_assert(_vector->size() == _function_space->dofmap()->global_dimension());

    // Set values in vector
    this->_vector->set(&gathered_values[0], collapsed_map.size(), &new_rows[0]);
    this->_vector->apply("insert");
  }

  // Call assignment operator for base class
  Hierarchical<Function>::operator=(v);

  return *this;
}
//-----------------------------------------------------------------------------
const Function& Function::operator= (const Expression& v)
{
  interpolate(v);
  return *this;
}
//-----------------------------------------------------------------------------
Function& Function::operator[] (std::size_t i) const
{
  // Check if sub-Function is in the cache, otherwise create and add to cache
  boost::ptr_map<std::size_t, Function>::iterator sub_function = sub_functions.find(i);
  if (sub_function != sub_functions.end())
    return *(sub_function->second);
  else
  {
    // Extract function subspace
    std::vector<std::size_t> component = boost::assign::list_of(i);
    boost::shared_ptr<const FunctionSpace> sub_space(_function_space->extract_sub_space(component));

    // Insert sub-Function into map and return reference
    sub_functions.insert(i, new Function(sub_space, _vector));
    return *(sub_functions.find(i)->second);
  }
}
//-----------------------------------------------------------------------------
FunctionAXPY Function::operator+(const Function& other) const
{
  return FunctionAXPY(*this, other, FunctionAXPY::ADD_ADD);
}
//-----------------------------------------------------------------------------
FunctionAXPY Function::operator+(const FunctionAXPY& axpy) const
{
  return FunctionAXPY(axpy, *this, FunctionAXPY::ADD_ADD);
}
//-----------------------------------------------------------------------------
FunctionAXPY Function::operator-(const Function& other) const
{
  return FunctionAXPY(*this, other, FunctionAXPY::ADD_SUB);
}
//-----------------------------------------------------------------------------
FunctionAXPY Function::operator-(const FunctionAXPY& axpy) const
{
  return FunctionAXPY(axpy, *this, FunctionAXPY::SUB_ADD);
}
//-----------------------------------------------------------------------------
FunctionAXPY Function::operator*(double scalar) const
{
  return FunctionAXPY(*this, scalar);
}
//-----------------------------------------------------------------------------
FunctionAXPY Function::operator/(double scalar) const
{
  return FunctionAXPY(*this, 1.0/scalar);
}
//-----------------------------------------------------------------------------
void Function::operator=(const FunctionAXPY& axpy)
{
  if (axpy.pairs().size()==0)
  {
    dolfin_error("Function.cpp",
                 "assign function",
                 "FunctionAXPY is empty.");
  }
  
  // Make an initial assign and scale
  *this = *(axpy.pairs()[0].second);
  if (axpy.pairs()[0].first != 1.0)
    *_vector *= axpy.pairs()[0].first;

  // Start from item 2 and axpy 
  for (std::vector<std::pair<double, const Function*> >::const_iterator \
	 it=axpy.pairs().begin()+1;
       it!=axpy.pairs().end(); it++)
    _vector->axpy(it->first, *(it->second->vector()));
  
}
//-----------------------------------------------------------------------------
boost::shared_ptr<const FunctionSpace> Function::function_space() const
{
  dolfin_assert(_function_space);
  return _function_space;
}
//-----------------------------------------------------------------------------
boost::shared_ptr<GenericVector> Function::vector()
{
  dolfin_assert(_vector);
  dolfin_assert(_function_space->dofmap());

  // Check that this is not a sub function.
  if (_vector->size() != _function_space->dofmap()->global_dimension())
  {
    cout << "Size of vector: " << _vector->size() << endl;
    cout << "Size of function space: " << _function_space->dofmap()->global_dimension() << endl;
    dolfin_error("Function.cpp",
                 "access vector of degrees of freedom",
                 "Cannot access a non-const vector from a subfunction");
  }

  return _vector;
}
//-----------------------------------------------------------------------------
boost::shared_ptr<const GenericVector> Function::vector() const
{
  dolfin_assert(_vector);
  return _vector;
}
//-----------------------------------------------------------------------------
bool Function::in(const FunctionSpace& V) const
{
  dolfin_assert(_function_space);
  return *_function_space == V;
}
//-----------------------------------------------------------------------------
std::size_t Function::geometric_dimension() const
{
  dolfin_assert(_function_space);
  dolfin_assert(_function_space->mesh());
  return _function_space->mesh()->geometry().dim();
}
//-----------------------------------------------------------------------------
void Function::eval(Array<double>& values, const Array<double>& x) const
{
  dolfin_assert(_function_space);
  dolfin_assert(_function_space->mesh());
  const Mesh& mesh = *_function_space->mesh();

  // Find the cell that contains x
  const double* _x = x.data();
  const Point point(mesh.geometry().dim(), _x);
  int id = mesh.intersected_cell(point);

  // If not found, use the closest cell
  if (id == -1)
  {
    if (allow_extrapolation)
    {
      id = mesh.closest_cell(point);
      cout << "Extrapolating function value at x = " << point << " (not inside domain)." << endl;
    }
    else
    {
      cout << "Evaluating at x = " << point << endl;
      dolfin_error("Function.cpp",
                   "evaluate function at point",
                   "The point is not inside the domain. Consider setting \"allow_extrapolation\" to allow extrapolation");
    }
  }

  // Create cell that contains point
  const Cell cell(mesh, id);
  const UFCCell ufc_cell(cell);

  // Call evaluate function
  eval(values, x, cell, ufc_cell);
}
//-----------------------------------------------------------------------------
void Function::eval(Array<double>& values,
                    const Array<double>& x,
                    const Cell& dolfin_cell,
                    const ufc::cell& ufc_cell) const
{
  // Developer note: work arrays/vectors are re-created each time this
  //                 function is called for thread-safety

  dolfin_assert(_function_space->element());
  const FiniteElement& element = *_function_space->element();

  // Compute in tensor (one for scalar function, . . .)
  const std::size_t value_size_loc = value_size();

  dolfin_assert(values.size() == value_size_loc);

  // Create work vector for expansion coefficients
  std::vector<double> coefficients(element.space_dimension());

  // Restrict function to cell
  restrict(&coefficients[0], element, dolfin_cell, ufc_cell);

  // Create work vector for basis
  std::vector<double> basis(value_size_loc);

  // Initialise values
  for (std::size_t j = 0; j < value_size_loc; ++j)
    values[j] = 0.0;

  // Compute linear combination
  for (std::size_t i = 0; i < element.space_dimension(); ++i)
  {
    element.evaluate_basis(i, &basis[0], &x[0], ufc_cell);
    for (std::size_t j = 0; j < value_size_loc; ++j)
      values[j] += coefficients[i]*basis[j];
  }
}
//-----------------------------------------------------------------------------
void Function::interpolate(const GenericFunction& v)
{
  // Gather off-process dofs
  v.update();

  // Initialise vector
  init_vector();

  // Interpolate
  dolfin_assert(_function_space);
  _function_space->interpolate(*_vector, v);
}
//-----------------------------------------------------------------------------
void Function::extrapolate(const Function& v)
{
  Extrapolation::extrapolate(*this, v);
}
//-----------------------------------------------------------------------------
std::size_t Function::value_rank() const
{
  dolfin_assert(_function_space);
  dolfin_assert(_function_space->element());
  return _function_space->element()->value_rank();
}
//-----------------------------------------------------------------------------
std::size_t Function::value_dimension(std::size_t i) const
{
  dolfin_assert(_function_space);
  dolfin_assert(_function_space->element());
  return _function_space->element()->value_dimension(i);
}
//-----------------------------------------------------------------------------
void Function::eval(Array<double>& values, const Array<double>& x,
                    const ufc::cell& ufc_cell) const
{
  dolfin_assert(_function_space);
  dolfin_assert(_function_space->mesh());
  const Mesh& mesh = *_function_space->mesh();

  // Check if UFC cell comes from mesh, otherwise redirect to
  // evaluate on non-matching cell
  if (ufc_cell.mesh_identifier == (int) mesh.id())
  {
    const Cell cell(mesh, ufc_cell.index);
    eval(values, x, cell, ufc_cell);
  }
  else
    non_matching_eval(values, x, ufc_cell);
}
//-----------------------------------------------------------------------------
void Function::non_matching_eval(Array<double>& values,
                                 const Array<double>& x,
                                 const ufc::cell& ufc_cell) const
{
  dolfin_assert(_function_space);
  dolfin_assert(_function_space->mesh());
  const Mesh& mesh = *_function_space->mesh();

  const double* _x = x.data();
  const std::size_t dim = mesh.geometry().dim();
  const Point point(dim, _x);

  // Alternative 1: Find cell that point (x) intersects
  int id = mesh.intersected_cell(point);

  if (id == -1 && !allow_extrapolation)
  {
    dolfin_error("Function.cpp",
                 "evaluate function at point",
                 "The point is not inside the domain. Consider setting \"allow_extrapolation\" to allow extrapolation");
  }

  // Alternative 2: Compute closest cell to point (x)
  if (id == -1 && allow_extrapolation && dim == 2)
    id = mesh.closest_cell(point);

  // Alternative 3: Compute cell that contains barycenter of ufc_cell
  // NB: This is slightly heuristic, but should work well for
  // evaluation of points on refined meshes
  if (id == -1 && allow_extrapolation)
  {
    // Extract vertices of ufc_cell
    const double * const * vertices = ufc_cell.coordinates;

    Point barycenter;
    for (std::size_t i = 0; i <= dim; i++)
    {
      Point vertex(dim, vertices[i]);
      barycenter += vertex;
    }
    barycenter /= (dim + 1);
    id = mesh.intersected_cell(barycenter);
  }

  // Throw error if all alternatives failed.
  if (id == -1)
  {
    dolfin_error("Function.cpp",
                 "evaluate function at point",
                 "No matching cell found");
  }

  // Create cell that contains point
  const Cell cell(mesh, id);
  const UFCCell new_ufc_cell(cell);

  // Call evaluate function
  eval(values, x, cell, new_ufc_cell);
}
//-----------------------------------------------------------------------------
void Function::restrict(double* w,
                        const FiniteElement& element,
                        const Cell& dolfin_cell,
                        const ufc::cell& ufc_cell) const
{
  dolfin_assert(w);
  dolfin_assert(_function_space);
  dolfin_assert(_function_space->dofmap());

  // Check if we are restricting to an element of this function space
  if (_function_space->has_element(element)
      && _function_space->has_cell(dolfin_cell))
  {
    // Get dofmap for cell
    const GenericDofMap& dofmap = *_function_space->dofmap();
    const std::vector<dolfin::la_index>& dofs = dofmap.cell_dofs(dolfin_cell.index());

    // Pick values from vector(s)
    _vector->get_local(w, dofs.size(), dofs.data());
  }
  else
  {
    // Restrict as UFC function (by calling eval)
    restrict_as_ufc_function(w, element, dolfin_cell, ufc_cell);
  }
}
//-----------------------------------------------------------------------------
void Function::compute_vertex_values(std::vector<double>& vertex_values,
                                     const Mesh& mesh) const
{
  dolfin_assert(_function_space);
  dolfin_assert(_function_space->mesh());

  // Check that the mesh matches. Notice that the hash is only
  // compared if the pointers are not matching.
  if (&mesh != _function_space->mesh().get() &&
      mesh.hash() != _function_space->mesh()->hash())
  {
    dolfin_error("Function.cpp",
                 "interpolate function values at vertices",
                 "Non-matching mesh");
  }

  // Update ghosts dofs
  update();

  // Get finite element
  dolfin_assert(_function_space->element());
  const FiniteElement& element = *_function_space->element();

  // Get restriction if any
  boost::shared_ptr<const Restriction> restriction
    = _function_space->dofmap()->restriction();

  // Local data for interpolation on each cell
  const std::size_t num_cell_vertices = mesh.type().num_vertices(mesh.topology().dim());

  // Compute in tensor (one for scalar function, . . .)
  const std::size_t value_size_loc = value_size();

  // Resize Array for holding vertex values
  vertex_values.resize(value_size_loc*(mesh.num_vertices()));

  // Create vector to hold cell vertex values
  std::vector<double> cell_vertex_values(value_size_loc*num_cell_vertices);

  // Create vector for expansion coefficients
  std::vector<double> coefficients(element.space_dimension());

  // Interpolate vertex values on each cell (using last computed value
  // if not continuous, e.g. discontinuous Galerkin methods)
  UFCCell ufc_cell(mesh);
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    // Skip cells not included in restriction
    if (restriction && !restriction->contains(*cell))
      continue;

    // Update to current cell
    ufc_cell.update(*cell);

    // Pick values from global vector
    restrict(&coefficients[0], element, *cell, ufc_cell);

    // Interpolate values at the vertices
    element.interpolate_vertex_values(&cell_vertex_values[0],
                                      &coefficients[0], ufc_cell);

    // Copy values to array of vertex values
    for (VertexIterator vertex(*cell); !vertex.end(); ++vertex)
    {
      for (std::size_t i = 0; i < value_size_loc; ++i)
      {
        const std::size_t local_index  = vertex.pos()*value_size_loc + i;
        const std::size_t global_index = i*mesh.num_vertices() + vertex->index();
        vertex_values[global_index] = cell_vertex_values[local_index];
      }
    }
  }
}
//-----------------------------------------------------------------------------
void Function::compute_vertex_values(std::vector<double>& vertex_values)
{
  dolfin_assert(_function_space);
  dolfin_assert(_function_space->mesh());
  compute_vertex_values(vertex_values, *_function_space->mesh());
}
//-----------------------------------------------------------------------------
void Function::update() const
{
  if (MPI::num_processes() > 1)
    _vector->update_ghost_values();
}
//-----------------------------------------------------------------------------
void Function::init_vector()
{
  Timer timer("Init dof vector");

  // Check that function space is not a subspace (view)
  dolfin_assert(_function_space);
  if (_function_space->dofmap()->is_view())
  {
    dolfin_error("Function.cpp",
                 "initialize vector of degrees of freedom for function",
                 "Cannot be created from subspace. Consider collapsing the function space");
  }

  // Get global size
  const std::size_t N = _function_space->dofmap()->global_dimension();

  // Get local range
  const std::pair<std::size_t, std::size_t> range = _function_space->dofmap()->ownership_range();
  const std::size_t local_size = range.second - range.first;

  // Determine ghost vertices if dof map is distributed
  std::vector<std::size_t> ghost_indices;
  if (N > local_size)
    compute_ghost_indices(range, ghost_indices);

  // Create vector of dofs
  if (!_vector)
  {
    DefaultFactory factory;
    _vector = factory.create_vector();
  }
  dolfin_assert(_vector);

  // Initialize vector of dofs
  _vector->resize(range, ghost_indices);
  _vector->zero();
}
//-----------------------------------------------------------------------------
void Function::compute_ghost_indices(std::pair<std::size_t, std::size_t> range,
                                     std::vector<std::size_t>& ghost_indices) const
{
  // Clear data
  ghost_indices.clear();

  // Get mesh
  dolfin_assert(_function_space);
  dolfin_assert(_function_space->mesh());
  const Mesh& mesh = *_function_space->mesh();

  // Get dof map
  dolfin_assert(_function_space->dofmap());
  const GenericDofMap& dofmap = *_function_space->dofmap();

  // Get local range
  const std::size_t n0 = range.first;
  const std::size_t n1 = range.second;

  // Iterate over local mesh and check which dofs are needed
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    // Get dofs on cell
    const std::vector<dolfin::la_index>& dofs = dofmap.cell_dofs(cell->index());
    for (std::size_t d = 0; d < dofs.size(); ++d)
    {
      const std::size_t dof = dofs[d];
      if (dof < n0 || dof >= n1)
      {
        // FIXME: Could we use dolfin::Set here? Or unordered_set?
        if (std::find(ghost_indices.begin(), ghost_indices.end(), dof) == ghost_indices.end())
          ghost_indices.push_back(dof);
      }
    }
  }
}
//-----------------------------------------------------------------------------
