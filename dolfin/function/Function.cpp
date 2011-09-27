// Copyright (C) 2003-2011 Anders Logg
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
// Last changed: 2011-03-15

#include <algorithm>
#include <map>
#include <utility>
#include <vector>
#include <boost/assign/list_of.hpp>

#include <dolfin/adaptivity/Extrapolation.h>
#include <dolfin/common/utils.h>
#include <dolfin/common/Array.h>
#include <dolfin/common/NoDeleter.h>
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
#include <dolfin/mesh/ParallelData.h>
#include <dolfin/mesh/Point.h>
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
  if (V.component().size() > 0)
    error("Cannot create Functions using subspaces. Consider collapsing the FunctionSpace.");

  // Initialize vector
  init_vector();
}
//-----------------------------------------------------------------------------
Function::Function(boost::shared_ptr<const FunctionSpace> V)
  : Hierarchical<Function>(*this), _function_space(V),
    allow_extrapolation(dolfin::parameters["allow_extrapolation"])
{
  // Check that we don't have a subspace
  if (V->component().size() > 0)
    error("Cannot create Functions using subspaces. Consider collapsing the FunctionSpace.");

  // Initialize vector
  init_vector();
}
//-----------------------------------------------------------------------------
Function::Function(const FunctionSpace& V, GenericVector& x)
  : Hierarchical<Function>(*this),
    _function_space(reference_to_no_delete_pointer(V)),
    _vector(reference_to_no_delete_pointer(x)),
    allow_extrapolation(dolfin::parameters["allow_extrapolation"])
{
  // Check that we don't have a subspace
  if (V.component().size() > 0)
    error("Cannot create Functions using subspaces. Consider collapsing the FunctionSpace.");

  // Assertion uses '<=' to deal with sub-functions
  assert(V.dofmap().global_dimension() <= x.size());
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
  assert(V->dofmap().global_dimension() <= x->size());
}
//-----------------------------------------------------------------------------
Function::Function(const FunctionSpace& V, std::string filename_vector,
                   std::string filename_dofdata)
  : Hierarchical<Function>(*this),
    _function_space(reference_to_no_delete_pointer(V)),
    allow_extrapolation(dolfin::parameters["allow_extrapolation"])
{
  // Check that we don't have a subspace
  if (V.component().size() > 0)
    error("Cannot create Functions using subspaces. Consider collapsing the FunctionSpace.");

  // Initialize vector
  init_vector();

  // Read vector from file
  if (MPI::num_processes() == 1)
  {
    File file(filename_vector);
    file >> *_vector;
  }
  else
    distribute_vector_from_file(filename_vector, filename_dofdata);

  // Check size of vector
  if (_vector->size() != _function_space->dim())
    error("Unable to read Function from file, number of degrees of freedom (%d) does not match dimension of function space (%d).", _vector->size(), _function_space->dim());
}
//-----------------------------------------------------------------------------
Function::Function(boost::shared_ptr<const FunctionSpace> V,
                   std::string filename_vector, std::string filename_dofdata)
  : Hierarchical<Function>(*this), _function_space(V),
    allow_extrapolation(dolfin::parameters["allow_extrapolation"])
{
  // Check that we don't have a subspace
  if (V->component().size() > 0)
    error("Cannot create Functions using subspaces. Consider collapsing the FunctionSpace.");

  // Create vector
  DefaultFactory factory;
  _vector.reset(factory.create_vector());

  // Initialize vector
  init_vector();

  // Read vector from file
  if (MPI::num_processes() == 1)
  {
    File file(filename_vector);
    file >> *_vector;
  }
  else
    distribute_vector_from_file(filename_vector, filename_dofdata);

  // Check size of vector
  if (_vector->size() != _function_space->dim())
    error("Unable to read Function from file, number of degrees of freedom (%d) does not match dimension of function space (%d).", _vector->size(), _function_space->dim());
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
Function::Function(const Function& v, uint i)
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
  assert(v._vector);

  // Make a copy of all the data, or if v is a sub-function, then we collapse
  // the dof map and copy only the relevant entries from the vector of v.
  if (v._vector->size() == v._function_space->dim())
  {
    // Copy function space
    _function_space = v._function_space;

    // Copy vector
    _vector.reset(v._vector->copy());

    // Clear subfunction cache
    sub_functions.clear();
  }
  else
  {
    // Create new collapase FunctionsSpapce
    boost::unordered_map<uint, uint> collapsed_map;
    _function_space = v._function_space->collapse(collapsed_map);

    // FIXME: This assertion doesn't work in parallel
    //assert(collapsed_map.size() == _function_space->dofmap().global_dimension());
    //assert(collapsed_map.size() == _function_space->dofmap().local_dimension());

    // Get row indices of original and new vectors
    boost::unordered_map<uint, uint>::const_iterator entry;
    std::vector<uint> new_rows(collapsed_map.size());
    Array<uint> old_rows(collapsed_map.size());
    uint i = 0;
    for (entry = collapsed_map.begin(); entry != collapsed_map.end(); ++entry)
    {
      new_rows[i]   = entry->first;
      old_rows[i++] = entry->second;
    }

    // Gather values into an Array
    Array<double> gathered_values;
    v.vector().gather(gathered_values, old_rows);

    // Initial new vector (global)
    init_vector();
    assert(_vector->size() == _function_space->dofmap().global_dimension());

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
    boost::shared_ptr<const FunctionSpace> sub_space(_function_space->extract_sub_space(component));

    // Insert sub-Function into map and return reference
    sub_functions.insert(i, new Function(sub_space, _vector));
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
  assert(_function_space->mesh());
  return _function_space->mesh()->geometry().dim();
}
//-----------------------------------------------------------------------------
void Function::eval(Array<double>& values, const Array<double>& x) const
{
  assert(_function_space);
  assert(_function_space->mesh());
  const Mesh& mesh = *_function_space->mesh();

  // Find the cell that contains x
  const double* _x = x.data().get();
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
      error("Unable to evaluate function at given point (not inside domain). Set option 'allow_extrapolation' to allow extrapolation'.");
    }
  }

  // Create cell that contains point
  const Cell cell(mesh, id);
  const UFCCell ufc_cell(cell, false);

  // Call evaluate function
  eval(values, x, cell, ufc_cell);
}
//-----------------------------------------------------------------------------
void Function::eval(Array<double>& values,
                    const Array<double>& x,
                    const Cell& dolfin_cell,
                    const ufc::cell& ufc_cell) const
{
  // Developer note: work arrays/vectors are re-created each time this function
  //                 is called for thread-safety

  const FiniteElement& element = _function_space->element();

  // FIXME: Rather than computing num_tensor_entries, we could just use
  //        values.size()

  // Compute in tensor (one for scalar function, . . .)
  uint num_tensor_entries = 1;
  for (uint i = 0; i < element.value_rank(); i++)
    num_tensor_entries *= element.value_dimension(i);

  assert(values.size() == num_tensor_entries);

  // Create work vector for expansion coefficients
  std::vector<double> coefficients(element.space_dimension());

  // Restrict function to cell
  restrict(&coefficients[0], element, dolfin_cell, ufc_cell);

  // Create work vector for basis
  std::vector<double> basis(num_tensor_entries);

  // Initialise values
  for (uint j = 0; j < num_tensor_entries; ++j)
    values[j] = 0.0;

  // Compute linear combination
  for (uint i = 0; i < element.space_dimension(); ++i)
  {
    element.evaluate_basis(i, &basis[0], &x[0], ufc_cell);
    for (uint j = 0; j < num_tensor_entries; ++j)
      values[j] += coefficients[i]*basis[j];
  }
}
//-----------------------------------------------------------------------------
void Function::interpolate(const GenericFunction& v)
{
  // Gather off-process dofs
  v.gather();

  // Initialise vector
  init_vector();

  // Interpolate
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
  assert(_function_space);
  return _function_space->element().value_rank();
}
//-----------------------------------------------------------------------------
dolfin::uint Function::value_dimension(uint i) const
{
  assert(_function_space);
  return _function_space->element().value_dimension(i);
}
//-----------------------------------------------------------------------------
void Function::eval(Array<double>& values, const Array<double>& x,
                    const ufc::cell& ufc_cell) const
{
  assert(_function_space);
  assert(_function_space->mesh());
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
  assert(_function_space);
  assert(_function_space->mesh());
  const Mesh& mesh = *_function_space->mesh();

  const double* _x = x.data().get();
  const uint dim = mesh.geometry().dim();
  const Point point(dim, _x);

  // Alternative 1: Find cell that point (x) intersects
  int id = mesh.intersected_cell(point);

  if (id == -1 && !allow_extrapolation)
    error("Unable to evaluate function at given point (not inside domain). Set parameter 'allow_extrapolation' to true to allow extrapolation'.");

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
    for (uint i = 0; i <= dim; i++)
    {
      Point vertex(dim, vertices[i]);
      barycenter += vertex;
    }
    barycenter /= (dim + 1);
    id = mesh.intersected_cell(barycenter);
  }

  // Throw error if all alternatives failed.
  if (id == -1)
    error("Cannot evaluate function at given point. No matching cell found.");

  // Create cell that contains point
  const Cell cell(mesh, id);
  const UFCCell new_ufc_cell(cell, false);

  // Call evaluate function
  eval(values, x, cell, new_ufc_cell);
}
//-----------------------------------------------------------------------------
void Function::restrict(double* w,
                        const FiniteElement& element,
                        const Cell& dolfin_cell,
                        const ufc::cell& ufc_cell) const
{
  assert(w);
  assert(_function_space);

  // Check if we are restricting to an element of this function space
  if (_function_space->has_element(element)
      && _function_space->has_cell(dolfin_cell))
  {
    // Get dofmap for cell
    const GenericDofMap& dofmap = _function_space->dofmap();
    const std::vector<uint>& dofs = dofmap.cell_dofs(dolfin_cell.index());

    // Pick values from vector(s)
    _vector->get_local(w, dofs.size(), &dofs[0]);
  }
  else
  {
    // Restrict as UFC function (by calling eval)
    restrict_as_ufc_function(w, element, dolfin_cell, ufc_cell);
  }
}
//-----------------------------------------------------------------------------
void Function::compute_vertex_values(Array<double>& vertex_values,
                                     const Mesh& mesh) const
{
  assert(_function_space);
  assert(_function_space->mesh());
  assert(&mesh == _function_space->mesh().get());

  // Gather ghosts dofs
  gather();

  // Get finite element
  const FiniteElement& element = _function_space->element();

  // Local data for interpolation on each cell
  const uint num_cell_vertices = mesh.type().num_vertices(mesh.topology().dim());

  // Compute in tensor (one for scalar function, . . .)
  uint num_tensor_entries = 1;
  for (uint i = 0; i < element.value_rank(); i++)
    num_tensor_entries *= element.value_dimension(i);

  // Resize Array for holding vertex values
  vertex_values.resize(num_tensor_entries*(_function_space->mesh()->num_vertices()));

  // Create vector to hold cell vertex values
  std::vector<double> cell_vertex_values(num_tensor_entries*num_cell_vertices);

  // Create vector for expansion coefficients
  std::vector<double> coefficients(element.space_dimension());

  // Interpolate vertex values on each cell (using last computed value if not
  // continuous, e.g. discontinuous Galerkin methods)
  UFCCell ufc_cell(mesh, false);
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
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
  if (MPI::num_processes() > 1)
    _vector->update_ghost_values();
}
//-----------------------------------------------------------------------------
void Function::init_vector()
{
  // Check that function space is not a subspace (view)
  if (function_space().dofmap().is_view())
    error("Cannot create a Function from a subspace. The subspace needs to be collapsed.");

  // Get global size
  const uint N = _function_space->dofmap().global_dimension();

  // Get local range
  const std::pair<uint, uint> range = _function_space->dofmap().ownership_range();
  const uint local_size = range.second - range.first;

  // Determine ghost vertices if dof map is distributed
  std::vector<uint> ghost_indices;
  if (N  > local_size)
    compute_ghost_indices(range, ghost_indices);

  // Create vector of dofs
  if (!_vector)
  {
    DefaultFactory factory;
    _vector.reset(factory.create_vector());
  }
  assert(_vector);

  // Initialize vector of dofs
  _vector->resize(range, ghost_indices);
  _vector->zero();
}
//-----------------------------------------------------------------------------
void Function::compute_ghost_indices(std::pair<uint, uint> range,
                                     std::vector<uint>& ghost_indices) const
{
  // Clear data
  ghost_indices.clear();

  // Get mesh
  assert(_function_space);
  assert(_function_space->mesh());
  const Mesh& mesh = *_function_space->mesh();

  // Get dof map
  const GenericDofMap& dofmap = _function_space->dofmap();

  // Dofs per cell
  const uint num_dofs_per_cell = _function_space->element().space_dimension();

  // Get local range
  const uint n0 = range.first;
  const uint n1 = range.second;

  // Iterate over local mesh and check which dofs are needed
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    // Get dofs on cell
    const std::vector<uint>& dofs = dofmap.cell_dofs(cell->index());

    for (uint d = 0; d < num_dofs_per_cell; ++d)
    {
      const uint dof = dofs[d];
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
void Function::distribute_vector_from_file(std::string filename_vector,
                                           std::string filename_dofdata)
{
  not_working_in_parallel("Reading Functions from file.");

  if (filename_dofdata.empty())
    error("Reading vectors for Functions in parallel requires dofmap data.");

  /*
  std::vector<std::vector<double> > send_vector_values;
  std::vector<std::vector<uint> > send_vector_indices;
  std::vector<std::vector<std::vector<uint> > > send_dofs;

  // Read data on root process and prepare data for scatter
  if (MPI::process_number() == 0)
  {
    // Read vector entries
    Array<double> _vec;
    Array<uint> _vec_indices;
    XMLFile file_vector(filename_vector);
    file_vector.read_vector(_vec, _vec_indices);
    assert(_vec.size() == _vec_indices.size());

    // Read dofmap data
    std::map<uint, std::vector<uint> > dofmap_data;
    XMLFile file_dofdata(filename_dofdata);
    file_dofdata.read_dofmap_data(dofmap_data);

    // Extract data on main process and split among processes
    const uint num_processes = MPI::num_processes();
    send_dofs.resize(num_processes);

    // Pack data for scatter
    for (uint p = 0; p < num_processes; p++)
    {
      std::vector<std::vector<uint> >& proc_send_dofs = send_dofs[p];

      // Pack dofmap data
      std::map<uint, std::vector<uint> >::const_iterator it = dofmap_data.begin();
      std::pair<uint, uint> local_range = MPI::local_range(p, send_dofs.size());
      std::advance(it, local_range.first);
      for (uint i = local_range.first; i < local_range.second; ++i)
      {
        // Add dof data
        proc_send_dofs.push_back(it->second);

        // Add cell index at end
        std::vector<uint>& last = proc_send_dofs.back();
        last.push_back(it->first);
        std::advance(it, 1);
      }

      // Pack vector data
      local_range = MPI::local_range(p, _vec.size());
      for (uint i = local_range.first; i < local_range.second; ++i)
      {
        send_vector_values[p].push_back(_vec[i]);
        send_vector_indices[p].push_back(_vec_indices[i]);
      }

    }
  }

  // Scatter file dofmap data to appropriate process
  std::vector<std::vector<uint> > received_packed_dofs;
  MPI::scatter(send_dofs, received_packed_dofs);

  // Scatter vector data appropriate process
  std::vector<double> received_vector_values;
  std::vector<uint> received_vector_indices;
  MPI::scatter(send_vector_values, received_vector_values);
  MPI::scatter(send_vector_indices, received_vector_indices);

  // Rebuild dof map data
  //std::map<uint, std::vector<uint> > my_dofmap_data;
  //std::vector<std::vector<uint> >::iterator it;
  //for (it = received_packed_dofs.begin(); it != received_packed_dofs.end(); ++it)
  //{
  //  const uint cell_index = it->back();
  //  it->pop_back();
  //  my_dofmap_data.insert(std::make_pair(cell_index, *it));
  //}

  // Build map for input map from global index to (global_cell_index, local_dof_index)
  std::map<uint, std::pair<uint, uint> > my_dofmap_data;
  std::vector<std::vector<uint> >::const_iterator cell;
  std::vector<uint>::const_iterator dof;
  for (cell = received_packed_dofs.begin(); cell != received_packed_dofs.end(); ++cell)
  {
    const uint cell_index = cell->back();
    for (dof = cell->begin(); dof != cell->end() - 1; ++dof)
    {
      const uint local_dof_index = std::distance(cell->begin(), dof);
      std::pair<uint, uint> dof_pair(cell_index, local_dof_index);
      my_dofmap_data.insert(std::make_pair(*dof, dof_pair));
    }
  }

  // Get global cell indices
  assert(_function_space);
  assert(_function_space->mesh());
  const Mesh& mesh = *_function_space->mesh();
  assert(mesh.parallel_data().have_global_entity_indices(mesh.topology().dim()));
  const MeshFunction<uint>& global_cell_indices
    = mesh.parallel_data().global_entity_indices(mesh.topology().dim());

  // Build map (global_cell, local_dof_index) -> global dof for actual dof map
  const GenericDofMap& current_dofmap = _function_space->dofmap();
  std::map<std::pair<uint, uint>, uint> new_dofmap_data;
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    const uint global_cell_index = global_cell_indices[*cell];
    const std::vector<uint>& dofs = current_dofmap.cell_dofs(cell->index());
    for (dof = dofs.begin(); dof != dofs.end(); ++dof)
    {
      const uint local_dof_index = std::distance(dofs.begin(), dof);
      new_dofmap_data.insert(std::make_pair(std::make_pair(global_cell_index, local_dof_index), *dof));
    }
  }
  */

  /*
  for (uint i = 0; i < received_vector_indices; ++i)
  {
    const uint old_index = received_vector_indices[i];
    std::map<uint, std::pair<uint, uint> >::const_iterator it;
    it = my_dofmap_data.find(old_index)
    if (it != my_dofmap_data.end()
    {
      const std::pair<uint, uint> old_index_mesh_pair = it->second;
    }
  }
  */

  // Set vector entries

}
//-----------------------------------------------------------------------------
