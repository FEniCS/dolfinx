// Copyright (C) 2007-2011 Garth N. Wells
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
// Modified by Anders Logg 2008-2014
// Modified by Martin Alnes 2008
//
// First added:  2007-12-10
// Last changed: 2015-11-08

#include <memory>
#include <string>

#include <dolfin/common/NoDeleter.h>
#include <dolfin/fem/FiniteElement.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/function/GenericFunction.h>
#include <dolfin/log/log.h>
#include <dolfin/log/LogStream.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshData.h>
#include <dolfin/mesh/MeshFunction.h>
#include "Form.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
Form::Form(std::size_t rank, std::size_t num_coefficients)
  : Hierarchical<Form>(*this),  _function_spaces(rank),
  _coefficients(num_coefficients), _rank(rank)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Form::Form(std::shared_ptr<const ufc::form> ufc_form,
           std::vector<std::shared_ptr<const FunctionSpace>> function_spaces)
  : Hierarchical<Form>(*this), _ufc_form(ufc_form),
    _function_spaces(function_spaces), _coefficients(ufc_form->num_coefficients()),
    _rank(ufc_form->rank())
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Form::~Form()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
std::size_t Form::rank() const
{
  if (!_ufc_form)
    return _rank;
  else
  {
    dolfin_assert(_ufc_form->rank() == _rank);
    return _rank;
  }
}
//-----------------------------------------------------------------------------
std::size_t Form::num_coefficients() const
{
  if (!_ufc_form)
    return _coefficients.size();
  else
  {
    dolfin_assert(_ufc_form->num_coefficients() == _coefficients.size());
    return _coefficients.size();
  }
}
//-----------------------------------------------------------------------------
std::size_t Form::original_coefficient_position(std::size_t i) const
{
  dolfin_assert(_ufc_form);
  return _ufc_form->original_coefficient_position(i);
}
//-----------------------------------------------------------------------------
std::vector<std::size_t> Form::coloring(std::size_t entity_dim) const
{
  warning("Form::coloring does not properly consider form type.");

  // Get mesh
  dolfin_assert(this->mesh());
  const Mesh& mesh = *(this->mesh());
  const std::size_t cell_dim = mesh.topology().dim();

  std::vector<std::size_t> _coloring;
  if (entity_dim == cell_dim)
    _coloring = {{cell_dim, 0, cell_dim}};
  else if (entity_dim == cell_dim - 1)
    _coloring = {{cell_dim - 1, cell_dim, 0, cell_dim, cell_dim - 1}};
  else
  {
    dolfin_error("Form.cpp",
                 "color form for multicore computing",
                 "Only cell and facet coloring are currently supported");
  }

  return _coloring;
}
//-----------------------------------------------------------------------------
void Form::set_mesh(std::shared_ptr<const Mesh> mesh)
{
  _mesh = mesh;
}
//-----------------------------------------------------------------------------
std::shared_ptr<const Mesh> Form::mesh() const
{
  // In the case when there are no function spaces (in the case of a
  // a functional) the (generated) subclass must set the mesh directly
  // by calling set_mesh().

  // Extract meshes from function spaces
  std::vector<std::shared_ptr<const Mesh>> meshes;
  for (std::size_t i = 0; i < _function_spaces.size(); i++)
  {
    if (_function_spaces[i])
    {
      dolfin_assert(_function_spaces[i]->mesh());
      meshes.push_back(_function_spaces[i]->mesh());
    }
  }

  // Add common mesh if any
  if (_mesh)
    meshes.push_back(_mesh);

  // Extract meshes from markers if any
  if (dx)
    meshes.push_back(dx->mesh());
  if (ds)
    meshes.push_back(ds->mesh());
  if (dS)
    meshes.push_back(dS->mesh());
  if (dP)
    meshes.push_back(dP->mesh());

  // Extract meshes from coefficients. Note that this is only done
  // when we don't already have a mesh sine it may otherwise conflict
  // with existing meshes (if coefficient is defined on another mesh).
  if (meshes.empty())
  {
    for (std::size_t i = 0; i < _coefficients.size(); i++)
    {
      const Function* function
        = dynamic_cast<const Function*>(&*_coefficients[i]);
      if (function && function->function_space()->mesh())
        meshes.push_back(function->function_space()->mesh());
    }
  }

  // Check that we have at least one mesh
  if (meshes.empty())
  {
    dolfin_error("Form.cpp",
                 "extract mesh from form",
                 "No mesh was found. Try passing mesh to the assemble function");
  }

  // Check that all meshes are the same
  for (std::size_t i = 1; i < meshes.size(); i++)
  {
    if (meshes[i] != meshes[i - 1])
    {
      dolfin_error("Form.cpp",
                   "extract mesh from form",
                   "Non-matching meshes for function spaces and/or measures");
    }
  }

  // Return first mesh
  dolfin_assert(meshes[0]);
  return meshes[0];
}
//-----------------------------------------------------------------------------
std::shared_ptr<const FunctionSpace> Form::function_space(std::size_t i) const
{
  dolfin_assert(i < _function_spaces.size());
  return _function_spaces[i];
}
//-----------------------------------------------------------------------------
std::vector<std::shared_ptr<const FunctionSpace>> Form::function_spaces() const
{
  return _function_spaces;
}
//-----------------------------------------------------------------------------
void Form::set_coefficient(std::size_t i,
                           std::shared_ptr<const GenericFunction> coefficient)
{
  dolfin_assert(i < _coefficients.size());
  _coefficients[i] = coefficient;
}
//-----------------------------------------------------------------------------
void Form::set_coefficient(std::string name,
                           std::shared_ptr<const GenericFunction> coefficient)
{
  set_coefficient(coefficient_number(name), coefficient);
}
//-----------------------------------------------------------------------------
void Form::set_coefficients(std::map<std::string, std::shared_ptr<const GenericFunction>> coefficients)
{
  for (auto it = coefficients.begin(); it != coefficients.end(); ++it)
    set_coefficient(it->first, it->second);
}
//-----------------------------------------------------------------------------
void Form::set_some_coefficients(std::map<std::string,
                                 std::shared_ptr<const GenericFunction>> coefficients)
{
  // Build map of which coefficients has been set
  std::map<std::string, bool> markers;
  for (std::size_t i = 0; i < num_coefficients(); i++)
    markers[coefficient_name(i)] = false;

  // Set all coefficients that need to be set
  for (auto it = coefficients.begin(); it != coefficients.end(); ++it)
  {
    auto name = it->first;
    auto coefficient = it->second;
    if (markers.find(name) != markers.end())
    {
      set_coefficient(name, coefficient);
      markers[name] = true;
    }
  }

  // Check which coefficients that have been set
  std::stringstream s_set;
  std::stringstream s_unset;
  std::size_t num_set = 0;
  for (auto it = markers.begin(); it != markers.end(); ++it)
  {
    if (it->second)
    {
      num_set++;
      s_set << " " << it->first;
    }
    else
      s_unset << " " << it->second;
  }

  // Report status of set coefficients
  if (num_set == num_coefficients())
    info("All coefficients attached to form:%s", s_set.str().c_str());
  else
  {
    info("%d coefficient(s) attached to form:%s",
         num_set, s_set.str().c_str());
    info("%d coefficient(s) missing: %s",
         num_coefficients() - num_set, s_unset.str().c_str());
  }
}
//-----------------------------------------------------------------------------
std::shared_ptr<const GenericFunction> Form::coefficient(std::size_t i) const
{
  dolfin_assert(i < _coefficients.size());
  return _coefficients[i];
}
//-----------------------------------------------------------------------------
std::shared_ptr<const GenericFunction> Form::coefficient(std::string name) const
{
  return coefficient(coefficient_number(name));
}
//-----------------------------------------------------------------------------
std::vector<std::shared_ptr<const GenericFunction>> Form::coefficients() const
{
  return _coefficients;
}
//-----------------------------------------------------------------------------
std::size_t Form::coefficient_number(const std::string & name) const
{
  // TODO: Dissect name, assuming "wi", and return i.
  dolfin_not_implemented();
  return 0;
}
//-----------------------------------------------------------------------------
std::string Form::coefficient_name(std::size_t i) const
{
  // Create name like "w0", overloaded by Form subclasses generated by form compilers
  std::ostringstream name;
  name << "w" << i;
  return name.str();
}
//-----------------------------------------------------------------------------
std::shared_ptr<const MeshFunction<std::size_t>> Form::cell_domains() const
{
  return dx;
}
//-----------------------------------------------------------------------------
std::shared_ptr<const MeshFunction<std::size_t>> Form::exterior_facet_domains() const
{
  return ds;
}
//-----------------------------------------------------------------------------
std::shared_ptr<const MeshFunction<std::size_t>> Form::interior_facet_domains() const
{
  return dS;
}
//-----------------------------------------------------------------------------
std::shared_ptr<const MeshFunction<std::size_t>> Form::vertex_domains() const
{
  return dP;
}
//-----------------------------------------------------------------------------
void Form::set_cell_domains
(std::shared_ptr<const MeshFunction<std::size_t>> cell_domains)
{
  dx = cell_domains;
}
//-----------------------------------------------------------------------------
void Form::set_exterior_facet_domains
(std::shared_ptr<const MeshFunction<std::size_t>> exterior_facet_domains)
{
  ds = exterior_facet_domains;
}
//-----------------------------------------------------------------------------
void Form::set_interior_facet_domains
(std::shared_ptr<const MeshFunction<std::size_t>> interior_facet_domains)
{
  dS = interior_facet_domains;
}
//-----------------------------------------------------------------------------
void Form::set_vertex_domains
(std::shared_ptr<const MeshFunction<std::size_t>> vertex_domains)
{
  dP = vertex_domains;
}
//-----------------------------------------------------------------------------
std::shared_ptr<const ufc::form> Form::ufc_form() const
{
  return _ufc_form;
}
//-----------------------------------------------------------------------------
void Form::check() const
{
  dolfin_assert(_ufc_form);

  // Check that the number of argument function spaces is correct
  if (_ufc_form->rank() != _function_spaces.size())
  {
    dolfin_error("Form.cpp",
                 "assemble form",
                 "Expecting %d function spaces (not %d)",
                 _ufc_form->rank(), _function_spaces.size());
  }

  // Check that the number of coefficient function spaces is correct
  if (_ufc_form->num_coefficients() != _coefficients.size())
  {
   dolfin_error("Form.cpp",
                "assemble form",
                "Expecting %d coefficient (not %d)",
                _ufc_form->num_coefficients(), _coefficients.size());
  }

  // Check argument function spaces
  for (std::size_t i = 0; i < _function_spaces.size(); ++i)
  {
    std::unique_ptr<ufc::finite_element>
      element(_ufc_form->create_finite_element(i));
    dolfin_assert(element);
    dolfin_assert(_function_spaces[i]->element());
    if (element->signature() != _function_spaces[i]->element()->signature())
    {
      log(ERROR, "Expected element: %s", element->signature());
      log(ERROR, "Input element:    %s",
          _function_spaces[i]->element()->signature().c_str());
      dolfin_error("Form.cpp",
                   "assemble form",
                   "Wrong type of function space for argument %d", i);
    }
  }
}
//-----------------------------------------------------------------------------
Equation Form::operator==(const Form& rhs) const
{
  Equation equation(reference_to_no_delete_pointer(*this),
                    reference_to_no_delete_pointer(rhs));
  return equation;
}
//-----------------------------------------------------------------------------
Equation Form::operator==(int rhs) const
{
  Equation equation(reference_to_no_delete_pointer(*this), 0);
  return equation;
}
//-----------------------------------------------------------------------------
