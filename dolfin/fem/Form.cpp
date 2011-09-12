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
// Modified by Anders Logg, 2008-2011.
// Modified by Martin Alnes, 2008.
//
// First added:  2007-12-10
// Last changed: 2011-09-12

#include <string>
#include <boost/scoped_ptr.hpp>
#include <dolfin/common/NoDeleter.h>
#include <dolfin/fem/FiniteElement.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/function/GenericFunction.h>
#include <dolfin/log/log.h>
#include <dolfin/log/LogStream.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshData.h>
#include <dolfin/mesh/MeshEntity.h>
#include <dolfin/mesh/MeshFunction.h>
#include "Form.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
Form::Form(uint rank, uint num_coefficients)
  : Hierarchical<Form>(*this),
    dx(*this), ds(*this), dS(*this),
    _function_spaces(rank), _coefficients(num_coefficients), _rank(rank)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Form::Form(boost::shared_ptr<const ufc::form> ufc_form,
           std::vector<boost::shared_ptr<const FunctionSpace> > function_spaces,
           std::vector<boost::shared_ptr<const GenericFunction> > coefficients)
  : Hierarchical<Form>(*this),
    dx(*this), ds(*this), dS(*this), _ufc_form(ufc_form),
    _function_spaces(function_spaces), _coefficients(coefficients),
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
dolfin::uint Form::rank() const
{
  if (!_ufc_form)
    return _rank;
  else
  {
    assert(_ufc_form->rank() == _rank);
    return _rank;
  }
}
//-----------------------------------------------------------------------------
dolfin::uint Form::num_coefficients() const
{
  if (!_ufc_form)
    return _coefficients.size();
  else
  {
    assert(_ufc_form->num_coefficients() == _coefficients.size());
    return _coefficients.size();
  }
}
//-----------------------------------------------------------------------------
std::vector<dolfin::uint> Form::coloring(uint entity_dim) const
{
  warning("Form::coloring does not properly consider form type.");

  // Get mesh from first function space
  assert(_function_spaces[0]);
  const Mesh& mesh = _function_spaces[0]->mesh();
  const uint cell_dim = mesh.topology().dim();

  std::vector<uint> _coloring;
  if (entity_dim == cell_dim)
  {
    _coloring.push_back(cell_dim);
    _coloring.push_back(0);
    _coloring.push_back(cell_dim);
  }
  else if (entity_dim == cell_dim - 1)
  {
    _coloring.push_back(cell_dim - 1);
    _coloring.push_back(cell_dim);
    _coloring.push_back(0);
    _coloring.push_back(cell_dim);
    _coloring.push_back(cell_dim - 1);
  }
  else
    error("Coloring for other than cell or facet assembly not supported.");

  return _coloring;
}
//-----------------------------------------------------------------------------
void Form::set_mesh(boost::shared_ptr<const Mesh> mesh)
{
  _mesh = mesh;
}
//-----------------------------------------------------------------------------
const Mesh& Form::mesh() const
{
  // In the case when there are no function spaces (in the case of a
  // a functional) the (generated) subclass must set the mesh directly
  // by calling set_mesh().

  // Extract meshes from function spaces
  std::vector<const Mesh*> meshes;
  for (uint i = 0; i < _function_spaces.size(); i++)
  {
    if (_function_spaces[i])
      meshes.push_back(&_function_spaces[i]->mesh());
  }

  // Add common mesh if any
  if (_mesh)
    meshes.push_back(&*_mesh);

  // Extract meshes from coefficients. Note that this is only done
  // when we don't already have a mesh sine it may otherwise conflict
  // with existing meshes (if coefficient is defined on another mesh).
  if (meshes.size() == 0)
  {
    for (uint i = 0; i < _coefficients.size(); i++)
    {
      const Function* function = dynamic_cast<const Function*>(&*_coefficients[i]);
      if (function)
        meshes.push_back(&function->function_space().mesh());
    }
  }

  // Check that we have at least one mesh
  if (meshes.size() == 0)
  {
    error("Unable to extract mesh from form (no mesh found). Are you trying to assemble a functional and forgot to specify the mesh?");
  }

  // Check that all meshes are the same
  for (uint i = 1; i < meshes.size(); i++)
  {
    if (meshes[i] != meshes[i - 1])
      error("Unable to extract mesh from form (nonmatching meshes for function spaces).");
  }

  // Return first mesh
  assert(meshes[0]);
  return *meshes[0];
}
//-----------------------------------------------------------------------------
boost::shared_ptr<const dolfin::Mesh> Form::mesh_shared_ptr() const
{
  return _mesh;
}
//-----------------------------------------------------------------------------
boost::shared_ptr<const FunctionSpace> Form::function_space(uint i) const
{
  assert(i < _function_spaces.size());
  return _function_spaces[i];
}
//-----------------------------------------------------------------------------
std::vector<boost::shared_ptr<const FunctionSpace> > Form::function_spaces() const
{
  return _function_spaces;
}
//-----------------------------------------------------------------------------
void Form::set_coefficient(uint i,
                           boost::shared_ptr<const GenericFunction> coefficient)
{
  assert(i < _coefficients.size());
  _coefficients[i] = coefficient;
}
//-----------------------------------------------------------------------------
void Form::set_coefficient(std::string name,
                           boost::shared_ptr<const GenericFunction> coefficient)
{
  set_coefficient(coefficient_number(name), coefficient);
}
//-----------------------------------------------------------------------------
void Form::set_coefficients(std::map<std::string, boost::shared_ptr<const GenericFunction> > coefficients)
{
  std::map<std::string, boost::shared_ptr<const GenericFunction> >::iterator it;
  for (it = coefficients.begin(); it != coefficients.end(); ++it)
    set_coefficient(it->first, it->second);
}
//-----------------------------------------------------------------------------
boost::shared_ptr<const GenericFunction> Form::coefficient(uint i) const
{
  assert(i < _coefficients.size());
  return _coefficients[i];
}
//-----------------------------------------------------------------------------
boost::shared_ptr<const GenericFunction> Form::coefficient(std::string name) const
{
  return coefficient(coefficient_number(name));
}
//-----------------------------------------------------------------------------
std::vector<boost::shared_ptr<const GenericFunction> > Form::coefficients() const
{
  return _coefficients;
}
//-----------------------------------------------------------------------------
dolfin::uint Form::coefficient_number(const std::string & name) const
{
  // TODO: Dissect name, assuming "wi", and return i.
  dolfin_not_implemented();
  return 0;
}
//-----------------------------------------------------------------------------
std::string Form::coefficient_name(uint i) const
{
  // Create name like "w0", overloaded by Form subclasses generated by form compilers
  std::ostringstream name;
  name << "w" << i;
  return name.str();
}
//-----------------------------------------------------------------------------
boost::shared_ptr<const MeshFunction<dolfin::uint> >
Form::cell_domains_shared_ptr() const
{
  return _cell_domains;
}
//-----------------------------------------------------------------------------
boost::shared_ptr<const MeshFunction<dolfin::uint> >
Form::exterior_facet_domains_shared_ptr() const
{
  return _exterior_facet_domains;
}
//-----------------------------------------------------------------------------
boost::shared_ptr<const MeshFunction<dolfin::uint> >
Form::interior_facet_domains_shared_ptr() const
{
  return _interior_facet_domains;
}
//-----------------------------------------------------------------------------
void Form::set_cell_domains
(boost::shared_ptr<const MeshFunction<uint> > cell_domains)
{
  _cell_domains = cell_domains;
}
//-----------------------------------------------------------------------------
void Form::set_exterior_facet_domains
(boost::shared_ptr<const MeshFunction<uint> > exterior_facet_domains)
{
  _exterior_facet_domains = exterior_facet_domains;
}
//-----------------------------------------------------------------------------
void Form::set_interior_facet_domains
(boost::shared_ptr<const MeshFunction<uint> > interior_facet_domains)
{
  _interior_facet_domains = interior_facet_domains;
}
//-----------------------------------------------------------------------------
boost::shared_ptr<const ufc::form> Form::ufc_form() const
{
  return _ufc_form;
}
//-----------------------------------------------------------------------------
void Form::check() const
{
  assert(_ufc_form);

  // Check that the number of argument function spaces is correct
  if (_ufc_form->rank() != _function_spaces.size())
  {
    error("Form expects %d FunctionSpace(s), %d provided.",
          _ufc_form->rank(), _function_spaces.size());
  }

  // Check that the number of coefficient function spaces is correct
  if (_ufc_form->num_coefficients() != _coefficients.size())
  {
   error("Form expects %d coefficient function(s), %d provided.",
          _ufc_form->num_coefficients(), _coefficients.size());
  }

  // Check argument function spaces
  for (uint i = 0; i < _function_spaces.size(); ++i)
  {
    boost::scoped_ptr<ufc::finite_element> element(_ufc_form->create_finite_element(i));
    assert(element.get());
    if (element->signature() != _function_spaces[i]->element().signature())
    {
      log(ERROR, "Expected element: %s", element->signature());
      log(ERROR, "Input element:    %s", _function_spaces[i]->element().signature().c_str());
      error("Wrong type of function space for argument %d.", i);
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
