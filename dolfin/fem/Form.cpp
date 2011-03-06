// Copyright (C) 2007-2008 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg, 2008-2011.
// Modified by Martin Alnes, 2008.
//
// First added:  2007-12-10
// Last changed: 2011-02-03

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
Form::Form(uint rank, uint num_coefficients) : Hierarchical<Form>(*this),
                         _function_spaces(rank), _coefficients(num_coefficients)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Form::Form(boost::shared_ptr<const ufc::form> ufc_form,
           std::vector<boost::shared_ptr<const FunctionSpace> > function_spaces,
           std::vector<boost::shared_ptr<const GenericFunction> > coefficients)
  : Hierarchical<Form>(*this), _ufc_form(ufc_form),
    _function_spaces(function_spaces), _coefficients(coefficients)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Form::Form(const ufc::form& ufc_form,
           const std::vector<const FunctionSpace*>& function_spaces,
           const std::vector<const GenericFunction*>& coefficients)
  : Hierarchical<Form>(*this),
    _function_spaces(function_spaces.size()),
    _coefficients(coefficients.size())
{
  for (uint i = 0; i < function_spaces.size(); i++)
    _function_spaces[i] = reference_to_no_delete_pointer(*function_spaces[i]);

  for (uint i = 0; i < coefficients.size(); i++)
    _coefficients[i] = reference_to_no_delete_pointer(*coefficients[i]);

  _ufc_form = reference_to_no_delete_pointer(ufc_form);
}
//-----------------------------------------------------------------------------
Form::~Form()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
dolfin::uint Form::rank() const
{
  assert(_ufc_form);
  return _ufc_form->rank();
}
//-----------------------------------------------------------------------------
dolfin::uint Form::num_coefficients() const
{
  assert(_ufc_form);
  return _ufc_form->num_coefficients();
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
const ufc::form& Form::ufc_form() const
{
  assert(_ufc_form);
  return *_ufc_form;
}
//-----------------------------------------------------------------------------
boost::shared_ptr<const ufc::form> Form::ufc_form_shared_ptr() const
{
  return _ufc_form;
}
//-----------------------------------------------------------------------------
void Form::check() const
{
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
      info(ERROR, "Expected element: %s", element->signature());
      info(ERROR, "Input element:    %s", _function_spaces[i]->element().signature().c_str());
      error("Wrong type of function space for argument %d.", i);
    }
  }

  // Unable to check function spaces for coefficients (only works for Functions)

  /*
  // Check coefficients
  for (uint i = 0; i < _coefficients.size(); ++i)
  {
    if (!_coefficients[i])
      error("Coefficient %d with name '%s' has not been defined.", i, coefficient_name(i).c_str());

    std::auto_ptr<ufc::finite_element> element(_ufc_form->create_finite_element(_ufc_form->rank() + i));
    assert(element.get());
    if (element->signature() != _coefficients[i]->function_space().element().signature())
      error("Wrong type of function space for coefficient %d with name '%s', form expects\n%s\nbut we got\n%s\n...",
        i, coefficient_name(i).c_str(), element->signature(), _coefficients[i]->function_space().element().signature().c_str());
  }
  */
}
//-----------------------------------------------------------------------------
