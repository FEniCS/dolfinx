// Copyright (C) 2007-2008 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg, 2008.
// Modified by Martin Alnes, 2008.
//
// First added:  2007-12-10
// Last changed: 2008-12-15

#include <string>
#include <ufc.h>
#include <dolfin/common/NoDeleter.h>
#include <dolfin/log/log.h>
#include <dolfin/log/LogStream.h>
#include <dolfin/fem/FiniteElement.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/function/Function.h>
#include "Form.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
Form::Form()
{
  // Do nothing (TODO: Remove?)
}
//-----------------------------------------------------------------------------
Form::Form(dolfin::uint rank, dolfin::uint num_coefficients)
  : _function_spaces(rank),
    _coefficients(num_coefficients)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Form::Form(const ufc::form& ufc_form,
           const std::vector<FunctionSpace*>& function_spaces,
           const std::vector<Function*>& coefficients)
  : _function_spaces(function_spaces.size()),
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
  dolfin_assert(_ufc_form); 
  return _ufc_form->rank(); 
}
//-----------------------------------------------------------------------------
const Mesh& Form::mesh() const
{
  // Extract all meshes
  std::vector<const Mesh*> meshes;
  for (uint i = 0; i < _function_spaces.size(); i++)
    if (_function_spaces[i])
      meshes.push_back(&_function_spaces[i]->mesh());

  for (uint i = 0; i < _coefficients.size(); i++)
    if (_coefficients[i])
      meshes.push_back(&_coefficients[i]->function_space().mesh());

  // Check that we have at least one mesh
  if (meshes.size() == 0)
    error("Unable to extract mesh from form (no mesh found).");

  // Check that all meshes are the same
  for (uint i = 1; i < meshes.size(); i++)
    if (meshes[i] != meshes[i - 1])
      error("Unable to extract mesh from form (nonmatching meshes for function spaces).");

  // Return first mesh
  dolfin_assert(meshes[0]);
  return *meshes[0];
}
//-----------------------------------------------------------------------------
const FunctionSpace& Form::function_space(dolfin::uint i) const
{
  dolfin_assert(i < _function_spaces.size());
  return *_function_spaces[i];
}
//-----------------------------------------------------------------------------
std::vector<const FunctionSpace*> Form::function_spaces() const
{
  std::vector<const FunctionSpace*> V;
  for (uint i = 0; i < _function_spaces.size(); ++i)
    V.push_back(_function_spaces[i].get());

  return V;
}
//-----------------------------------------------------------------------------
const Function& Form::coefficient(dolfin::uint i) const
{
  dolfin_assert(i < _coefficients.size());
  return *_coefficients[i];
}
//-----------------------------------------------------------------------------
std::vector<const Function*> Form::coefficients() const
{
  std::vector<const Function*> V;
  for (uint i = 0; i < _coefficients.size(); ++i)
    V.push_back(_coefficients[i].get());

  return V;
}
//-----------------------------------------------------------------------------
dolfin::uint Form::coefficient_number(const std::string & name) const
{
  // TODO: Dissect name, assuming "wi", and return i.
  error("Not implemented.");
  return 0;
}
//-----------------------------------------------------------------------------
std::string Form::coefficient_name(dolfin::uint i) const
{
  // TODO: Create name "wi"
  error("Not implemented.");
  return "unnamed";
}
//-----------------------------------------------------------------------------
const ufc::form& Form::ufc_form() const
{
  dolfin_assert(_ufc_form);
  return *_ufc_form;
}
//-----------------------------------------------------------------------------
void Form::check() const
{
  // Check that the number of argument function spaces is correct
  if (_ufc_form->rank() != _function_spaces.size())
    error("Form expects %d FunctionSpaces, only %d provided.",
          _ufc_form->rank(), _function_spaces.size());

  // Check that the number of coefficient function spaces is correct
  if (_ufc_form->num_coefficients() != _coefficients.size())
    error("Form expects %d coefficient functions, only %d provided.",
          _ufc_form->num_coefficients(), _coefficients.size());

  // Check argument function spaces
  for (uint i = 0; i < _function_spaces.size(); ++i)
  {
    std::auto_ptr<ufc::finite_element> element(_ufc_form->create_finite_element(i));
    dolfin_assert(element.get());
    if (element->signature() != _function_spaces[i]->element().signature())
      error("Wrong type of function space for argument %d.", i);
  }

  // Check coefficients
  for (uint i = 0; i < _coefficients.size(); ++i)
  {
    if (!_coefficients[i])
      error("Coefficient %d has not been defined.", i);

    std::auto_ptr<ufc::finite_element> element(_ufc_form->create_finite_element(_ufc_form->rank() + i));
    dolfin_assert(element.get());
    if (element->signature() != _coefficients[i]->function_space().element().signature())
      error("Wrong type of function space for coefficient %d.", i);
  }
}
//-----------------------------------------------------------------------------
