// Copyright (C) 2007-2011 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "Form.h"
#include "GenericDofMap.h"
#include <dolfin/fem/FiniteElement.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/log/LogStream.h>
#include <dolfin/log/log.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshFunction.h>
#include <memory>
#include <string>

using namespace dolfin;
using namespace dolfin::fem;

//-----------------------------------------------------------------------------
Form::Form(
    std::shared_ptr<const ufc::form> ufc_form,
    std::vector<std::shared_ptr<const function::FunctionSpace>> function_spaces)
    : _integrals(*ufc_form), _coeffs(*ufc_form),
      _function_spaces(function_spaces)
{
  dolfin_assert(ufc_form->rank() == function_spaces.size());

  // Check argument function spaces
  for (std::size_t i = 0; i < function_spaces.size(); ++i)
  {
    dolfin_assert(function_spaces[i]->element());
    std::unique_ptr<ufc::finite_element> element(
        ufc_form->create_finite_element(i));

    if (element->signature() != function_spaces[i]->element()->signature())
    {
      log(ERROR, "Expected element: %s", element->signature());
      log(ERROR, "Input element:    %s",
          function_spaces[i]->element()->signature().c_str());
      dolfin_error("Form.cpp", "create form",
                   "Wrong type of function space for argument %d", i);
    }
  }

  // Set _mesh from function::FunctionSpace and check they are the same
  if (!function_spaces.empty())
    _mesh = function_spaces[0]->mesh();
  for (auto& f : function_spaces)
    dolfin_assert(_mesh == f->mesh());
}
//-----------------------------------------------------------------------------
Form::~Form()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
std::size_t Form::rank() const { return _function_spaces.size(); }
//-----------------------------------------------------------------------------
std::size_t Form::original_coefficient_position(std::size_t i) const
{
  return _coeffs.original_position(i);
}
//-----------------------------------------------------------------------------
std::size_t Form::max_element_tensor_size() const
{
  std::size_t num_entries = 1;
  for (const auto& V : _function_spaces)
  {
    dolfin_assert(V->dofmap());
    num_entries *= V->dofmap()->max_element_dofs();
  }
  return num_entries;
}
//-----------------------------------------------------------------------------
void Form::set_mesh(std::shared_ptr<const mesh::Mesh> mesh)
{
  dolfin_assert(mesh);
  _mesh = mesh;
}
//-----------------------------------------------------------------------------
std::shared_ptr<const mesh::Mesh> Form::mesh() const
{
  dolfin_assert(_mesh);
  return _mesh;
}
//-----------------------------------------------------------------------------
std::shared_ptr<const function::FunctionSpace>
Form::function_space(std::size_t i) const
{
  dolfin_assert(i < _function_spaces.size());
  return _function_spaces[i];
}
//-----------------------------------------------------------------------------
std::vector<std::shared_ptr<const function::FunctionSpace>>
Form::function_spaces() const
{
  return _function_spaces;
}
//-----------------------------------------------------------------------------
std::shared_ptr<const mesh::MeshFunction<std::size_t>>
Form::cell_domains() const
{
  return dx;
}
//-----------------------------------------------------------------------------
std::shared_ptr<const mesh::MeshFunction<std::size_t>>
Form::exterior_facet_domains() const
{
  return ds;
}
//-----------------------------------------------------------------------------
std::shared_ptr<const mesh::MeshFunction<std::size_t>>
Form::interior_facet_domains() const
{
  return dS;
}
//-----------------------------------------------------------------------------
std::shared_ptr<const mesh::MeshFunction<std::size_t>>
Form::vertex_domains() const
{
  return dP;
}
//-----------------------------------------------------------------------------
void Form::set_cell_domains(
    std::shared_ptr<const mesh::MeshFunction<std::size_t>> cell_domains)
{
  dx = cell_domains;
}
//-----------------------------------------------------------------------------
void Form::set_exterior_facet_domains(
    std::shared_ptr<const mesh::MeshFunction<std::size_t>>
        exterior_facet_domains)
{
  ds = exterior_facet_domains;
}
//-----------------------------------------------------------------------------
void Form::set_interior_facet_domains(
    std::shared_ptr<const mesh::MeshFunction<std::size_t>>
        interior_facet_domains)
{
  dS = interior_facet_domains;
}
//-----------------------------------------------------------------------------
void Form::set_vertex_domains(
    std::shared_ptr<const mesh::MeshFunction<std::size_t>> vertex_domains)
{
  dP = vertex_domains;
}
//-----------------------------------------------------------------------------
