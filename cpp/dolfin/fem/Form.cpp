// Copyright (C) 2007-2019 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "Form.h"
#include "GenericDofMap.h"
#include <dolfin/common/types.h>
#include <dolfin/fem/CoordinateMapping.h>
#include <dolfin/fem/FiniteElement.h>
#include <dolfin/fem/utils.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Facet.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshFunction.h>
#include <memory>
#include <string>
#include <ufc.h>

using namespace dolfin;
using namespace dolfin::fem;

//-----------------------------------------------------------------------------
Form::Form(const ufc_form& ufc_form,
           const std::vector<std::shared_ptr<const function::FunctionSpace>>
               function_spaces)
    : _coefficients(fem::get_coeffs_from_ufc_form(ufc_form)),
      _function_spaces(function_spaces)
{
  assert(ufc_form.rank == (int)function_spaces.size());

  // Check argument function spaces
  for (std::size_t i = 0; i < function_spaces.size(); ++i)
  {
    assert(function_spaces[i]->element());
    std::unique_ptr<ufc_finite_element, decltype(free)*> ufc_element(
        ufc_form.create_finite_element(i), free);

    if (std::string(ufc_element->signature)
        != function_spaces[i]->element()->signature())
    {
      throw std::runtime_error(
          "Cannot create form. Wrong type of function space for argument.");
    }
  }

  // Set _mesh from function::FunctionSpace and check they are the same
  if (!function_spaces.empty())
    _mesh = function_spaces[0]->mesh();
  for (auto& f : function_spaces)
  {
    if (_mesh != f->mesh())
      throw std::runtime_error("Incompatible mesh");
  }

  // Get list of integral IDs, and load tabulate tensor into memory for each
  std::vector<int> cell_integral_ids(ufc_form.num_cell_integrals);
  ufc_form.get_cell_integral_ids(cell_integral_ids.data());
  for (int id : cell_integral_ids)
  {
    ufc_cell_integral* cell_integral = ufc_form.create_cell_integral(id);
    assert(cell_integral);
    _integrals.register_tabulate_tensor_cell(id,
                                             cell_integral->tabulate_tensor);
    std::free(cell_integral);
  }

  std::vector<int> exterior_facet_integral_ids(
      ufc_form.num_exterior_facet_integrals);
  ufc_form.get_exterior_facet_integral_ids(exterior_facet_integral_ids.data());
  for (int id : exterior_facet_integral_ids)
  {
    ufc_exterior_facet_integral* exterior_facet_integral
        = ufc_form.create_exterior_facet_integral(id);
    assert(exterior_facet_integral);
    _integrals.register_tabulate_tensor_exterior_facet(
        id, exterior_facet_integral->tabulate_tensor);
    std::free(exterior_facet_integral);
  }

  std::vector<int> interior_facet_integral_ids(
      ufc_form.num_interior_facet_integrals);
  ufc_form.get_interior_facet_integral_ids(interior_facet_integral_ids.data());
  for (int id : interior_facet_integral_ids)
  {
    ufc_interior_facet_integral* interior_facet_integral
        = ufc_form.create_interior_facet_integral(id);
    assert(interior_facet_integral);
    _integrals.register_tabulate_tensor_interior_facet(
        id, interior_facet_integral->tabulate_tensor);
    std::free(interior_facet_integral);
  }

  // Not currently working
  std::vector<int> vertex_integral_ids(ufc_form.num_vertex_integrals);
  ufc_form.get_vertex_integral_ids(vertex_integral_ids.data());
  if (vertex_integral_ids.size() > 0)
  {
    throw std::runtime_error(
        "Vertex integrals not supported. Under development.");
  }

  // Set markers for default integrals
  if (_mesh)
    _integrals.set_default_domains(*_mesh);

  // Create CoordinateMapping
  ufc_coordinate_mapping* cmap = ufc_form.create_coordinate_mapping();
  _coord_mapping = std::make_shared<fem::CoordinateMapping>(*cmap);
  std::free(cmap);
}
//-----------------------------------------------------------------------------
Form::Form(const std::vector<std::shared_ptr<const function::FunctionSpace>>
               function_spaces)
    : _coefficients({}), _function_spaces(function_spaces)
{
  // Set _mesh from function::FunctionSpace and check they are the same
  if (!function_spaces.empty())
    _mesh = function_spaces[0]->mesh();
  for (auto& f : function_spaces)
  {
    if (_mesh != f->mesh())
      throw std::runtime_error("Incompatible mesh");
  }
}
//-----------------------------------------------------------------------------
std::size_t Form::rank() const { return _function_spaces.size(); }
//-----------------------------------------------------------------------------
void Form::set_coefficients(
    std::map<std::size_t, std::shared_ptr<const function::Function>>
        coefficients)
{
  for (auto c : coefficients)
    _coefficients.set(c.first, c.second);
}
//-----------------------------------------------------------------------------
void Form::set_coefficients(
    std::map<std::string, std::shared_ptr<const function::Function>>
        coefficients)
{
  for (auto c : coefficients)
    _coefficients.set(c.first, c.second);
}
//-----------------------------------------------------------------------------
std::size_t Form::original_coefficient_position(std::size_t i) const
{
  return _coefficients.original_position(i);
}
//-----------------------------------------------------------------------------
std::size_t Form::max_element_tensor_size() const
{
  std::size_t num_entries = 1;
  for (auto& V : _function_spaces)
  {
    assert(V->dofmap());
    num_entries *= V->dofmap()->max_element_dofs();
  }
  return num_entries;
}
//-----------------------------------------------------------------------------
void Form::set_mesh(std::shared_ptr<const mesh::Mesh> mesh)
{
  _mesh = mesh;
  // Set markers for default integrals
  _integrals.set_default_domains(*_mesh);
}
//-----------------------------------------------------------------------------
std::shared_ptr<const mesh::Mesh> Form::mesh() const
{
  assert(_mesh);
  return _mesh;
}
//-----------------------------------------------------------------------------
std::shared_ptr<const function::FunctionSpace>
Form::function_space(std::size_t i) const
{
  assert(i < _function_spaces.size());
  return _function_spaces[i];
}
//-----------------------------------------------------------------------------
std::vector<std::shared_ptr<const function::FunctionSpace>>
Form::function_spaces() const
{
  return _function_spaces;
}
//-----------------------------------------------------------------------------
void Form::register_tabulate_tensor_cell(int i, void (*fn)(PetscScalar*,
                                                           const PetscScalar*,
                                                           const double*, int))
{
  _integrals.register_tabulate_tensor_cell(i, fn);
  if (i == -1 and _mesh)
    _integrals.set_default_domains(*_mesh);
}
//-----------------------------------------------------------------------------
void Form::set_cell_domains(const mesh::MeshFunction<std::size_t>& cell_domains)
{
  _integrals.set_domains(FormIntegrals::Type::cell, cell_domains);
}
//-----------------------------------------------------------------------------
void Form::set_exterior_facet_domains(
    const mesh::MeshFunction<std::size_t>& exterior_facet_domains)
{
  _integrals.set_domains(FormIntegrals::Type::exterior_facet,
                         exterior_facet_domains);
}
//-----------------------------------------------------------------------------
void Form::set_interior_facet_domains(
    const mesh::MeshFunction<std::size_t>& interior_facet_domains)
{
  _integrals.set_domains(FormIntegrals::Type::interior_facet,
                         interior_facet_domains);
}
//-----------------------------------------------------------------------------
void Form::set_vertex_domains(
    const mesh::MeshFunction<std::size_t>& vertex_domains)
{
  _integrals.set_domains(FormIntegrals::Type::vertex, vertex_domains);
}
//-----------------------------------------------------------------------------
std::shared_ptr<const fem::CoordinateMapping> Form::coordinate_mapping() const
{
  return _coord_mapping;
}
//-----------------------------------------------------------------------------
