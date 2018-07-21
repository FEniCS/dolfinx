// Copyright (C) 2007-2011 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "Form.h"
#include "GenericDofMap.h"
#include <dolfin/common/types.h>
#include <dolfin/fem/CoordinateMapping.h>
#include <dolfin/fem/FiniteElement.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/log/LogStream.h>
#include <dolfin/log/log.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshFunction.h>
#include <memory>
#include <string>
#include <ufc.h>

using namespace dolfin;
using namespace dolfin::fem;

//-----------------------------------------------------------------------------
Form::Form(std::shared_ptr<const ufc_form> ufc_form,
           const std::vector<std::shared_ptr<const function::FunctionSpace>>
               function_spaces)
    : _integrals(*ufc_form), _coefficients(*ufc_form),
      _function_spaces(function_spaces)
{
  assert(ufc_form);
  assert(ufc_form->rank == (int)function_spaces.size());

  init_coeff_scratch_space();

  // Check argument function spaces
  for (std::size_t i = 0; i < function_spaces.size(); ++i)
  {
    assert(function_spaces[i]->element());
    std::unique_ptr<ufc_finite_element> ufc_element(
        ufc_form->create_finite_element(i));

    if (std::string(ufc_element->signature)
        != function_spaces[i]->element()->signature())
    {
      log::log(ERROR, "Expected element: %s", ufc_element->signature);
      log::log(ERROR, "Input element:    %s",
               function_spaces[i]->element()->signature().c_str());
      throw std::runtime_error(
          "Cannot create form. Wrong type of function space for argument");
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

  // Create CoordinateMapping
  _coord_mapping = std::make_shared<fem::CoordinateMapping>(
      std::shared_ptr<const ufc_coordinate_mapping>(
          ufc_form->create_coordinate_mapping()));
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
Form::~Form()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
std::size_t Form::rank() const { return _function_spaces.size(); }
//-----------------------------------------------------------------------------
int Form::get_coefficient_index(std::string name) const
{
  try
  {
    return _coefficient_index_map(name.c_str());
  }
  catch (const std::bad_function_call& e)
  {
    std::cerr
        << "Unable to get coefficient index. Name-to-index map not set on Form."
        << std::endl;
    throw e;
  }

  return -1;
}
//-----------------------------------------------------------------------------
std::string Form::get_coefficient_name(int i) const
{
  try
  {
    return _coefficient_name_map(i);
  }
  catch (const std::bad_function_call& e)
  {
    std::cerr
        << "Unable to get coefficient name. Index-to-name map not set on Form."
        << std::endl;
    throw e;
  }

  return std::string();
}
//-----------------------------------------------------------------------------
void Form::set_coefficient_index_to_name_map(
    std::function<int(const char*)> coefficient_index_map)
{
  _coefficient_index_map = coefficient_index_map;
}
//-----------------------------------------------------------------------------
void Form::set_coefficient_name_to_index_map(
    std::function<const char*(int)> coefficient_name_map)
{
  _coefficient_name_map = coefficient_name_map;
}
//-----------------------------------------------------------------------------
void Form::set_coefficients(
    std::map<std::size_t, std::shared_ptr<const function::GenericFunction>>
        coefficients)

{
  for (auto c : coefficients)
    _coefficients.set(c.first, c.second);
}
//-----------------------------------------------------------------------------
void Form::set_coefficients(
    std::map<std::string, std::shared_ptr<const function::GenericFunction>>
        coefficients)
{
  for (auto c : coefficients)
  {
    // Get index
    int index = this->get_coefficient_index(c.first);
    if (index < 0)
    {
      throw std::runtime_error("Cannot find coefficient index for \"" + c.first
                               + "\"");
    }

    _coefficients.set(index, c.second);
  }
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
  for (const auto& V : _function_spaces)
  {
    assert(V->dofmap());
    num_entries *= V->dofmap()->max_element_dofs();
  }
  return num_entries;
}
//-----------------------------------------------------------------------------
void Form::set_mesh(std::shared_ptr<const mesh::Mesh> mesh)
{
  assert(mesh);
  _mesh = mesh;
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
void Form::tabulate_tensor(
    PetscScalar* A, mesh::Cell cell,
    Eigen::Ref<const EigenRowArrayXXd> coordinate_dofs) const
{
  // Switch integral based on domain from dx MeshFunction
  std::uint32_t idx = 0;
  if (dx)
  {
    // FIXME: check on idx validity
    idx = (*dx)[cell] + 1;
  }

  // Restrict coefficients to cell
  const bool* enabled_coefficients = _integrals.cell_enabled_coefficients(idx);
  for (std::size_t i = 0; i < _coefficients.size(); ++i)
  {
    if (enabled_coefficients[i])
    {
      std::shared_ptr<const function::GenericFunction> coefficient
          = _coefficients.get(i);
      const FiniteElement& element = _coefficients.element(i);
      coefficient->restrict(_wpointer[i], element, cell, coordinate_dofs);
    }
  }

  // Compute cell matrix
  auto tab_fn = _integrals.cell_tabulate_tensor(idx);
  tab_fn(A, _wpointer.data(), coordinate_dofs.data(), 1);
}
//-----------------------------------------------------------------------------
void Form::init_coeff_scratch_space()
{
  const std::size_t num_coeffs = _coefficients.size();

  // Calculate space needed for each coefficient's values
  // and create a vector of offsets from zero.
  // Allowing double space here, so that the same scratch
  // space can be also used for "macro" elements (two
  // neighbouring cells) for interior facet integrals.
  std::vector<std::uint32_t> n = {0};
  for (std::uint32_t i = 0; i < num_coeffs; ++i)
  {
    const FiniteElement& element = _coefficients.element(i);
    n.push_back(n.back() + element.space_dimension() * 2);
  }
  // Allocate memory capable of storing all coefficient values
  // in a contiguous block
  _w.resize(n.back());
  // Create pointers into _w for each coefficient
  _wpointer.resize(num_coeffs);
  for (std::uint32_t i = 0; i < num_coeffs; ++i)
    _wpointer[i] = _w.data() + n[i];
}
//-----------------------------------------------------------------------------
