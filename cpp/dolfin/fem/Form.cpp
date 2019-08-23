// Copyright (C) 2007-2019 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "Form.h"
#include "DofMap.h"
#include <dolfin/common/types.h>
#include <dolfin/fem/CoordinateMapping.h>
#include <dolfin/fem/FiniteElement.h>
#include <dolfin/fem/utils.h>
#include <dolfin/function/Constant.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshEntity.h>
#include <dolfin/mesh/MeshFunction.h>
#include <memory>
#include <string>
#include <ufc.h>

using namespace dolfin;
using namespace dolfin::fem;

//-----------------------------------------------------------------------------
Form::Form(const std::vector<std::shared_ptr<const function::FunctionSpace>>&
               function_spaces,
           const FormIntegrals& integrals, const FormCoefficients& coefficients,
           const std::vector<
               std::pair<std::string, std::shared_ptr<function::Constant>>>
               constants,
           std::shared_ptr<const CoordinateMapping> coord_mapping)
    : _integrals(integrals), _coefficients(coefficients), _constants(constants),
      _function_spaces(function_spaces), _coord_mapping(coord_mapping)
{
  // Set _mesh from function::FunctionSpace, and check they are the same
  if (!function_spaces.empty())
    _mesh = function_spaces[0]->mesh;
  for (auto& V : function_spaces)
  {
    if (_mesh != V->mesh)
      throw std::runtime_error("Incompatible mesh");
  }

  // Set markers for default integrals
  if (_mesh)
    _integrals.set_default_domains(*_mesh);
}
//-----------------------------------------------------------------------------
Form::Form(const std::vector<std::shared_ptr<const function::FunctionSpace>>&
               function_spaces)
    : Form(function_spaces, FormIntegrals(), FormCoefficients({}),
           std::vector<
               std::pair<std::string, std::shared_ptr<function::Constant>>>(),
           nullptr)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
int Form::rank() const { return _function_spaces.size(); }
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
int Form::original_coefficient_position(int i) const
{
  return _coefficients.original_position(i);
}
//-----------------------------------------------------------------------------
void Form::set_constants(
    std::map<std::string, std::shared_ptr<function::Constant>> constants)
{
  for (auto const& constant_in : constants)
  {
    std::string name_in = constant_in.first;

    // Find matching string in existing constants
    const auto it = std::find_if(
        _constants.begin(), _constants.end(),
        [&](const std::pair<std::string, std::shared_ptr<function::Constant>>&
                q) { return (q.first == name_in); });

    if (it == _constants.end())
      throw std::runtime_error("Constant '" + name_in + "' not found in form");

    it->second = constant_in.second;
  }
}
//-----------------------------------------------------------------------------
void Form::set_constants(
    std::vector<std::shared_ptr<function::Constant>> constants)
{
  if (constants.size() != _constants.size())
    throw std::runtime_error("Incorrect number of constants.");

  // Loop every constant that user wants to attach
  for (std::size_t i = 0; i < constants.size(); ++i)
  {
    // In this case, the constants don't have names
    _constants[i] = std::make_pair("", constants[i]);
  }
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
std::shared_ptr<const function::FunctionSpace> Form::function_space(int i) const
{
  assert(i < (int)_function_spaces.size());
  return _function_spaces[i];
}
//-----------------------------------------------------------------------------
void Form::register_tabulate_tensor_cell(
    int i, void (*fn)(PetscScalar*, const PetscScalar*, const PetscScalar*,
                      const double*, const int*, const int*))
{
  _integrals.register_tabulate_tensor(FormIntegrals::Type::cell, i, fn);
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
fem::FormCoefficients& Form::coefficients() { return _coefficients; }
//-----------------------------------------------------------------------------
const fem::FormCoefficients& Form::coefficients() const
{
  return _coefficients;
}
//-----------------------------------------------------------------------------
std::vector<std::pair<std::string, std::shared_ptr<function::Constant>>>&
Form::constants()
{
  return _constants;
}
//-----------------------------------------------------------------------------
const std::vector<std::pair<std::string, std::shared_ptr<function::Constant>>>&
Form::constants() const
{
  return _constants;
}
//-----------------------------------------------------------------------------
const fem::FormIntegrals& Form::integrals() const { return _integrals; }
//-----------------------------------------------------------------------------
std::shared_ptr<const fem::CoordinateMapping> Form::coordinate_mapping() const
{
  return _coord_mapping;
}
//-----------------------------------------------------------------------------
