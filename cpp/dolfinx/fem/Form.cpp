// Copyright (C) 2007-2019 Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "Form.h"
#include "DofMap.h"
#include <dolfinx/common/types.h>
#include <dolfinx/fem/CoordinateElement.h>
#include <dolfinx/fem/FiniteElement.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/function/Constant.h>
#include <dolfinx/function/Function.h>
#include <dolfinx/function/FunctionSpace.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/MeshEntity.h>
#include <dolfinx/mesh/MeshFunction.h>
#include <memory>
#include <string>
#include <ufc.h>

using namespace dolfinx;
using namespace dolfinx::fem;

//-----------------------------------------------------------------------------
Form::Form(
    const std::vector<std::shared_ptr<const function::FunctionSpace>>&
        function_spaces,
    const FormIntegrals& integrals, const FormCoefficients& coefficients,
    const std::vector<
        std::pair<std::string, std::shared_ptr<const function::Constant>>>
        constants,
    std::shared_ptr<const CoordinateElement> coord_mapping)
    : _integrals(integrals), _coefficients(coefficients), _constants(constants),
      _function_spaces(function_spaces), _coord_mapping(coord_mapping)
{
  // Set _mesh from function::FunctionSpace, and check they are the same
  if (!function_spaces.empty())
    _mesh = function_spaces[0]->mesh();
  for (auto& V : function_spaces)
  {
    if (_mesh != V->mesh())
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
           std::vector<std::pair<std::string,
                                 std::shared_ptr<const function::Constant>>>(),
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
    std::map<std::string, std::shared_ptr<const function::Constant>> constants)
{
  for (auto const& constant : constants)
  {
    std::string name = constant.first;

    // Find matching string in existing constants
    const auto it = std::find_if(
        _constants.begin(), _constants.end(),
        [&](const std::pair<std::string,
                            std::shared_ptr<const function::Constant>>& q) {
          return (q.first == name);
        });

    if (it == _constants.end())
      throw std::runtime_error("Constant '" + name + "' not found in form");

    it->second = constant.second;
  }
}
//-----------------------------------------------------------------------------
void Form::set_constants(
    std::vector<std::shared_ptr<const function::Constant>> constants)
{
  if (constants.size() != _constants.size())
    throw std::runtime_error("Incorrect number of constants.");

  // Loop every constant that user wants to attach
  for (std::size_t i = 0; i < constants.size(); ++i)
  {
    // In this case, the constants don't have names
    _constants[i] = std::pair("", constants[i]);
  }
}
//-----------------------------------------------------------------------------
bool Form::all_constants_set() const
{
  for (auto& constant : _constants)
    if (!constant.second)
      return false;

  return true;
}
//-----------------------------------------------------------------------------
std::set<std::string> Form::get_unset_constants() const
{
  std::set<std::string> unset;
  for (auto& constant : _constants)
  {
    if (!constant.second)
      unset.insert(constant.first);
  }
  return unset;
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
  return _function_spaces.at(i);
}
//-----------------------------------------------------------------------------
void Form::set_tabulate_tensor(
    FormIntegrals::Type type, int i,
    std::function<void(PetscScalar*, const PetscScalar*, const PetscScalar*,
                       const double*, const int*, const std::uint8_t*,
                       const bool*, const bool*, const std::uint8_t*)>
        fn)
{
  _integrals.set_tabulate_tensor(type, i, fn);
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
const std::vector<
    std::pair<std::string, std::shared_ptr<const function::Constant>>>&
Form::constants() const
{
  return _constants;
}
//-----------------------------------------------------------------------------
const fem::FormIntegrals& Form::integrals() const { return _integrals; }
//-----------------------------------------------------------------------------
std::shared_ptr<const fem::CoordinateElement> Form::coordinate_mapping() const
{
  return _coord_mapping;
}
//-----------------------------------------------------------------------------
