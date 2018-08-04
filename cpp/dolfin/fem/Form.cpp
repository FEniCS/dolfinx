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
unsigned int Form::cell_batch_size() const
{
  // FIXME: Current implementation assumes that all subdmains/integrals share
  //   the same cell_batch_size
  std::uint32_t idx = 0;

  return _integrals.cell_batch_size(idx);
}
//-----------------------------------------------------------------------------
void Form::tabulate_tensor(
    PetscScalar* A, mesh::Cell cell,
    Eigen::Ref<const EigenRowArrayXXd> coordinate_dofs) const
{
  assert(cell_batch_size() == 1);

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
      coefficient->restrict(_w_ptr[i], element, cell, coordinate_dofs);
    }
  }

  // Compute cell matrix
  auto tab_fn = _integrals.cell_tabulate_tensor(idx);
  tab_fn(A, _w_ptr.data(), coordinate_dofs.data(), 1);
}
//-----------------------------------------------------------------------------
void Form::tabulate_tensor_batch(
    PetscScalar* A, const std::vector<mesh::Cell>& cell_batch,
    const std::vector<EigenRowArrayXXd>& coordinate_dofs_batch) const
{
  const unsigned int cell_batch_size = this->cell_batch_size();

  assert(cell_batch.size() == cell_batch_size);
  assert(cell_batch.size() == coordinate_dofs_batch.size());

  // Switch integral based on domain from dx MeshFunction
  std::uint32_t idx = 0;
  if (dx)
  {
    // FIXME: check on idx validity
    idx = (*dx)[cell_batch.front()] + 1;

    // FIXME: see cell_batch_size() and Assembler::assemble()
    for (const auto& cell : cell_batch)
      if (idx != (*dx)[cell] + 1)
        throw std::runtime_error("Different integrals in a single cell batch "
                                 "are not supported.");
  }

  // The following code can unfortunately not be trivially implemented without
  // copying values from non-interleaved single-cell arrays to the interleaved
  // arrays required by the batched tabulate_tensor functions.
  // Reasons:
  //   - Expression-type coefficients are user provided
  //   - Function-type coefficients use PETSc get_local functions
  //   - If the cases above are not batched, coordinate_dofs are also
  //     required to not be interleaved for correct per-cell evaluation

  // Restrict coefficients and make cell-interleaved copy
  for (std::size_t i = 0; i < cell_batch.size(); ++i)
  {
    const auto& cell = cell_batch[i];
    const auto& cell_coord_dofs = coordinate_dofs_batch[i];

    const bool* enabled_coefficients
        = _integrals.cell_enabled_coefficients(idx);

    for (std::size_t j = 0; j < _coefficients.size(); ++j)
    {
      if (enabled_coefficients[j])
      {
        std::shared_ptr<const function::GenericFunction> coefficient
            = _coefficients.get(j);
        const FiniteElement& element = _coefficients.element(j);

        coefficient->restrict(_w_temp.data(), element, cell, cell_coord_dofs);

        // Copy restricted values to cell-interleaved coefficient array
        for (std::uint32_t k = 0; k < element.space_dimension() * 2; ++k)
          _w_ptr[j][cell_batch_size * k + i] = _w_temp[k];
      }
    }
  }

  // Ensure that temporary coordinate dofs storage is large enough
  _coord_dofs.conservativeResize(coordinate_dofs_batch.front().rows(),
                                 coordinate_dofs_batch.front().cols() *
                                 coordinate_dofs_batch.size());

  // Create cell-interleaved copy of coordinate dofs
  for (std::size_t i = 0; i < cell_batch.size(); ++i)
  {
    const auto& cell_coord_dofs = coordinate_dofs_batch[i];
    for (int k = 0; k < cell_coord_dofs.rows(); ++k)
    {
      for (int l = 0; l < cell_coord_dofs.cols(); ++l)
      {
        _coord_dofs(k, cell_batch_size * l + i)
            = cell_coord_dofs(k, l);
      }
    }
  }

  // Call to batched tabulate_tensor method with interleaved inputs/outputs
  auto tab_fn = _integrals.cell_tabulate_tensor(idx);
  tab_fn(A, _w_ptr.data(), _coord_dofs.data(), 1);
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
  std::vector<std::uint32_t> ns = {0};
  std::uint32_t n_max = 0;
  for (std::uint32_t i = 0; i < num_coeffs; ++i)
  {
    const FiniteElement& element = _coefficients.element(i);
    std::uint32_t n = element.space_dimension() * cell_batch_size() * 2;
    n_max = std::max(n_max, n);
    ns.push_back(ns.back() + n);
  }

  // Allocate memory capable of temporarily storing values of any coefficient
  // in a contiguous block
  _w_temp.resize(n_max);

  // Memory for storing all coefficients as input for tabulate_tensor
  // May be used for interleaved storage when using batched assembly
  _w_interleaved.resize(ns.back());
  // Create pointers into _w for each coefficient
  _w_ptr.resize(num_coeffs);
  for (std::uint32_t i = 0; i < num_coeffs; ++i)
    _w_ptr[i] = _w_interleaved.data() + ns[i];
}
//-----------------------------------------------------------------------------
