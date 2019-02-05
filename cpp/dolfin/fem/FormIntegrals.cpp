// Copyright (C) 2018 Chris Richardson and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "FormIntegrals.h"
#include <ufc.h>

using namespace dolfin;
using namespace dolfin::fem;

//-----------------------------------------------------------------------------
FormIntegrals::FormIntegrals()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
FormIntegrals::FormIntegrals(const ufc_form& ufc_form)
{

  // Integrals
  std::vector<std::unique_ptr<ufc_cell_integral>> _integrals_cell;
  std::vector<std::unique_ptr<ufc_exterior_facet_integral>>
      _integrals_exterior_facet;
  std::vector<std::unique_ptr<ufc_interior_facet_integral>>
      _interior_facet_integrals;
  std::vector<std::unique_ptr<ufc_vertex_integral>> _vertex_integrals;

  // -- Create cell integrals
  ufc_cell_integral* _default_cell_integral
      = ufc_form.create_default_cell_integral();
  if (_default_cell_integral)
  {
    _integrals_cell.push_back(
        std::unique_ptr<ufc_cell_integral>(_default_cell_integral));
  }

  for (int i = 0; i < ufc_form.max_cell_subdomain_id; ++i)
  {
    _integrals_cell.push_back(
        std::unique_ptr<ufc_cell_integral>(ufc_form.create_cell_integral(i)));
  }

  // Experimental function pointers for tabulate_tensor cell integral
  _enabled_coefficients_cell.resize(_integrals_cell.size(),
                                    ufc_form.num_coefficients);
  for (std::size_t i = 0; i < _integrals_cell.size(); ++i)
  {
    const ufc_cell_integral* ci = _integrals_cell[i].get();
    _tabulate_tensor_cell.push_back(ci->tabulate_tensor);
    std::copy(ci->enabled_coefficients,
              ci->enabled_coefficients + ufc_form.num_coefficients,
              _enabled_coefficients_cell.row(i).data());
  }

  // -- Create exterior facet integrals
  ufc_exterior_facet_integral* _default_exterior_facet_integral
      = ufc_form.create_default_exterior_facet_integral();
  if (_default_exterior_facet_integral)
  {
    _integrals_exterior_facet.push_back(
        std::unique_ptr<ufc_exterior_facet_integral>(
            _default_exterior_facet_integral));
  }

  for (int i = 0; i < ufc_form.max_exterior_facet_subdomain_id; ++i)
  {
    _integrals_exterior_facet.push_back(
        std::unique_ptr<ufc_exterior_facet_integral>(
            ufc_form.create_exterior_facet_integral(i)));
  }

  // Experimental function pointers for tabulate_tensor cell integral
  _enabled_coefficients_exterior_facet.resize(_integrals_exterior_facet.size(),
                                              ufc_form.num_coefficients);
  for (std::size_t i = 0; i < _integrals_exterior_facet.size(); ++i)
  {
    const ufc_exterior_facet_integral* fi = _integrals_exterior_facet[i].get();
    _tabulate_tensor_exterior_facet.push_back(fi->tabulate_tensor);
    std::copy(fi->enabled_coefficients,
              fi->enabled_coefficients + ufc_form.num_coefficients,
              _enabled_coefficients_exterior_facet.row(i).data());
  }

  // Interior facet integrals
  ufc_interior_facet_integral* _default_interior_facet_integral
      = ufc_form.create_default_interior_facet_integral();
  if (_default_interior_facet_integral)
  {
    _interior_facet_integrals.push_back(
        std::unique_ptr<ufc_interior_facet_integral>(
            _default_interior_facet_integral));
  }

  for (int i = 0; i < ufc_form.max_interior_facet_subdomain_id; ++i)
  {
    _interior_facet_integrals.push_back(
        std::unique_ptr<ufc_interior_facet_integral>(
            ufc_form.create_interior_facet_integral(i)));
  }

  // Vertex integrals
  ufc_vertex_integral* _default_vertex_integral
      = ufc_form.create_default_vertex_integral();
  if (_default_vertex_integral)
  {
    _vertex_integrals.push_back(
        std::unique_ptr<ufc_vertex_integral>(_default_vertex_integral));
  }

  for (int i = 0; i < ufc_form.max_vertex_subdomain_id; ++i)
  {
    _vertex_integrals.push_back(std::unique_ptr<ufc_vertex_integral>(
        ufc_form.create_vertex_integral(i)));
  }
}
//-----------------------------------------------------------------------------
const std::function<void(PetscScalar*, const PetscScalar*, const double*, int)>&
FormIntegrals::tabulate_tensor_fn_cell(int i) const
{
  return _tabulate_tensor_cell[i];
}
//-----------------------------------------------------------------------------
const std::function<void(PetscScalar*, const PetscScalar*, const double*, int,
                         int)>&
FormIntegrals::tabulate_tensor_fn_exterior_facet(int i) const
{
  return _tabulate_tensor_exterior_facet[i];
}
//-----------------------------------------------------------------------------
Eigen::Array<bool, Eigen::Dynamic, 1>
FormIntegrals::enabled_coefficients_cell(int i) const
{
  return _enabled_coefficients_cell.row(i);
}
//-----------------------------------------------------------------------------
Eigen::Array<bool, Eigen::Dynamic, 1>
FormIntegrals::enabled_coefficients_exterior_facet(int i) const
{
  return _enabled_coefficients_exterior_facet.row(i);
}
//-----------------------------------------------------------------------------
void FormIntegrals::set_tabulate_tensor_cell(
    int i, void (*fn)(PetscScalar*, const PetscScalar*, const double*, int))
{
  _tabulate_tensor_cell.resize(i + 1);
  _tabulate_tensor_cell[i] = fn;

  // Enable all coefficients for this integral
  _enabled_coefficients_cell.conservativeResize(i + 1, Eigen::NoChange);
  _enabled_coefficients_cell.row(i) = true;
}
//-----------------------------------------------------------------------------
void FormIntegrals::set_tabulate_tensor_exterior_facet(
    int i,
    void (*fn)(PetscScalar*, const PetscScalar*, const double*, int, int))
{
  _tabulate_tensor_exterior_facet.resize(i + 1);
  _tabulate_tensor_exterior_facet[i] = fn;

  // Enable all coefficients for this integral
  _enabled_coefficients_exterior_facet.conservativeResize(i + 1,
                                                          Eigen::NoChange);
  _enabled_coefficients_exterior_facet.row(i) = true;
}
//-----------------------------------------------------------------------------
int FormIntegrals::num_integrals(FormIntegrals::Type type) const
{
  switch (type)
  {
  case Type::cell:
    return _tabulate_tensor_cell.size();
  case Type::interior_facet:
    return _tabulate_tensor_exterior_facet.size();
    // case Type::exterior_facet:
    //   return _integrals_exterior_facet.size();
    // case Type::vertex:
    //   return _vertex_integrals.size();
  default:
    throw std::runtime_error("FormIntegral type not yet supported.");
  }

  return 0;
}
//-----------------------------------------------------------------------------
