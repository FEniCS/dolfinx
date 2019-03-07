// Copyright (C) 2018 Chris Richardson and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "FormIntegrals.h"
#include <dolfin/common/types.h>

#include <cstdlib>
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
  // Get list of integral IDs, and load tabulate tensor into memory for each
  _cell_integral_ids.resize(ufc_form.num_cell_integrals);
  ufc_form.get_cell_integral_ids(_cell_integral_ids.data());
  for (const auto& id : _cell_integral_ids)
  {
    std::unique_ptr<ufc_cell_integral, decltype(&std::free)> cell_integral{
        ufc_form.create_cell_integral(id), &std::free};
    assert(cell_integral);
    _tabulate_tensor_cell.push_back(cell_integral->tabulate_tensor);
  }

  _exterior_facet_integral_ids.resize(ufc_form.num_exterior_facet_integrals);
  ufc_form.get_exterior_facet_integral_ids(_exterior_facet_integral_ids.data());
  for (const auto& id : _exterior_facet_integral_ids)
  {
    std::unique_ptr<ufc_exterior_facet_integral, decltype(&std::free)>
        exterior_facet_integral{ufc_form.create_exterior_facet_integral(id),
                                &std::free};
    assert(exterior_facet_integral);
    _tabulate_tensor_exterior_facet.push_back(
        exterior_facet_integral->tabulate_tensor);
  }

  // At the moment, only accept one integral with index -1 (or none)
  if (_cell_integral_ids.size() > 1
      or (_cell_integral_ids.size() == 1 and _cell_integral_ids[0] != -1))
  {
    throw std::runtime_error(
        "Cell integral subdomain not supported. Under development.");
  }

  if (_exterior_facet_integral_ids.size() > 1
      or (_exterior_facet_integral_ids.size() == 1
          and _exterior_facet_integral_ids[0] != -1))
  {
    throw std::runtime_error(
        "Exterior facet integral subdomain not supported. Under development.");
  }
}
//-----------------------------------------------------------------------------
const std::function<void(PetscScalar*, const PetscScalar*, const double*, int)>&
FormIntegrals::get_tabulate_tensor_fn_cell(int i) const
{
  return _tabulate_tensor_cell[i];
}
//-----------------------------------------------------------------------------
const std::function<void(PetscScalar*, const PetscScalar*, const double*, int,
                         int)>&
FormIntegrals::get_tabulate_tensor_fn_exterior_facet(int i) const
{
  return _tabulate_tensor_exterior_facet[i];
}
//-----------------------------------------------------------------------------
void FormIntegrals::register_tabulate_tensor_cell(
    int i, void (*fn)(PetscScalar*, const PetscScalar*, const double*, int))
{
  if (std::find(_cell_integral_ids.begin(), _cell_integral_ids.end(), i)
      != _cell_integral_ids.end())
  {
    throw std::runtime_error("Cell integral with ID " + std::to_string(i)
                             + " already exists");
  }

  // Find insertion position
  int pos = std::distance(_cell_integral_ids.begin(),
                          std::upper_bound(_cell_integral_ids.begin(),
                                           _cell_integral_ids.end(), i));

  _cell_integral_ids.insert(_cell_integral_ids.begin() + pos, i);
  _tabulate_tensor_cell.insert(_tabulate_tensor_cell.begin() + pos, fn);
}
//-----------------------------------------------------------------------------
void FormIntegrals::register_tabulate_tensor_exterior_facet(
    int i,
    void (*fn)(PetscScalar*, const PetscScalar*, const double*, int, int))
{
  if (std::find(_exterior_facet_integral_ids.begin(),
                _exterior_facet_integral_ids.end(), i)
      != _exterior_facet_integral_ids.end())
  {
    throw std::runtime_error("Exterior facet integral with ID "
                             + std::to_string(i) + " already exists");
  }

  // Find insertion position
  int pos
      = std::distance(_exterior_facet_integral_ids.begin(),
                      std::upper_bound(_exterior_facet_integral_ids.begin(),
                                       _exterior_facet_integral_ids.end(), i));

  _exterior_facet_integral_ids.insert(
      _exterior_facet_integral_ids.begin() + pos, i);
  _tabulate_tensor_exterior_facet.insert(
      _tabulate_tensor_exterior_facet.begin() + pos, fn);
}
//-----------------------------------------------------------------------------
int FormIntegrals::num_integrals(FormIntegrals::Type type) const
{
  switch (type)
  {
  case Type::cell:
    return _tabulate_tensor_cell.size();
  case Type::exterior_facet:
    return _tabulate_tensor_exterior_facet.size();
  case Type::interior_facet:
    return 0;
  case Type::vertex:
    return 0;
  default:
    throw std::runtime_error("FormIntegral type not supported.");
  }

  return 0;
}
//-----------------------------------------------------------------------------
