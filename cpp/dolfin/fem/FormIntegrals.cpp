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
  // This function makes many unsupported assumptiosn with regard to the
  // UFC interface. Rather than overcomplicate the code here, it needs
  // to be fixed inm UFC/FFC. See
  // https://github.com/FEniCS/ffcx/issues/103

  if (ufc_form.num_cell_integrals > 1)
  {
    throw std::runtime_error(
        "Cell integral subdomain not supported. Under development.");
  }
  if (ufc_form.num_exterior_facet_integrals > 1)
  {
    throw std::runtime_error(
        "Exterior facet integral subdomain not supported. Under development.");
  }

  // -- Create cell integrals
  std::unique_ptr<ufc_cell_integral, decltype(&std::free)>
      _default_cell_integral{ufc_form.create_cell_integral(-1), &std::free};

  if (_default_cell_integral)
    _tabulate_tensor_cell.push_back(_default_cell_integral->tabulate_tensor);

  // -- Create exterior facet integrals
  std::unique_ptr<ufc_exterior_facet_integral, decltype(&std::free)>
      _default_exterior_facet_integral{
          ufc_form.create_exterior_facet_integral(-1), &std::free};

  if (_default_exterior_facet_integral)
  {
    // Extract tabulate tensor function
    _tabulate_tensor_exterior_facet.push_back(
        _default_exterior_facet_integral->tabulate_tensor);
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
void FormIntegrals::set_tabulate_tensor_cell(
    int i, void (*fn)(PetscScalar*, const PetscScalar*, const double*, int))
{
  _tabulate_tensor_cell.resize(i + 1);
  _tabulate_tensor_cell[i] = fn;
}
//-----------------------------------------------------------------------------
void FormIntegrals::set_tabulate_tensor_exterior_facet(
    int i,
    void (*fn)(PetscScalar*, const PetscScalar*, const double*, int, int))
{
  _tabulate_tensor_exterior_facet.resize(i + 1);
  _tabulate_tensor_exterior_facet[i] = fn;
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
