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
FormIntegrals::FormIntegrals(const ufc_form& ufc_form)
{
  // -- Create cell integrals
  ufc_cell_integral* _default_cell_integral
      = ufc_form.create_default_cell_integral();
  if (_default_cell_integral)
  {
    _integrals_cell.push_back(
        std::shared_ptr<ufc_cell_integral>(_default_cell_integral));
  }

  const std::size_t num_cell_domains = ufc_form.max_cell_subdomain_id;
  if (num_cell_domains > 0)
  {
    _integrals_cell.resize(num_cell_domains + 1);
    for (std::size_t i = 0; i < num_cell_domains; ++i)
    {
      _integrals_cell[i + 1] = std::shared_ptr<ufc_cell_integral>(
          ufc_form.create_cell_integral(i));
    }
  }

  // Experimental function pointers for tabulate_tensor cell integral
  _enabled_coefficients_cell.resize(_integrals_cell.size(),
                                    ufc_form.num_coefficients);
  for (unsigned int i = 0; i < _integrals_cell.size(); ++i)
  {
    std::shared_ptr<const ufc_cell_integral> ci = _integrals_cell[i];
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
        std::shared_ptr<ufc_exterior_facet_integral>(
            _default_exterior_facet_integral));
  }

  const std::size_t num_exterior_facet_domains
      = ufc_form.max_exterior_facet_subdomain_id;
  if (num_exterior_facet_domains > 0)
  {
    _integrals_exterior_facet.resize(num_exterior_facet_domains + 1);
    for (std::size_t i = 0; i < num_exterior_facet_domains; ++i)
    {
      _integrals_exterior_facet[i + 1]
          = std::shared_ptr<ufc_exterior_facet_integral>(
              ufc_form.create_exterior_facet_integral(i));
    }
  }

  // Experimental function pointers for tabulate_tensor cell integral
  _enabled_coefficients_exterior_facet.resize(_integrals_exterior_facet.size(),
                                              ufc_form.num_coefficients);
  for (unsigned int i = 0; i < _integrals_exterior_facet.size(); ++i)
  {
    std::shared_ptr<const ufc_exterior_facet_integral> fi
        = _integrals_exterior_facet[i];
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
        std::shared_ptr<ufc_interior_facet_integral>(
            _default_interior_facet_integral));
  }

  const std::size_t num_interior_facet_domains
      = ufc_form.max_interior_facet_subdomain_id;

  if (num_interior_facet_domains > 0)
  {
    _interior_facet_integrals.resize(num_interior_facet_domains + 1);
    for (std::size_t i = 0; i < num_interior_facet_domains; ++i)
    {
      _interior_facet_integrals[i + 1]
          = std::shared_ptr<ufc_interior_facet_integral>(
              ufc_form.create_interior_facet_integral(i));
    }
  }

  // Vertex integrals
  ufc_vertex_integral* _default_vertex_integral
      = ufc_form.create_default_vertex_integral();
  if (_default_vertex_integral)
  {
    _vertex_integrals.push_back(
        std::shared_ptr<ufc_vertex_integral>(_default_vertex_integral));
  }

  const std::size_t num_vertex_domains = ufc_form.max_vertex_subdomain_id;

  if (num_vertex_domains > 0)
  {
    _vertex_integrals.resize(num_vertex_domains + 1);
    for (std::size_t i = 0; i < num_vertex_domains; ++i)
    {
      _vertex_integrals[i + 1] = std::shared_ptr<ufc_vertex_integral>(
          ufc_form.create_vertex_integral(i));
    }
  }
}
//-----------------------------------------------------------------------------
std::shared_ptr<const ufc_cell_integral> FormIntegrals::cell_integral() const
{
  if (_integrals_cell.empty())
    return std::shared_ptr<const ufc_cell_integral>();
  else
    return _integrals_cell[0];
}
//-----------------------------------------------------------------------------
std::shared_ptr<const ufc_cell_integral>
FormIntegrals::cell_integral(unsigned int i) const
{
  if ((i + 1) >= _integrals_cell.size())
    return std::shared_ptr<const ufc_cell_integral>();
  else
    return _integrals_cell[i + 1];
}
//-----------------------------------------------------------------------------
const std::function<void(PetscScalar*, const PetscScalar*, const double*, int)>&
FormIntegrals::tabulate_tensor_cell(int i) const
{
  return _tabulate_tensor_cell[i];
}
//-----------------------------------------------------------------------------
const std::function<void(PetscScalar*, const PetscScalar*, const double*, int,
                         int)>&
FormIntegrals::tabulate_tensor_exterior_facet(int i) const
{
  return _tabulate_tensor_exterior_facet[i];
}
//-----------------------------------------------------------------------------
const bool* FormIntegrals::enabled_coefficients_cell(int i) const
{
  return _enabled_coefficients_cell.row(i).data();
}
//-----------------------------------------------------------------------------
const bool* FormIntegrals::enabled_coefficients_exterior_facet(int i) const
{
  return _enabled_coefficients_exterior_facet.row(i).data();
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
int FormIntegrals::count(FormIntegrals::Type t) const
{
  switch (t)
  {
  case Type::cell:
    return _tabulate_tensor_cell.size();
  case Type::interior_facet:
    return _interior_facet_integrals.size();
  case Type::exterior_facet:
    return _integrals_exterior_facet.size();
  case Type::vertex:
    return _vertex_integrals.size();
  }

  return 0;
}
//-----------------------------------------------------------------------------
int FormIntegrals::num_cell_integrals() const
{
  return _tabulate_tensor_cell.size();
}
//-----------------------------------------------------------------------------
int FormIntegrals::num_exterior_facet_integrals() const
{
  return _integrals_exterior_facet.size();
}
//-----------------------------------------------------------------------------
std::shared_ptr<const ufc_exterior_facet_integral>
FormIntegrals::exterior_facet_integral() const
{
  if (_integrals_exterior_facet.empty())
    return std::shared_ptr<const ufc_exterior_facet_integral>();
  else
    return _integrals_exterior_facet[0];
}
//-----------------------------------------------------------------------------
std::shared_ptr<const ufc_exterior_facet_integral>
FormIntegrals::exterior_facet_integral(unsigned int i) const
{
  if (i + 1 >= _integrals_exterior_facet.size())
    return std::shared_ptr<const ufc_exterior_facet_integral>();
  else
    return _integrals_exterior_facet[i + 1];
}
//-----------------------------------------------------------------------------
std::shared_ptr<const ufc_interior_facet_integral>
FormIntegrals::interior_facet_integral() const
{
  if (_interior_facet_integrals.empty())
    return std::shared_ptr<const ufc_interior_facet_integral>();
  else
    return _interior_facet_integrals[0];
}
//-----------------------------------------------------------------------------
std::shared_ptr<const ufc_interior_facet_integral>
FormIntegrals::interior_facet_integral(unsigned int i) const
{
  if (i + 1 >= _interior_facet_integrals.size())
    return std::shared_ptr<const ufc_interior_facet_integral>();
  else
    return _interior_facet_integrals[i + 1];
}
//-----------------------------------------------------------------------------
int FormIntegrals::num_interior_facet_integrals() const
{
  return _interior_facet_integrals.size();
}
//-----------------------------------------------------------------------------
std::shared_ptr<const ufc_vertex_integral>
FormIntegrals::vertex_integral() const
{
  if (_vertex_integrals.empty())
    return std::shared_ptr<const ufc_vertex_integral>();
  else
    return _vertex_integrals[0];
}
//-----------------------------------------------------------------------------
std::shared_ptr<const ufc_vertex_integral>
FormIntegrals::vertex_integral(unsigned int i) const
{
  if (i + 1 >= _vertex_integrals.size())
    return std::shared_ptr<const ufc_vertex_integral>();
  else
    return _vertex_integrals[i + 1];
}
//-----------------------------------------------------------------------------
int FormIntegrals::num_vertex_integrals() const
{
  return _vertex_integrals.size();
}
//-----------------------------------------------------------------------------
