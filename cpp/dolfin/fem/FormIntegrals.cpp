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
  // Create cell integrals
  ufc_cell_integral* _default_cell_integral
      = ufc_form.create_default_cell_integral();
  if (_default_cell_integral)
  {
    _cell_integrals.push_back(
        std::shared_ptr<ufc_cell_integral>(_default_cell_integral));
  }

  const std::size_t num_cell_domains = ufc_form.max_cell_subdomain_id;
  if (num_cell_domains > 0)
  {
    _cell_integrals.resize(num_cell_domains + 1);

    for (std::size_t i = 0; i < num_cell_domains; ++i)
    {
      _cell_integrals[i + 1] = std::shared_ptr<ufc_cell_integral>(
          ufc_form.create_cell_integral(i));
    }
  }

  _cell_batch_size.resize(_cell_integrals.size(), Eigen::NoChange);
  _enabled_coefficients.resize(_cell_integrals.size(),
                               ufc_form.num_coefficients);

  // Experimental function pointers for tabulate_tensor cell integral
  for (unsigned int i = 0; i != _cell_integrals.size(); ++i)
  {
    const auto ci = _cell_integrals[i];
    _cell_tabulate_tensor.push_back(ci->tabulate_tensor);
    _cell_batch_size[i] = ci->cell_batch_size;
    std::copy(ci->enabled_coefficients,
              ci->enabled_coefficients + ufc_form.num_coefficients,
              _enabled_coefficients.row(i).data());
  }

  // Exterior facet integrals
  ufc_exterior_facet_integral* _default_exterior_facet_integral
      = ufc_form.create_default_exterior_facet_integral();
  if (_default_exterior_facet_integral)
  {
    _exterior_facet_integrals.push_back(
        std::shared_ptr<ufc_exterior_facet_integral>(
            _default_exterior_facet_integral));
  }

  const std::size_t num_exterior_facet_domains
      = ufc_form.max_exterior_facet_subdomain_id;

  if (num_exterior_facet_domains > 0)
  {
    _exterior_facet_integrals.resize(num_exterior_facet_domains + 1);

    for (std::size_t i = 0; i < num_exterior_facet_domains; ++i)
    {
      _exterior_facet_integrals[i + 1]
          = std::shared_ptr<ufc_exterior_facet_integral>(
              ufc_form.create_exterior_facet_integral(i));
    }
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
  if (_cell_integrals.empty())
    return std::shared_ptr<const ufc_cell_integral>();
  else
    return _cell_integrals[0];
}
//-----------------------------------------------------------------------------
std::shared_ptr<const ufc_cell_integral>
FormIntegrals::cell_integral(unsigned int i) const
{
  if ((i + 1) >= _cell_integrals.size())
    return std::shared_ptr<const ufc_cell_integral>();
  else
    return _cell_integrals[i + 1];
}
//-----------------------------------------------------------------------------
const std::function<void(PetscScalar*, const PetscScalar* const*, const double*,
                         int)>&
FormIntegrals::cell_tabulate_tensor(int i) const
{
  return _cell_tabulate_tensor[i];
}
//-----------------------------------------------------------------------------
unsigned int FormIntegrals::cell_batch_size(int i) const
{
  return _cell_batch_size[i];
}
//-----------------------------------------------------------------------------
const bool* FormIntegrals::cell_enabled_coefficients(int i) const
{
  return _enabled_coefficients.row(i).data();
}
//-----------------------------------------------------------------------------
void FormIntegrals::set_cell_tabulate_tensor(
    int i,
    void (*fn)(PetscScalar*, const PetscScalar* const*, const double*, int),
    unsigned int cell_batch_size)
{
  _cell_tabulate_tensor.resize(i + 1);
  _cell_tabulate_tensor[i] = fn;

  // Enable all coefficients for this integral
  _enabled_coefficients.conservativeResize(i + 1, Eigen::NoChange);
  _enabled_coefficients.row(i) = true;

  _cell_batch_size.conservativeResize(i + 1, Eigen::NoChange);
  _cell_batch_size.row(i) = cell_batch_size;
}
//-----------------------------------------------------------------------------
int FormIntegrals::count(FormIntegrals::Type t) const
{
  switch (t)
  {
  case Type::cell:
    return _cell_tabulate_tensor.size();
  case Type::interior_facet:
    return _interior_facet_integrals.size();
  case Type::exterior_facet:
    return _exterior_facet_integrals.size();
  case Type::vertex:
    return _vertex_integrals.size();
  }

  return 0;
}
//-----------------------------------------------------------------------------
int FormIntegrals::num_cell_integrals() const
{
  return _cell_tabulate_tensor.size();
}
//-----------------------------------------------------------------------------
std::shared_ptr<const ufc_exterior_facet_integral>
FormIntegrals::exterior_facet_integral() const
{
  if (_exterior_facet_integrals.empty())
    return std::shared_ptr<const ufc_exterior_facet_integral>();
  else
    return _exterior_facet_integrals[0];
}
//-----------------------------------------------------------------------------
std::shared_ptr<const ufc_exterior_facet_integral>
FormIntegrals::exterior_facet_integral(unsigned int i) const
{
  if (i + 1 >= _exterior_facet_integrals.size())
    return std::shared_ptr<const ufc_exterior_facet_integral>();
  else
    return _exterior_facet_integrals[i + 1];
}
//-----------------------------------------------------------------------------
int FormIntegrals::num_exterior_facet_integrals() const
{
  return _exterior_facet_integrals.size();
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
