// Copyright (C) 2018 Chris Richardson
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfin/log/log.h>
#include <functional>
#include <iostream>
#include <memory>
#include <ufc.h>
#include <vector>

namespace dolfin
{
namespace fem
{

/// Integrals of a Form, including those defined over cells,
/// interior and exterior facets, and vertices.
class FormIntegrals
{
public:

  /// Type of integral
  enum class Type
  {
    cell,
    exterior_facet,
    interior_facet,
    vertex
  };

  /// Initialise the FormIntegrals from a ufc::form
  /// instantiating all the required integrals
  FormIntegrals(const ufc::form& ufc_form)
  {
    // Create cell integrals
    ufc::cell_integral* _default_cell_integral
        = ufc_form.create_default_cell_integral();
    if (_default_cell_integral)
    {
      std::cout << "Got a default cell integral\n";
      _cell_integrals.push_back(
          std::shared_ptr<ufc::cell_integral>(_default_cell_integral));
    }

    const std::size_t num_cell_domains = ufc_form.max_cell_subdomain_id();
    if (num_cell_domains > 0)
    {
      _cell_integrals.resize(num_cell_domains + 1);

      for (std::size_t i = 0; i < num_cell_domains; ++i)
      {
        std::cout << "Got a cell integral for domain " << i << "\n";
        _cell_integrals[i + 1] = std::shared_ptr<ufc::cell_integral>(
            ufc_form.create_cell_integral(i));
      }
    }

    // Experimental function pointers for tabulate_tensor cell integral
    for (auto& ci : _cell_integrals)
      _cell_tabulate_tensor.push_back(std::bind(
          &ufc::cell_integral::tabulate_tensor, ci, std::placeholders::_1,
          std::placeholders::_2, std::placeholders::_3, std::placeholders::_4));

    // Exterior facet integrals
    ufc::exterior_facet_integral* _default_exterior_facet_integral
        = ufc_form.create_default_exterior_facet_integral();
    if (_default_exterior_facet_integral)
    {
      std::cout << "Got a default exterior integral \n";
      _exterior_facet_integrals.push_back(
          std::shared_ptr<ufc::exterior_facet_integral>(
              _default_exterior_facet_integral));
    }

    const std::size_t num_exterior_facet_domains
        = ufc_form.max_exterior_facet_subdomain_id();

    if (num_exterior_facet_domains > 0)
    {
      _exterior_facet_integrals.resize(num_exterior_facet_domains + 1);

      for (std::size_t i = 0; i < num_exterior_facet_domains; ++i)
      {
        std::cout << "Got an exterior integral for domain " << i << "\n";
        _exterior_facet_integrals[i + 1]
            = std::shared_ptr<ufc::exterior_facet_integral>(
                ufc_form.create_exterior_facet_integral(i));
      }
    }

    // Interior facet integrals
    ufc::interior_facet_integral* _default_interior_facet_integral
        = ufc_form.create_default_interior_facet_integral();
    if (_default_interior_facet_integral)
    {
      std::cout << "Got a default interior integral \n";
      _interior_facet_integrals.push_back(
          std::shared_ptr<ufc::interior_facet_integral>(
              _default_interior_facet_integral));
    }

    const std::size_t num_interior_facet_domains
        = ufc_form.max_interior_facet_subdomain_id();

    if (num_interior_facet_domains > 0)
    {
      _interior_facet_integrals.resize(num_interior_facet_domains + 1);
      for (std::size_t i = 0; i < num_interior_facet_domains; ++i)
      {
        std::cout << "Got an interior integral for domain " << i << "\n";
        _interior_facet_integrals[i + 1]
            = std::shared_ptr<ufc::interior_facet_integral>(
                ufc_form.create_interior_facet_integral(i));
      }
    }

    // Vertex integrals
    ufc::vertex_integral* _default_vertex_integral
        = ufc_form.create_default_vertex_integral();
    if (_default_vertex_integral)
      _vertex_integrals.push_back(
          std::shared_ptr<ufc::vertex_integral>(_default_vertex_integral));

    const std::size_t num_vertex_domains = ufc_form.max_vertex_subdomain_id();

    if (num_vertex_domains > 0)
    {
      _vertex_integrals.resize(num_vertex_domains + 1);
      for (std::size_t i = 0; i < num_vertex_domains; ++i)
      {
        std::cout << "Got a vertex integral for domain " << i << "\n";
        _vertex_integrals[i + 1] = std::shared_ptr<ufc::vertex_integral>(
            ufc_form.create_vertex_integral(i));
      }
    }
  }

  /// Default cell integral
  std::shared_ptr<const ufc::cell_integral> cell_integral() const
  {
    if (_cell_integrals.empty())
      return std::shared_ptr<const ufc::cell_integral>();

    return _cell_integrals[0];
  }

  /// Cell integral for domain i
  std::shared_ptr<const ufc::cell_integral> cell_integral(unsigned int i) const
  {
    if ((i + 1) >= _cell_integrals.size())
      return std::shared_ptr<const ufc::cell_integral>();

    return _cell_integrals[i + 1];
  }

  /// Get the function for 'tabulate_tensor' for cell integral i
  const std::function<void(double*, const double* const*, const double*, int)>&
  cell_tabulate_tensor(int i) const
  {
    return _cell_tabulate_tensor[i];
  }

  /// Set the function for 'tabulate_tensor' for cell integral i
  void set_cell_tabulate_tensor(int i, void (*fn)(double*, const double* const*,
                                                  const double*, int))
  {
    _cell_tabulate_tensor.resize(i + 1);
    _cell_tabulate_tensor[i] = fn;
  }

  /// Number of integrals of given type
  unsigned int count(FormIntegrals::Type t) const
  {
    switch (t)
    {
    case Type::cell:
      return _cell_integrals.size();
    case Type::interior_facet:
      return _interior_facet_integrals.size();
    case Type::exterior_facet:
      return _exterior_facet_integrals.size();
    case Type::vertex:
      return _vertex_integrals.size();
    }
    return 0;
  }

  /// Number of cell integrals
  unsigned int num_cell_integrals() const { return _cell_integrals.size(); }

  /// Default exterior facet integral
  std::shared_ptr<const ufc::exterior_facet_integral>
  exterior_facet_integral() const
  {
    if (_exterior_facet_integrals.empty())
      return std::shared_ptr<const ufc::exterior_facet_integral>();

    return _exterior_facet_integrals[0];
  }

  /// Exterior facet integral for domain i
  std::shared_ptr<const ufc::exterior_facet_integral>
  exterior_facet_integral(unsigned int i) const
  {
    if (i + 1 >= _exterior_facet_integrals.size())
      return std::shared_ptr<const ufc::exterior_facet_integral>();

    return _exterior_facet_integrals[i + 1];
  }

  /// Number of exterior facet integrals
  unsigned int num_exterior_facet_integrals() const
  {
    return _exterior_facet_integrals.size();
  }

  /// Default interior facet integral
  std::shared_ptr<const ufc::interior_facet_integral>
  interior_facet_integral() const
  {
    if (_interior_facet_integrals.empty())
      return std::shared_ptr<const ufc::interior_facet_integral>();

    return _interior_facet_integrals[0];
  }

  /// Interior facet integral for domain i
  std::shared_ptr<const ufc::interior_facet_integral>
  interior_facet_integral(unsigned int i) const
  {
    if (i + 1 >= _interior_facet_integrals.size())
      return std::shared_ptr<const ufc::interior_facet_integral>();

    return _interior_facet_integrals[i + 1];
  }

  /// Number of interior facet integrals
  unsigned int num_interior_facet_integrals() const
  {
    return _interior_facet_integrals.size();
  }

  /// Default interior facet integral
  std::shared_ptr<const ufc::vertex_integral> vertex_integral() const
  {
    if (_vertex_integrals.empty())
      return std::shared_ptr<const ufc::vertex_integral>();

    return _vertex_integrals[0];
  }

  /// Interior facet integral for domain i
  std::shared_ptr<const ufc::vertex_integral>
  vertex_integral(unsigned int i) const
  {
    if (i + 1 >= _vertex_integrals.size())
      return std::shared_ptr<const ufc::vertex_integral>();
    return _vertex_integrals[i + 1];
  }

  /// Number of vertex integrals
  unsigned int num_vertex_integrals() const { return _vertex_integrals.size(); }

private:
  // Cell integrals
  std::vector<std::shared_ptr<ufc::cell_integral>> _cell_integrals;

  // Function pointers to cell tabulate_tensor functions
  std::vector<
      std::function<void(double*, const double* const*, const double*, int)>>
      _cell_tabulate_tensor;

  // Exterior facet integrals
  std::vector<std::shared_ptr<ufc::exterior_facet_integral>>
      _exterior_facet_integrals;
  // Interior facet integrals
  std::vector<std::shared_ptr<ufc::interior_facet_integral>>
      _interior_facet_integrals;
  // Vertex integrals
  std::vector<std::shared_ptr<ufc::vertex_integral>> _vertex_integrals;
};
}
}
