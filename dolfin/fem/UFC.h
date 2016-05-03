// Copyright (C) 2007-2015 Anders Logg
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Garth N. Wells 2009
//
// First added:  2007-01-17
// Last changed: 2015-10-23

#ifndef __UFC_DATA_H
#define __UFC_DATA_H

#include <vector>
#include <memory>
#include <ufc.h>

namespace dolfin
{

  class Cell;
  class FiniteElement;
  class Form;
  class FunctionSpace;
  class GenericFunction;
  class Mesh;

  /// This class is a simple data structure that holds data used
  /// during assembly of a given UFC form. Data is created for each
  /// primary argument, that is, v_j for j < r. In addition, nodal
  /// basis expansion coefficients and a finite element are created
  /// for each coefficient function.

  class UFC
  {
  public:

    /// Constructor
    UFC(const Form& form);

    /// Copy constructor
    UFC(const UFC& ufc);

    /// Destructor
    ~UFC();

    /// Initialise memory
    void init(const Form& form);

    /// Update current cell
    void update(const Cell& cell,
                const std::vector<double>& coordinate_dofs0,
                const ufc::cell& ufc_cell,
                const std::vector<bool> & enabled_coefficients);

    /// Update current pair of cells for macro element
    void update(const Cell& cell0,
                const std::vector<double>& coordinate_dofs0,
                const ufc::cell& ufc_cell0,
                const Cell& cell1,
                const std::vector<double>& coordinate_dofs1,
                const ufc::cell& ufc_cell1,
                const std::vector<bool> & enabled_coefficients);

    /// Update current cell (TODO: Remove this when
    /// PointIntegralSolver supports the version with
    /// enabled_coefficients)
    void update(const Cell& cell,
                const std::vector<double>& coordinate_dofs0,
                const ufc::cell& ufc_cell);

    /// Update current pair of cells for macro element (TODO: Remove
    /// this when PointIntegralSolver supports the version with
    /// enabled_coefficients)
    void update(const Cell& cell0,
                const std::vector<double>& coordinate_dofs0,
                const ufc::cell& ufc_cell0,
                const Cell& cell1,
                const std::vector<double>& coordinate_dofs1,
                const ufc::cell& ufc_cell1);

    /// Pointer to coefficient data. Used to support UFC interface.
    const double* const * w() const
    { return w_pointer.data(); }

    /// Pointer to coefficient data. Used to support UFC
    /// interface. None const version
    double* * w()
    { return w_pointer.data(); }

    /// Pointer to macro element coefficient data. Used to support UFC
    /// interface.
    const double* const * macro_w() const
    { return macro_w_pointer.data(); }

  private:

    // Finite elements for coefficients
    std::vector<FiniteElement> coefficient_elements;

    // Cell integrals (access through get_cell_integral to get proper
    // fallback to default)
    std::vector<std::shared_ptr<ufc::cell_integral>>
      cell_integrals;

    // Exterior facet integrals (access through
    // get_exterior_facet_integral to get proper fallback to default)
    std::vector<std::shared_ptr<ufc::exterior_facet_integral>>
      exterior_facet_integrals;

    // Interior facet integrals (access through
    // get_interior_facet_integral to get proper fallback to default)
    std::vector<std::shared_ptr<ufc::interior_facet_integral>>
      interior_facet_integrals;

    // Point integrals (access through get_vertex_integral to get
    // proper fallback to default)
    std::vector<std::shared_ptr<ufc::vertex_integral>>
      vertex_integrals;

    // Custom integrals (access through get_custom_integral to get
    // proper fallback to default)
    std::vector<std::shared_ptr<ufc::custom_integral>> custom_integrals;

    // Cutcell integrals (access through get_cutcell_integral to get
    // proper fallback to default)
    std::vector<std::shared_ptr<ufc::cutcell_integral>> cutcell_integrals;

    // Interface integrals (access through get_interface_integral to
    // get proper fallback to default)
    std::vector<std::shared_ptr<ufc::interface_integral>> interface_integrals;

    // Overlap integrals (access through get_overlap_integral to get
    // proper fallback to default)
    std::vector<std::shared_ptr<ufc::overlap_integral>> overlap_integrals;

  public:

    // Default cell integral
    std::shared_ptr<ufc::cell_integral>
      default_cell_integral;

    // Default exterior facet integral
    std::shared_ptr<ufc::exterior_facet_integral>
      default_exterior_facet_integral;

    // Default interior facet integral
    std::shared_ptr<ufc::interior_facet_integral>
      default_interior_facet_integral;

    // Default point integral
    std::shared_ptr<ufc::vertex_integral>
      default_vertex_integral;

    // Default custom integral
    std::shared_ptr<ufc::custom_integral> default_custom_integral;

    // Default cutcell integral
    std::shared_ptr<ufc::cutcell_integral> default_cutcell_integral;

    // Default interface integral
    std::shared_ptr<ufc::interface_integral> default_interface_integral;

    // Default overlap integral
    std::shared_ptr<ufc::overlap_integral> default_overlap_integral;

    /// Get cell integral over a given domain, falling back to the
    /// default if necessary
    ufc::cell_integral*
      get_cell_integral(std::size_t domain)
    {
      if (domain < form.max_cell_subdomain_id())
      {
        ufc::cell_integral * integral
          = cell_integrals[domain].get();
        if (integral)
          return integral;
      }
      return default_cell_integral.get();
    }

    /// Get exterior facet integral over a given domain, falling back
    /// to the default if necessary
    ufc::exterior_facet_integral*
      get_exterior_facet_integral(std::size_t domain)
    {
      if (domain < form.max_exterior_facet_subdomain_id())
      {
        ufc::exterior_facet_integral* integral
          = exterior_facet_integrals[domain].get();
        if (integral)
          return integral;
      }
      return default_exterior_facet_integral.get();
    }

    /// Get interior facet integral over a given domain, falling back
    /// to the default if necessary
    ufc::interior_facet_integral*
      get_interior_facet_integral(std::size_t domain)
    {
      if (domain < form.max_interior_facet_subdomain_id())
      {
        ufc::interior_facet_integral* integral
          = interior_facet_integrals[domain].get();
        if (integral)
          return integral;
      }
      return default_interior_facet_integral.get();
    }

    /// Get point integral over a given domain, falling back to the
    /// default if necessary
    ufc::vertex_integral*
      get_vertex_integral(std::size_t domain)
    {
      if (domain < form.max_vertex_subdomain_id())
      {
        ufc::vertex_integral * integral
          = vertex_integrals[domain].get();
        if (integral)
          return integral;
      }
      return default_vertex_integral.get();
    }

    /// Get custom integral over a given domain, falling back to the
    /// default if necessary
    ufc::custom_integral * get_custom_integral(std::size_t domain)
    {
      if (domain < form.max_custom_subdomain_id())
      {
        ufc::custom_integral * integral = custom_integrals[domain].get();
        if (integral)
          return integral;
      }
      return default_custom_integral.get();
    }

    /// Get cutcell integral over a given domain, falling back to the
    /// default if necessary
    ufc::cutcell_integral * get_cutcell_integral(std::size_t domain)
    {
      if (domain < form.max_cutcell_subdomain_id())
      {
        ufc::cutcell_integral * integral = cutcell_integrals[domain].get();
        if (integral)
          return integral;
      }
      return default_cutcell_integral.get();
    }

    /// Get interface integral over a given domain, falling back to
    /// the default if necessary
    ufc::interface_integral * get_interface_integral(std::size_t domain)
    {
      if (domain < form.max_interface_subdomain_id())
      {
        ufc::interface_integral * integral = interface_integrals[domain].get();
        if (integral)
          return integral;
      }
      return default_interface_integral.get();
    }

    /// Get overlap integral over a given domain, falling back to the
    /// default if necessary
    ufc::overlap_integral * get_overlap_integral(std::size_t domain)
    {
      if (domain < form.max_overlap_subdomain_id())
      {
        ufc::overlap_integral * integral = overlap_integrals[domain].get();
        if (integral)
          return integral;
      }
      return default_overlap_integral.get();
    }

    // Form
    const ufc::form& form;

    // FIXME AL: Check which data is actually used and remove the rest

    // Local tensor
    std::vector<double> A;

    // Local tensor
    std::vector<double> A_facet;

    // Local tensor for macro element
    std::vector<double> macro_A;

  private:

    // Coefficients (std::vector<double*> is used to interface with
    // UFC)
    std::vector<std::vector<double>> _w;
    std::vector<double*> w_pointer;

    // Coefficients on macro element (std::vector<double*> is used to
    // interface with UFC)
    std::vector<std::vector<double>> _macro_w;
    std::vector<double*> macro_w_pointer;

    // Coefficient functions
    const std::vector<std::shared_ptr<const GenericFunction>> coefficients;

  public:

    /// The form
    const Form& dolfin_form;

  };
}

#endif
