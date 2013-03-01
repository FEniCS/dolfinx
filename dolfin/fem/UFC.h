// Copyright (C) 2007-2008 Anders Logg
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
// Modified by Garth N. Wells, 2009
//
// First added:  2007-01-17
// Last changed: 2011-01-31

#ifndef __UFC_DATA_H
#define __UFC_DATA_H

#include <vector>
#include <boost/shared_ptr.hpp>
#include <ufc.h>
#include "UFCCell.h"

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
    void update(const Cell& cell);

    /// Update current cell and facet
    void update(const Cell& cell, std::size_t local_facet);

    /// Update current pair of cells for macro element
    void update(const Cell& cell0, std::size_t local_facet0,
                const Cell& cell1, std::size_t local_facet1);

    /// Pointer to coefficient data. Used to support UFC interface.
    const double* const * w() const
    { return &w_pointer[0]; }

    /// Pointer to macro element coefficient data. Used to support UFC interface.
    const double* const * macro_w() const
    { return &macro_w_pointer[0]; }

  private:

    // Finite elements for coefficients
    std::vector<FiniteElement> coefficient_elements;

    // Cell integrals (access through get_cell_integral to get proper fallback to default)
    std::vector<boost::shared_ptr<ufc::cell_integral> > cell_integrals;

    // Exterior facet integrals (access through get_exterior_facet_integral to get proper fallback to default)
    std::vector<boost::shared_ptr<ufc::exterior_facet_integral> > exterior_facet_integrals;

    // Interior facet integrals (access through get_interior_facet_integral to get proper fallback to default)
    std::vector<boost::shared_ptr<ufc::interior_facet_integral> > interior_facet_integrals;

    // Point integrals (access through get_point_integral to get proper fallback to default)
    std::vector<boost::shared_ptr<ufc::point_integral> > point_integrals;

  public:

    // Default cell integral
    boost::shared_ptr<ufc::cell_integral> default_cell_integral;

    // Default exterior facet integral
    boost::shared_ptr<ufc::exterior_facet_integral> default_exterior_facet_integral;

    // Default interior facet integral
    boost::shared_ptr<ufc::interior_facet_integral> default_interior_facet_integral;

    // Default point integral
    boost::shared_ptr<ufc::point_integral> default_point_integral;

    /// Get cell integral over a given domain, falling back to the default if necessary
    ufc::cell_integral * get_cell_integral(std::size_t domain)
    {
      if (domain < form.num_cell_domains())
      {
        ufc::cell_integral * integral = cell_integrals[domain].get();
        if (integral)
          return integral;
      }
      return default_cell_integral.get();
    }

    /// Get exterior facet integral over a given domain, falling back to the 
    /// default if necessary
    ufc::exterior_facet_integral * get_exterior_facet_integral(std::size_t domain)
    {
      if (domain < form.num_exterior_facet_domains())
      {
        ufc::exterior_facet_integral * integral = exterior_facet_integrals[domain].get();
        if (integral)
          return integral;
      }
      return default_exterior_facet_integral.get();
    }

    /// Get interior facet integral over a given domain, falling back to the 
    /// default if necessary
    ufc::interior_facet_integral * get_interior_facet_integral(std::size_t domain)
    {
      if (domain < form.num_interior_facet_domains())
      {
        ufc::interior_facet_integral * integral = interior_facet_integrals[domain].get();
        if (integral)
          return integral;
      }
      return default_interior_facet_integral.get();
    }

    /// Get point integral over a given domain, falling back to the 
    /// default if necessary
    ufc::point_integral * get_point_integral(std::size_t domain)
    {
      if (domain < form.num_point_domains())
      {
        ufc::point_integral * integral = point_integrals[domain].get();
        if (integral)
          return integral;
      }
      return default_point_integral.get();
    }

    // Form
    const ufc::form& form;


    // FIXME AL: Check which data is actually used and remove the rest
    // FIXME AL: Remove UFCCell class


    // Current cell
    UFCCell cell;

    // Current pair of cells of macro element
    UFCCell cell0;
    UFCCell cell1;

    // Local tensor
    std::vector<double> A;

    // Local tensor
    std::vector<double> A_facet;

    // Local tensor for macro element
    std::vector<double> macro_A;

  private:

    // Coefficients (std::vector<double*> is used to interface with UFC)
    std::vector<std::vector<double> > _w;
    std::vector<double*> w_pointer;

    // Coefficients on macro element (std::vector<double*> is used to interface with UFC)
    std::vector<std::vector<double> > _macro_w;
    std::vector<double*> macro_w_pointer;

    // Coefficient functions
    const std::vector<boost::shared_ptr<const GenericFunction> > coefficients;

    // The form
    const Form& dolfin_form;

  };
}

#endif
