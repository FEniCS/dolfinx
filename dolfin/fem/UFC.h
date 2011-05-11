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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Garth N. Wells, 2009
//
// First added:  2007-01-17
// Last changed: 2011-01-31

#ifndef __UFC_DATA_H
#define __UFC_DATA_H

#include <vector>
#include <boost/scoped_array.hpp>
#include <ufc.h>
#include "UFCCell.h"
#include "UFCMesh.h"

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
    void update(const Cell& cell, uint local_facet);

    /// Update current pair of cells for macro element
    void update(const Cell& cell0, uint local_facet0,
                const Cell& cell1, uint local_facet1);

    private:

    // Finite elements for coefficients
    std::vector<FiniteElement> coefficient_elements;

    public:

    // Cell integrals
    std::vector<boost::shared_ptr<ufc::cell_integral> > cell_integrals;

    // Exterior facet integrals
    std::vector<boost::shared_ptr<ufc::exterior_facet_integral> > exterior_facet_integrals;

    // Interior facet integrals
    std::vector<boost::shared_ptr<ufc::interior_facet_integral> > interior_facet_integrals;

    // Form
    const ufc::form& form;

    // Mesh
    UFCMesh mesh;

    // Current cell
    UFCCell cell;

    // Current pair of cells of macro element
    UFCCell cell0;
    UFCCell cell1;

    // Local tensor
    boost::scoped_array<double> A;

    // Local tensor
    boost::scoped_array<double> A_facet;

    // Local tensor for macro element
    boost::scoped_array<double> macro_A;

    // Coefficients
    double** w;

    // Coefficients on macro element
    double** macro_w;

  private:

    // Coefficients
    const std::vector<boost::shared_ptr<const GenericFunction> > coefficients;

    // The form
    const Form& dolfin_form;

  };
}

#endif
