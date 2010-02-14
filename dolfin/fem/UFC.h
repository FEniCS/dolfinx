// Copyright (C) 2007-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2009
//
// First added:  2007-01-17
// Last changed: 2009-10-03

#ifndef __UFC_DATA_H
#define __UFC_DATA_H

#include <vector>
#include <boost/scoped_array.hpp>
#include <ufc.h>
#include "UFCCell.h"
#include "UFCMesh.h"

namespace dolfin
{

  class FiniteElement;
  class FunctionSpace;
  class GenericFunction;
  class Mesh;
  class Cell;
  class Form;

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

    // std::vector of finite elements for primary arguments
    FiniteElement** finite_elements;

    // std::vector of finite elements for coefficients
    FiniteElement** coefficient_elements;

    // std::vector of cell integrals
    ufc::cell_integral** cell_integrals;

    // std::vector of exterior facet integrals
    ufc::exterior_facet_integral** exterior_facet_integrals;

    // std::vector of interior facet integrals
    ufc::interior_facet_integral** interior_facet_integrals;

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

    // Local tensor for macro element
    boost::scoped_array<double> macro_A;

    // std::vector for local dimensions for each argument
    boost::scoped_array<uint> local_dimensions;

    // std::vector for local dimensions of macro element for primary arguments
    boost::scoped_array<uint> macro_local_dimensions;

    // std::vector of global dimensions for primary arguments
    boost::scoped_array<uint> global_dimensions;

    // std::vector of mapped dofs for primary arguments
    uint** dofs;

    // std::vector of mapped dofs of macro element for primary arguments
    uint** macro_dofs;

    // std::vector of coefficients
    double** w;

    // std::vector of coefficients on macro element
    double** macro_w;

  private:

    // Coefficients
    const std::vector<const GenericFunction*> coefficients;

    // The form
    const Form& dolfin_form;

  };
}

#endif
