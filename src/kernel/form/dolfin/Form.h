// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __FORM_H
#define __FORM_H

#include <dolfin/constants.h>

namespace dolfin
{

  class Cell;
  class NewFiniteElement;
  
  class Form
  {
  public:

    /// Constructor
    Form(const NewFiniteElement& element);

    /// Destructor
    virtual ~Form();

    /// Update map to current cell
    void update(const Cell& cell);

    /// Friends
    friend class NewFEM;

  protected:

    // Update affine map from reference triangle
    void updateTriLinMap(const Cell& cell);
    
    // Update affine map from reference tetrahedron
    void updateTetLinMap(const Cell& cell);

    // The finite element
    const NewFiniteElement& element;

    // Determinant of Jacobian of map
    real det;

    // Jacobian of map
    real f00, f01, f02, f10, f11, f12, f20, f21, f22;

    // Inverse of Jacobian of map
    real g00, g01, g02, g10, g11, g12, g20, g21, g22;

  };

}

#endif
