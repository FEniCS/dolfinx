// Copyright (C) 2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-07-11
// Last changed: 2007-07-11

#ifndef __BOUNDARY_CONDITION_H
#define __BOUNDARY_CONDITION_H

#include <ufc.h>
#include <dolfin/constants.h>

namespace dolfin
{

  class GenericMatrix;
  class GenericVector;
  class Form;
  class SubSystem;
  class Mesh;

  /// Common base class for boundary conditions

  class BoundaryCondition
  {
  public:

    /// Constructor
    BoundaryCondition();

    /// Destructor
    virtual ~BoundaryCondition();

    /// Apply boundary condition to linear system
    virtual void apply(GenericMatrix& A, GenericVector& b, const Form& form) = 0;

    /// Apply boundary condition to linear system for a nonlinear problem
    virtual void apply(GenericMatrix& A, GenericVector& b, const GenericVector& x, const Form& form) = 0;

  protected:

    // Local data for application of boundary conditions
    class LocalData
    {
    public:
      
      // Constructor
      LocalData(const Form& form, Mesh& mesh, const SubSystem& sub_system);
      
      // Destructor
      ~LocalData();

      // Finite element for sub system
      ufc::finite_element* finite_element;
      
      // Dof map for sub system
      ufc::dof_map* dof_map;

      // Offset for sub system
      uint offset;
      
      // Local data used to set boundary conditions
      real* w;
      uint* cell_dofs;
      real* values;
      real* x_values;
      uint* facet_dofs;
      uint* rows;
      real** coordinates;

    };

  };

}

#endif
