// Copyright (C) 2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells 2007
//
// First added:  2007-07-11
// Last changed: 2007-12-08

#ifndef __BOUNDARY_CONDITION_H
#define __BOUNDARY_CONDITION_H

#include <ufc.h>
#include <dolfin/constants.h>
#include <dolfin/UFCMesh.h>
#include <dolfin/DofMap.h>

namespace dolfin
{

  class DofMap;
  class GenericMatrix;
  class GenericVector;
  class SubSystem;
  class Mesh;
  class Form;

  /// Common base class for boundary conditions

  class BoundaryCondition
  {
  public:

    /// Constructor
    BoundaryCondition();

    /// Destructor
    virtual ~BoundaryCondition();

    /// Apply boundary condition to linear system
    virtual void apply(GenericMatrix& A, GenericVector& b, DofMap& dof_map, const Form& form) = 0;

    /// Apply boundary condition to linear system
    virtual void apply(GenericMatrix& A, GenericVector& b, DofMap& dof_map, const ufc::form& form) = 0;

    /// Apply boundary condition to linear system for a nonlinear problem
    virtual void apply(GenericMatrix& A, GenericVector& b, const GenericVector& x, DofMap& dof_map, const Form& form) = 0;

    /// Apply boundary condition to linear system for a nonlinear problem
    virtual void apply(GenericMatrix& A, GenericVector& b, const GenericVector& x, DofMap& dof_map, const ufc::form& form) = 0;

  protected:

    // Local data for application of boundary conditions
    class LocalData
    {
    public:
      
      // Constructor
      LocalData(const ufc::form& form, Mesh& mesh, DofMap& dof_map, const SubSystem& sub_system);
      
      // Destructor
      ~LocalData();
      
      // UFC view of mesh
      UFCMesh ufc_mesh;
      
      // Finite element for sub system
      ufc::finite_element* finite_element;
      
      // Dof map for sub system
      DofMap& dof_map;

      // Offset for sub system
      uint offset;
      
      // Local data used to set boundary conditions
      real* w;
      uint* cell_dofs;
      uint* facet_dofs;
      real** coordinates;

    };

  };

}

#endif
