// Copyright (C) 2007 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2007-04-02
// Last changed: 2007-04-03

#ifndef __NEW_DISCRETE_FUNCTION_H
#define __NEW_DISCRETE_FUNCTION_H

#include <dolfin/constants.h>
#include <dolfin/NewForm.h>
#include <dolfin/Vector.h>
#include <dolfin/UFCMesh.h>
#include <dolfin/GenericFunction.h>

namespace dolfin
{

  class Mesh;
  class DofMap;

  /// A DiscreteFunction is a function defined in terms of a vector
  /// of degrees of freedom, a finite element and a dof map (including
  /// the mesh on which the dof map is defined). The finite element
  /// determines how the function is defined locally on each cell of
  /// the mesh in terms of the local degrees of freedom, and the dof
  /// map determines how the degrees of freedom are distributed on the
  /// mesh.

  class NewDiscreteFunction
  {
  public:

    /// Create discrete function for argument function i of form
    NewDiscreteFunction(Mesh& mesh, const NewForm& form, uint i);

    /// Destructor
    ~NewDiscreteFunction();

    /// Interpolate function on cell
    void interpolate(real* coefficients,
                     const ufc::cell& cell,
                     const ufc::finite_element& finite_element);

  private:

    // The vector of dofs
    Vector x;

    // The finite element
    ufc::finite_element* finite_element;

    // The dof map
    DofMap* dof_map;

    // Local array for mapping of dofs
    uint* dofs;

  };

}

#endif
