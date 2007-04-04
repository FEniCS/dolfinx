// Copyright (C) 2003-2007 Johan Hoffman, Johan Jansson and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Garth N. Wells 2005.
//
// First added:  2003-11-28
// Last changed: 2007-04-04

#ifndef __FUNCTION_H
#define __FUNCTION_H

#include <ufc.h>

#include <dolfin/constants.h>
#include <dolfin/Variable.h>
#include <dolfin/FunctionPointer.h>

namespace dolfin
{

  class Mesh;
  class Form;
  class GenericFunction;

  /// This class represents a function that can be evaluated on a
  /// mesh. The actual representation of the function can vary, but
  /// the typical representation is in terms of a mesh, a vector of
  /// degrees of freedom, a finite element and a dof map that
  /// determines the distribution of degrees of freedom on the mesh.
  ///
  /// It is also possible to have user-defined functions, either by
  /// overloading the eval function of this class or by giving a
  /// function (pointer) that returns the value of the function.

  class Function : public Variable
  {
  public:

    /// Create user-defined function (evaluation operator must be overloaded)
    Function();

    /// Create function from function pointer
    Function(FunctionPointer fp);

    /// Create constant function from given value
    Function(real value);

    /// Create discrete function for argument function i of form
    Function(Mesh& mesh, const Form& form, uint i);

    /// Destructor
    virtual ~Function();

    /// Evaluate function at given point (must be implemented for user-defined function)
    virtual void eval(real* values, const real* coordinates);
    
    /// Interpolate function on cell
    void interpolate(real* coefficients,
                     const ufc::cell& cell,
                     const ufc::finite_element& finite_element);

  private:
    
    // Pointer to current implementation (letter base class)
    GenericFunction* f;

  };

}

#endif
