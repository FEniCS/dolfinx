// Copyright (C) 2003-2007 Johan Hoffman, Johan Jansson and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Garth N. Wells 2005.
//
// First added:  2003-11-28
// Last changed: 2007-04-12

#ifndef __FUNCTION_H
#define __FUNCTION_H

#include <ufc.h>
#include <dolfin/constants.h>
#include <dolfin/Vector.h>
#include <dolfin/Variable.h>

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

    /// Function types
    enum Type {user, constant, discrete};

    /// Create user-defined function (evaluation operator must be overloaded)
    Function(Mesh& mesh);

    /// Create constant function from given value
    Function(Mesh& mesh, real value);

    /// Create discrete function for argument function i of form
    Function(Mesh& mesh, Vector& x, const Form& form, uint i = 1);

    /// Destructor
    virtual ~Function();

    /// Return the type of function
    Type type() const;

    /// Return the rank of the value space
    uint rank() const;

    /// Return the dimension of the value space for axis i
    uint dim(unsigned int i) const;
    
    /// Return the mesh
    Mesh& mesh();

    /// Interpolate function to vertices of mesh
    void interpolate(real* values);

    /// Interpolate function to finite element space on cell
    void interpolate(real* coefficients,
                     const ufc::cell& cell,
                     const ufc::finite_element& finite_element);

    /// Evaluate function at given point (overload for user-defined function)
    virtual void eval(real* values, const real* x);

    /// Evaluate scalar function at given point (overload for scalar user-defined function)
    virtual real eval(const real* x);

    /// Friends
    friend class XMLFile;

  private:
    
    // Pointer to current implementation (letter base class)
    GenericFunction* f;

    // Type of function
    Type _type;

  };

}

#endif
