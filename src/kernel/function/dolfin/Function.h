// Copyright (C) 2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells 2005.
// Modified by Kristian B. Oelgaard, 2007.

// First added:  2003-11-28
// Last changed: 2007-05-29

#ifndef __FUNCTION_H
#define __FUNCTION_H

#include <ufc.h>
#include <dolfin/constants.h>
#include <dolfin/Vector.h>
#include <dolfin/SubFunction.h>
#include <dolfin/Variable.h>

namespace dolfin
{

  class Mesh;
  class Cell;
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
    enum Type {empty, user, constant, discrete};

    /// Create empty function (read data from file)
    Function();

    /// Create user-defined function (evaluation operator must be overloaded)
    Function(Mesh& mesh);

    /// Create constant function from given value
    Function(Mesh& mesh, real value);

    /// Create discrete function for argument function i of form
    Function(Mesh& mesh, Vector& x, const Form& form, uint i = 1);

    /// Create discrete function from sub function
    Function(SubFunction sub_function);

    /// Create function from data file
    Function(const std::string filename);

    /// Destructor
    virtual ~Function();

    /// Create discrete function for argument function i of form
    void init(Mesh& mesh, Vector& x, const Form& form, uint i = 1);

    /// Return the type of function
    Type type() const;

    /// Return the rank of the value space
    uint rank() const;

    /// Return the dimension of the value space for axis i
    uint dim(unsigned int i) const;
    
    /// Return the mesh
    Mesh& mesh();

    /// Extract sub function/slice (only for discrete function)
    SubFunction operator[] (uint i);

    /// Assign function
    const Function& operator= (Function& f);

    /// Assign sub function/slice
    const Function& operator= (SubFunction f);
    
    /// Interpolate function to vertices of mesh
    void interpolate(real* values);

    /// Interpolate function to finite element space on cell
    void interpolate(real* coefficients,
                     const ufc::cell& ufc_cell,
                     const ufc::finite_element& finite_element,
                     Cell& cell, int facet = -1);

    /// Evaluate function at given point (overload for user-defined function)
    virtual void eval(real* values, const real* x) const;

    /// Evaluate scalar function at given point (overload for scalar user-defined function)
    virtual real eval(const real* x) const;

    /// Friends
    friend class XMLFile;

  protected:
    
    // Access current cell (available during assembly for user-defined function)
    const Cell& cell() const;

    // Access current facet (available during assembly for user-defined functions)
    int facet() const;

    /// Return the mesh
    Mesh& mesh() const;

  private:
    
    // Pointer to current implementation (letter base class)
    GenericFunction* f;

    // Type of function
    Type _type;
    
    // Pointer to current cell (if any, otherwise 0)
    Cell* _cell;

    // Current facet (if any, otherwise -1)
    int _facet;

  };

}

#endif
