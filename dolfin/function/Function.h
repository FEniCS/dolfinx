// Copyright (C) 2007-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells 2005-2008.
// Modified by Kristian B. Oelgaard, 2007.
// Modified by Martin Sandve Alnes, 2008.
//
// First added:  2003-11-28
// Last changed: 2008-09-11

#ifndef __FUNCTION_H
#define __FUNCTION_H

#include <tr1/memory>
#include <ufc.h>
#include <dolfin/common/types.h>
#include <dolfin/common/simple_array.h>
#include <dolfin/la/Vector.h>
#include <dolfin/mesh/Point.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/common/Variable.h>
#include "SubFunction.h"

namespace dolfin
{

  class Mesh;
  class Form;
  class GenericFunction;
  class DofMap;

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
    enum Type {empty, user, constant, discrete, ufc};

    /// Create empty function (read data from file)
    Function();

    /// Create user-defined function (evaluation operator must be overloaded)
    explicit Function(Mesh& mesh);

    /// Create constant scalar function from given value
    Function(Mesh& mesh, real value);

    /// Create constant vector function from given size and value
    Function(Mesh& mesh, uint size, real value);

    /// Create constant vector function from given size and values
    Function(Mesh& mesh, const Array<real>& values);

    /// Create constant tensor function from given shape and values
    Function(Mesh& mesh, const Array<uint>& shape, const Array<real>& values);

    /// Create function from given ufc::function
    Function(Mesh& mesh, const ufc::function& function, uint size);

    /// Create discrete function for argument function i of form
    Function(Mesh& mesh, Form& form, uint i = 1);

    /// Create discrete function for argument function i of form
    Function(Mesh& mesh, GenericVector& x, DofMap& dof_map, const ufc::form& form, uint i = 1);

    /// Create discrete function for argument function i of form (data may be shared)
    Function(std::tr1::shared_ptr<Mesh> mesh, std::tr1::shared_ptr<GenericVector> x, 
             std::tr1::shared_ptr<DofMap> dof_map, const ufc::form& form, uint i = 1);

    /// Create discrete function from sub function
    explicit Function(SubFunction sub_function);

    /// Create function from data file
    explicit Function(const std::string filename);

    /// Create discrete function based on signatures
    Function(std::tr1::shared_ptr<Mesh> mesh, const std::string finite_element_signature,
             const std::string dof_map_signature);
    
    /// Copy constructor
    Function(const Function& f);

    /// Destructor
    virtual ~Function();

    /// Create discrete function for argument function i of form
    void init(Mesh& mesh, Form& form, uint i = 1);

    /// Create discrete function for argument function i of form
    void init(Mesh& mesh, GenericVector& x, DofMap& dof_map, const ufc::form& form, uint i = 1);

    /// Return the type of function
    Type type() const;

    /// Return the rank of the value space
    virtual uint rank() const;

    /// Return the dimension of the value space for axis i
    virtual uint dim(uint i) const;
    
    /// Return the mesh
    Mesh& mesh() const;

    /// Return the DofMap
    DofMap& dofMap() const;

    /// Return the signature of a DiscreteFunction
    std::string signature() const;

    /// Return the vector associated with a DiscreteFunction
    GenericVector& vector() const;

    /// Return the number of sub functions (only for discrete functions)
    uint numSubFunctions() const;
    
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

    /// Evaluate function at given point (used for subclassing through SWIG interface)
    virtual void eval(simple_array<real>& values, const simple_array<real>& x) const { eval(values.data, x.data); }

    /// Evaluate function at given point (overload for scalar user-defined function)
    virtual void eval(real* values, const real* x) const;

    /// Evaluate scalar function at given point (overload for scalar user-defined function)
    virtual real eval(const real* x) const;

  protected:
    
    /// Access current cell (available during assembly for user-defined function)
    const Cell& cell() const;

    /// Access current facet normal (available during assembly for user-defined function)
    Point normal() const;

    /// Access current facet (available during assembly for user-defined functions)
    int facet() const;

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
