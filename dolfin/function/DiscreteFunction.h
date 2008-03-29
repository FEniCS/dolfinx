// Copyright (C) 2007-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2007.
//
// First added:  2007-04-02
// Last changed: 2008-03-17

#ifndef __DISCRETE_FUNCTION_H
#define __DISCRETE_FUNCTION_H

#include <dolfin/la/Vector.h>
#include "GenericFunction.h"

namespace dolfin
{

  class Mesh;
  class Form;
  class DofMap;
  class SubFunction;
  class IntersectionDetector;

  /// This class implements the functionality for discrete functions.
  /// A discrete function is defined in terms of a mesh, a vector of
  /// degrees of freedom, a finite element and a dof map. The finite
  /// element determines how the function is defined locally on each
  /// cell of the mesh in terms of the local degrees of freedom, and
  /// the dof map determines how the degrees of freedom are
  /// distributed on the mesh.

  class DiscreteFunction : public GenericFunction
  {
  public:

    /// Create discrete function for argument function i of form
    DiscreteFunction(Mesh& mesh, GenericVector& x, Form& form, uint i);

    /// Create discrete function for argument function i of form
    DiscreteFunction(Mesh& mesh, GenericVector& x, DofMap& dof_map, const ufc::form& form, uint i);

    /// Create discrete function from given data and assume responsibility for data
    DiscreteFunction(Mesh& mesh, GenericVector& x, std::string finite_element_signature, std::string dof_map_signature);

    /// Create discrete function from sub function
    DiscreteFunction(SubFunction& sub_function);

    /// Copy constructor
    DiscreteFunction(const DiscreteFunction& f);
    
    /// Destructor
    ~DiscreteFunction();

    /// Return the rank of the value space
    uint rank() const;

    /// Return the dimension of the value space for axis i
    uint dim(uint i) const;

    /// Return the number of sub functions
    uint numSubFunctions() const;

    /// Assign discrete function
    const DiscreteFunction& operator= (const DiscreteFunction& f);

    /// Interpolate function to vertices of mesh
    void interpolate(real* values) const;

    /// Interpolate function to finite element space on cell
    void interpolate(real* coefficients,
                     const ufc::cell& cell,
                     const ufc::finite_element& finite_element) const;

    /// Evaluate function at given point
    void eval(real* values, const real* x) const;

    /// Return vector
    GenericVector& vector() const;

    /// Friends
    friend class XMLFile;

  private:

    // Scratch space
    class Scratch
    {
    public:

      // Constructor
      Scratch(ufc::finite_element& finite_element);

      // Destructor
      ~Scratch();

      // Value size (number of entries in tensor value)
      uint size;
      
      // Local array for mapping of dofs
      uint* dofs;
      
      // Local array for expansion coefficients
      real* coefficients;
      
      // Local array for values
      real* values;

    };

    // Initialize discrete function
    void init(Mesh& mesh, GenericVector& x, const ufc::form& form, uint i);

    // The vector of dofs
    GenericVector* x;

    // The finite element
    ufc::finite_element* finite_element;
    
    // The dof map
    DofMap* dof_map;

    // Pointers to local data if owned
    GenericVector* local_vector;
    DofMap* local_dof_map;

    // Intersection detector
    mutable IntersectionDetector* intersection_detector;

    // Scratch space
    Scratch* scratch;

  };

}

#endif
