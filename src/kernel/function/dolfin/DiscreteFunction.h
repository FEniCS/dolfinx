// Copyright (C) 2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2007.
//
// First added:  2007-04-02
// Last changed: 2007-08-27

#ifndef __DISCRETE_FUNCTION_H
#define __DISCRETE_FUNCTION_H

#include <dolfin/Vector.h>
#include <dolfin/GenericFunction.h>

namespace dolfin
{

  class Mesh;
  class Form;
  class DofMap;
  class SubFunction;

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
    DiscreteFunction(Mesh& mesh, DofMap& dof_map, Vector& x, const Form& form, uint i);

    /// Create discrete function for argument function i of form
    DiscreteFunction(Mesh& mesh, DofMap& dof_map, Vector& x, const ufc::form& form, uint i);

    /// Create discrete function from given data and assume responsibility for data
    DiscreteFunction(Mesh& mesh, Vector& x, std::string finite_element_signature, std::string dof_map_signature);

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
    void interpolate(real* values);

    /// Interpolate function to finite element space on cell
    void interpolate(real* coefficients,
                     const ufc::cell& cell,
                     const ufc::finite_element& finite_element);

    /// Return vector
    Vector& vector() const;

    /// Friends
    friend class XMLFile;

  private:

    // Initialize discrete function
    void init(Mesh& mesh, Vector& x, const ufc::form& form, uint i);

    // The vector of dofs
    Vector* x;

    // The finite element
    ufc::finite_element* finite_element;

    // The dof map
    DofMap* dof_map;

    // Local array for mapping of dofs
    uint* dofs;

    // Pointers to local data if owned
    Mesh* local_mesh;
    Vector* local_vector;
    DofMap* local_dof_map;

  };

}

#endif
