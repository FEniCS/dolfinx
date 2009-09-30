// Copyright (C) 2006-2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Kristian B. Oelgaard, 2007.
// Modified by Martin Sandve Alnes, 2008.
// Modified by Garth N. Wells, 2008.
//
// First added:  2006-02-09
// Last changed: 2009-03-11

#ifndef __SPECIAL_FUNCTIONS_H
#define __SPECIAL_FUNCTIONS_H

#include <map>
#include <vector>
#include <boost/assign/list_of.hpp>
#include <boost/ptr_container/ptr_map.hpp>

#include <dolfin/common/types.h>
#include <dolfin/fem/UFC.h>
#include "Function.h"

namespace dolfin
{

  class Form;
  class UFC;
  class FunctionSpace;
  class Data;

  // FIXME: Need to add checks for all these functions that
  // FIXME: the function spaces make sense (scalar, vector etc)

  /// This Function represents the mesh coordinates on a given mesh.
  class MeshCoordinates : public Function
  {
  public:

    /// Constructor
    MeshCoordinates();

    /// Constructor
    MeshCoordinates(const FunctionSpace& V);

    /// Function evaluation
    void eval(double* values, const Data& data) const;

  };

  /// This Function represents the local cell size on a given mesh.
  class CellSize : public Function
  {
  public:

    /// Constructor
    CellSize();

    /// Constructor
    CellSize(const FunctionSpace& V);

    /// Function evaluation
    void eval(double* values, const Data& data) const;

    /// Compute minimal cell diameter
    double min() const;

    /// Compute maximal cell diameter
    double max() const;

  };

  /// This function represents the area/length of a cell facet.
  class FacetArea : public Function
  {
  public:

    /// Constructor
    FacetArea();

    /// Constructor
    FacetArea(const FunctionSpace& V);

    /// Function evaluation
    void eval(double* values, const Data& data) const;

  };

  /// Streamline Upwind Petrov Galerkin stabilizing function
  /// Given the advective field a, this function computes the stabilizing factor
  ///
  ///           s = h*tau*a/(2*|a|)
  ///
  /// where h is the local size of the mesh, tau the local stabilizing factor
  /// calculated from the local Peclet number, a the advective field.
  class SUPGStabilizer : public Function
  {
  public:

    // Constructor
    SUPGStabilizer(const FunctionSpace& V, const Function& f, double sigma_=1.0);

    ~SUPGStabilizer(){}

    void eval(double* values, const Data& data) const;

    // The diffusion coeffisient, used to calculate the local Péclet number
    double sigma;

  private:

    const Function* field;
  };

  /// This function is used for the Python interface. By inheriting
  /// from this function instead of dolfin::Function, we avoid unnecessary
  /// calls through the SWIG created director class, when dealing
  /// with discrete functions in PyDOLFIN.
  class DiscreteFunction : public Function
  {
  public:

    /// Constructor
    DiscreteFunction() : Function() {}

    /// Constructor
    DiscreteFunction(boost::shared_ptr<const FunctionSpace> V) : Function(V)
    {
      vector();
    }

    /// Constructor
    DiscreteFunction(boost::shared_ptr<const FunctionSpace> V, 
                     std::string filename) : Function(V, filename){}

    /// Constructor (deep copy of the vector)
    DiscreteFunction(const Function& v) : Function(v) {}

    /// Sub-function constructor (shallow copy of the vector)
    DiscreteFunction(Function& v, uint i)
    {
      // Get sub-function (Function will store pointer to sub-Function)
      Function& sub_function = v[i]; 
      
      // Copy function space pointer
      this->_function_space = sub_function._function_space;

      // Copy vector pointer
      this->_vector = sub_function._vector;
    }

    /// Destructor
    ~DiscreteFunction() {}

  };
}

#endif
