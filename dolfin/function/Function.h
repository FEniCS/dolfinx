// Copyright (C) 2003-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2005-2009.
// Modified by Kristian B. Oelgaard, 2007.
// Modified by Martin Sandve Alnes, 2008.
//
// First added:  2003-11-28
// Last changed: 2009-09-29

#ifndef __FUNCTION_H
#define __FUNCTION_H

#include <map>
#include <vector>
#include <boost/ptr_container/ptr_map.hpp>
#include <boost/shared_ptr.hpp>
#include <dolfin/common/Variable.h>
#include <dolfin/log/log.h>
#include "Coefficient.h"

namespace ufc
{
  // Forward declarations
  class cell;
}

namespace dolfin
{

  // Forward declarations
  class Expression;
  class FunctionSpace;
  class GenericVector;
  class Data;

  /// This class represents a function u_h in a finite element
  /// function space V_h, given by
  ///
  ///   u_h = sum_i U_i phi_i
  ///
  /// where {phi_i}_i is a basis for V_h, and U is a vector of
  /// expansion coefficients for u_h.

  class Function : public Variable, public Coefficient
  {
  public:

    /// Create function on given function space
    explicit Function(const FunctionSpace& V);

    /// Create function on given function space (shared data)
    explicit Function(boost::shared_ptr<const FunctionSpace> V);

    /// Create function on given function space with a given vector
    Function(const FunctionSpace& V, GenericVector& x);

    /// Create function on given function space with a given vector (shared FunctionSpace, needed for the PyDOLFIN interface)
    Function(boost::shared_ptr<const FunctionSpace> V, GenericVector& x);

    /// Create function on given function space with a given vector (shared data)
    Function(boost::shared_ptr<const FunctionSpace> V, boost::shared_ptr<GenericVector> x);

    /// Create function from vector of dofs stored to file
    Function(const FunctionSpace& V, std::string filename);

    /// Create function from vector of dofs stored to file (shared data)
    Function(boost::shared_ptr<const FunctionSpace> V, std::string filename);

    /// Copy constructor
    Function(const Function& v);

    /// Destructor
    virtual ~Function();

    /// Assignment from function
    const Function& operator= (const Function& v);

    /// Assignment from Expression using interpolation
    const Function& operator= (const Expression& v);

    /// Extract sub function data
    Function& operator[] (uint i);

    /// Return the function space
    const FunctionSpace& function_space() const;

    /// Return the function space
    boost::shared_ptr<const FunctionSpace> function_space_ptr() const;

    /// Return the vector of expansion coefficients, automatically
    /// initialized to zero if coefficients have not been computed (non-const version)
    GenericVector& vector();

    /// Return the vector of expansion coefficients, automatically
    /// initialized to zero if coefficients have not been computed (const version)
    const GenericVector& vector() const;

    /// Check if function is a member of the given function space
    bool in(const FunctionSpace& V) const;

    /// Return geometric dimension
    uint geometric_dimension() const;

    /// Function evaluation (overload for user-defined function, simple version)
    void eval(double* values, const double* x) const;

    /// Function evaluation (overload for user-defined function, alternate version)
    void eval(double* values, const Data& data) const;

    /// Evaluate function v at given point in given cell
    void eval(double* values, const double* x, const ufc::cell& ufc_cell,
              uint cell_index) const;

    /// Interpolate function (possibly non-matching meshes)
    void interpolate(const Function& v);

    /// Interpolate expreesion a function space
    void interpolate(const Expression& v);

    /// Interpolate function to vertices of mesh
    void interpolate_vertex_values(double* vertex_values) const;

    //--- Implementation of Coefficient interface ---

    /// Restrict coefficient to local cell (compute expansion coefficients w)
    virtual void restrict(double* w,
                          const FiniteElement& element,
                          const Cell& dolfin_cell,
                          const ufc::cell& ufc_cell,
                          int local_facet) const;

    /// Collect off-process coefficients to prepare for interpolation
    virtual void gather() const;

    /// Friends
    friend class Coefficient;
    friend class DiscreteFunction;
    friend class VariationalProblem;

  protected:

    // The function space
    boost::shared_ptr<const FunctionSpace> _function_space;

  private:

    // Collection of sub-functions which share data with the function
    boost::ptr_map<uint, Function> sub_functions;

    void compute_off_process_dofs() const;

    // Initialize vector
    void init_vector();

    // Get coefficients from the vector(s)
    void get(double* block, uint m, const uint* rows) const;

    // The vector of expansion coefficients (local)
    boost::shared_ptr<GenericVector> _vector;

    // The vector of expansion coefficients (off-process)
    mutable boost::shared_ptr<GenericVector> _off_process_vector;

    mutable std::map<uint, uint> global_to_local;
    mutable std::vector<uint> _off_process_dofs;

    // Scratch data used in extracting coefficients from parallel vectors
    class Scratch
    {
    public:

      Scratch() : local_rows(0), nonlocal_rows(0), local_block(0),
                  nonlocal_block(0), local_index(0), nonlocal_index(0)
      {}

      ~Scratch()
      {
        clear();
      }

      void init(uint dim)
      {
        clear();
        local_rows     = new uint[dim];
        nonlocal_rows  = new uint[dim];
        local_block    = new double[dim];
        nonlocal_block = new double[dim];
        local_index    = new uint[dim];
        nonlocal_index = new uint[dim];
      }

      void clear()
      {
        delete [] local_rows;
        delete [] nonlocal_rows;
        delete [] local_block;
        delete [] nonlocal_block;
        delete [] local_index;
        delete [] nonlocal_index;
      }

      uint* local_rows;
      uint* nonlocal_rows;
      double* local_block;
      double* nonlocal_block;
      uint* local_index;
      uint* nonlocal_index;
    };
    mutable Scratch scratch;
  };

}

#endif
