// Copyright (C) 2003-2009 Anders Logg
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Garth N. Wells, 2005-2010.
// Modified by Kristian B. Oelgaard, 2007.
// Modified by Martin Sandve Alnes, 2008.
// Modified by Andre Massing, 2009.
//
// First added:  2003-11-28
// Last changed: 2011-03-15

#ifndef __FUNCTION_H
#define __FUNCTION_H

#include <utility>
#include <vector>
#include <boost/ptr_container/ptr_map.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/shared_ptr.hpp>

#include <dolfin/common/Hierarchical.h>
#include "GenericFunction.h"

namespace ufc
{
  // Forward declarations
  class cell;
}

namespace dolfin
{

  // Forward declarations
  class DirichletBC;
  class Expression;
  class FunctionSpace;
  class GenericVector;
  class SubDomain;
  template<class T> class Array;

  /// This class represents a function u_h in a finite element
  /// function space V_h, given by
  ///
  ///   u_h = sum_i U_i phi_i
  ///
  /// where {phi_i}_i is a basis for V_h, and U is a vector of
  /// expansion coefficients for u_h.

  class Function : public GenericFunction, public Hierarchical<Function>

  {
  public:

    /// Create function on given function space
    explicit Function(const FunctionSpace& V);

    /// Create function on given function space (shared data)
    explicit Function(boost::shared_ptr<const FunctionSpace> V);

    /// Create function on given function space with a given vector
    Function(const FunctionSpace& V,
             GenericVector& x);

    /// Create function on given function space with a given vector
    /// (shared data)
    Function(boost::shared_ptr<const FunctionSpace> V,
             boost::shared_ptr<GenericVector> x);

    /// Create function on given function space with a given vector (used by
    /// Python interface)
    Function(boost::shared_ptr<const FunctionSpace> V,
             GenericVector& x);

    /// Create function from vector of dofs stored to file
    Function(const FunctionSpace& V,
             std::string filename);

    /// Create function from vector of dofs stored to file (shared data)
    Function(boost::shared_ptr<const FunctionSpace> V,
             std::string filename);

    /// Copy constructor
    Function(const Function& v);

    /// Sub-function constructor with shallow copy of vector (used in Python
    /// interface)
    Function(const Function& v, uint i);

    /// Destructor
    virtual ~Function();

    /// Assignment from function
    const Function& operator= (const Function& v);

    /// Assignment from expression using interpolation
    const Function& operator= (const Expression& v);

    /// Extract sub-function
    Function& operator[] (uint i) const;

    /// Return function space
    const FunctionSpace& function_space() const;

    /// Return shared pointer to function space
    boost::shared_ptr<const FunctionSpace> function_space_ptr() const;

    /// Return vector of expansion coefficients (non-const version)
    GenericVector& vector();

    /// Return vector of expansion coefficients (const version)
    const GenericVector& vector() const;

    /// Check if function is a member of the given function space
    bool in(const FunctionSpace& V) const;

    /// Return geometric dimension
    uint geometric_dimension() const;

    /// Evaluate function for given coordinate
    void eval(Array<double>& values, const Array<double>& x) const;

    /// Evaluate function for given coordinate in given cell
    void eval(Array<double>& values,
              const Array<double>& x,
              const Cell& dolfin_cell,
              const ufc::cell& ufc_cell) const;

    /// Interpolate function (possibly non-matching meshes)
    void interpolate(const GenericFunction& v);

    /// Extrapolate function (from a possibly lower-degree function space)
    void extrapolate(const Function& v);

    //--- Implementation of GenericFunction interface ---

    /// Return value rank
    virtual uint value_rank() const;

    /// Return value dimension for given axis
    virtual uint value_dimension(uint i) const;

    /// Evaluate function for given data
    virtual void eval(Array<double>& values, const Array<double>& x,
                      const ufc::cell& cell) const;

    /// Evaluate function for given data
    void non_matching_eval(Array<double>& values, const Array<double>& x,
                           const ufc::cell& ufc_cell) const;

    /// Restrict function to local cell (compute expansion coefficients w)
    virtual void restrict(double* w,
                          const FiniteElement& element,
                          const Cell& dolfin_cell,
                          const ufc::cell& ufc_cell) const;

    /// Compute values at all mesh vertices
    virtual void compute_vertex_values(Array<double>& vertex_values,
                                       const Mesh& mesh) const;

    /// Collect off-process coefficients to prepare for interpolation
    virtual void gather() const;

  private:

    // Friends
    friend class FunctionSpace;

    // Collection of sub-functions which share data with the function
    mutable boost::ptr_map<uint, Function> sub_functions;

    // Compute lists of off-process dofs
    void compute_off_process_dofs() const;

    // Initialize vector
    void init_vector();

    // Get coefficients from the vector(s)
    void compute_ghost_indices(std::pair<uint, uint> range,
                               std::vector<uint>& ghost_indices) const;

    // The function space
    boost::shared_ptr<const FunctionSpace> _function_space;

    // The vector of expansion coefficients (local)
    boost::shared_ptr<GenericVector> _vector;

    // True if extrapolation should be allowed
    bool allow_extrapolation;

  };

}

#endif
