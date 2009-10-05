// Copyright (C) 2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-09-28
// Last changed: 2009-10-05

#ifndef __EXPRESSION_H
#define __EXPRESSION_H

#include <vector>
#include "GenericFunction.h"

namespace dolfin
{

  class Data;

  /// This class represents a user-defined expression. Expressions can
  /// be used as coefficients in variational forms or interpolated
  /// into finite element spaces.
  ///
  /// An expression is defined by overloading the eval() method. Users
  /// may choose to overload either a simple version of eval(), in the
  /// case of expressions only depending on the coordinate x, or an
  /// optional version for expressions depending on x and mesh data
  /// like cell indices or facet normals.
  ///
  /// The geometric dimension (the size of x) and the value rank and
  /// dimensions of an expression must supplied as arguments to the
  /// constructor.

  class Expression : public GenericFunction
  {
  public:

    /// Create scalar expression
    explicit Expression(uint geometric_dimension);

    /// Create vector-valued expression with given dimension
    Expression(uint geometric_dimension, uint dim);

    /// Create tensor-valued expression with given shape
    Expression(uint geometric_dimension, const std::vector<uint>& value_shape);

    /// Copy constructor
    Expression(const Expression& expression);

    /// Destructor
    virtual ~Expression();

    /// Return geometric dimension
    uint geometric_dimension() const;

    /// Return value rank
    uint value_rank() const;

    /// Return value dimension for given axis
    uint value_dimension(uint i) const;

    /// Return value shape
    const std::vector<uint>& value_shape() const;

    //--- User-supplied callbacks for expression evaluation ---

    /// Evaluate expression, must be overloaded by user (simple version)
    virtual void eval(double* values, const double* x) const;

    /// Evaluate expression, must be overloaded by user (optional version)
    virtual void eval(double* values, const Data& data) const;

    //--- Implementation of GenericFunction interface ---

    /// Restrict function to local cell (compute expansion coefficients w)
    virtual void restrict(double* w,
                          const FiniteElement& element,
                          const Cell& dolfin_cell,
                          const ufc::cell& ufc_cell,
                          int local_facet) const;

  private:

    // Geometric dimension
    uint _geometric_dimension;

    // Value shape
    std::vector<uint> _value_shape;

  };

}

#endif
