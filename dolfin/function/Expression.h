// Copyright (C) 2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-09-28
// Last changed: 2011-01-19

#ifndef __EXPRESSION_H
#define __EXPRESSION_H

#include <vector>
#include <ufc.h>
#include <dolfin/common/Array.h>
#include "GenericFunction.h"

namespace dolfin
{

  class Mesh;

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
    Expression();

    /// Create vector-valued expression with given dimension
    Expression(uint dim);

    /// Create matrix-valued expression with given dimensions
    Expression(uint dim0, uint dim1);

    /// Create tensor-valued expression with given shape
    Expression(std::vector<uint> value_shape);

    /// Copy constructor
    Expression(const Expression& expression);

    /// Destructor
    virtual ~Expression();

    //--- Implementation of GenericFunction interface ---

    /// Return value rank
    virtual uint value_rank() const;

    /// Return value dimension for given axis
    virtual uint value_dimension(uint i) const;

    /// Restrict function to local cell (compute expansion coefficients w)
    virtual void restrict(double* w,
                          const FiniteElement& element,
                          const Cell& dolfin_cell,
                          const ufc::cell& ufc_cell) const;

    /// Compute values at all mesh vertices
    virtual void compute_vertex_values(Array<double>& vertex_values,
                                       const Mesh& mesh) const;

  protected:

    // Value shape
    std::vector<uint> value_shape;

  };

}

#endif
