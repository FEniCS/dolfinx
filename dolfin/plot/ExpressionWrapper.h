// Copyright (C) 2012 Fredrik Valdmanis
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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Anders Logg 2012
//
// First added:  2012-06-02
// Last changed: 2012-06-25

#ifndef __EXPRESSION_WRAPPER_H
#define __EXPRESSION_WRAPPER_H

#include <boost/shared_ptr.hpp>
#include <dolfin/function/Expression.h>

namespace dolfin
{

  class Expression;
  class Mesh;

  /// A light wrapper class to hold an expression to plot, along with the mesh
  /// to plot it on. Allows clean, templated plotter code in plot.cpp

  class ExpressionWrapper
  {
  public:

    /// Create wrapped expression object
    explicit ExpressionWrapper(boost::shared_ptr<const Expression> expression,
                               boost::shared_ptr<const Mesh> mesh);

    /// Return unique ID of the expression
    uint id() const
    { return _expression->id(); }

    /// Get shared pointer to the expression
    boost::shared_ptr<const Expression> expression() const
    { return _expression; }

    /// Get shared pointer to the mesh
    boost::shared_ptr<const Mesh> mesh() const
    { return _mesh; }

  private:

    // The mesh to plot on
    boost::shared_ptr<const Mesh> _mesh;

    // The expression itself
    boost::shared_ptr<const Expression> _expression;

  };

}

#endif
