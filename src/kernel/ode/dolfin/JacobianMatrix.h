// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __JACOBIAN_MATRIX_H
#define __JACOBIAN_MATRIX_H

#include <dolfin/Matrix.h>

namespace dolfin
{
  
  class RHS;
  class cGqElement;
  class dGqElement;

  /// This class represents the Jacobian matrix of the system of
  /// equations defined on a time slab.

  class JacobianMatrix : public Matrix
  {
  public:

    /// Constructor
    JacobianMatrix(RHS& f);

    /// Destructor
    ~JacobianMatrix();

    /// Return dimension of matrix
    unsigned int size(unsigned int dim) const;
    
    /// Multiplication with vector (use with GMRES solver)
    void mult(const Vector& x, Vector& Ax) const;

    /// Recompute Jacobian at given time
    void update(real t);

    /// Update Jacobian for time slab and compute the number of unknowns
    unsigned int update(ElementGroupList& elements);

    /// Show Jacobian of the system of equations on the time slab
    void show() const;

  private:

    /// Multiplication for cG(q) elements
    unsigned int cGmult(const Vector& x, Vector& Ax, unsigned int dof,
			const dolfin::Element& element) const;

    /// Multiplication for dG(q) elements
    unsigned int dGmult(const Vector& x, Vector& Ax, unsigned int dof, 
			const dolfin::Element& element) const;

    // The right-hand side
    RHS& f;
    
    // A (sparse) Matrix storing the Jacobian of f
    Matrix dfdu;

    // Number of unknowns
    unsigned int n;

    // List of elements (time slab)
    ElementGroupList* elements;
    
    // List of the latest degree of freedom (closest to T)
    NewArray<unsigned int> latest;

    // List of predecessors to degrees of freedoms
    NewArray<unsigned int> predecessors;
    
    // Start time for the time slab
    real t0;

  };

}

#endif
