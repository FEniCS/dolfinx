// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __NEW_PDE_H
#define __NEW_PDE_H

#include <dolfin/constants.h>
#include <dolfin/IndexPair.h>
#include <dolfin/FunctionPair.h>
#include <dolfin/NewArray.h>

namespace dolfin
{

  class Cell;
  class Function;

  /// EXPERIMENTAL: Redesign of the evaluation of variational forms

  class NewPDE
  {
  public:

    /// Constructor
    NewPDE(unsigned int size, bool interior, bool boundary);

    /// Destructor
    virtual ~NewPDE();

    /// Return size of element matrix
    unsigned int size() const;

    /// Return true if form contains integrals over interior of domain
    bool interior() const;

    /// Return true if form contains integrals over boundary of domain
    bool boundary() const;

    /// Return dimension of solution vector
    virtual unsigned int dim() const = 0;

    /// Return map from local to global degree of freedom
    virtual unsigned int dof(unsigned int i, const Cell& cell) const = 0;
   
    /// Update map
    virtual void update(const Cell& cell);

    /// Compute interior element matrix
    virtual void interiorElementMatrix(NewArray< NewArray<real> >& A) const;

    /// Compute boundary element matrix
    virtual void boundaryElementMatrix(NewArray< NewArray<real> >& A) const;

    /// Compute interior element vector
    virtual void interiorElementVector(NewArray<real>& b) const;

    /// Compute boundary element vector
    virtual void boundaryElementVector(NewArray<real>& b) const;

    /// Friends
    friend class NewFEM;

  protected:

    /// Add function pair (local and global)
    void add(NewArray<real>& w, Function& f);

    /// Update functions
    void updateFunctions(const Cell& cell);

    /// Update affine map from reference triangle
    void updateTriLinMap(const Cell& cell);
    
    /// Update affine map from reference tetrahedron
    void updateTetLinMap(const Cell& cell);

    // List of nonzero indices
    NewArray<IndexPair> nonzero;

    // Determinant of Jacobian of map
    real det;

    // Jacobian of map
    real f00, f01, f02, f10, f11, f12, f20, f21, f22;

    // Inverse of Jacobian of map
    real g00, g01, g02, g10, g11, g12, g20, g21, g22;

    // Current time
    real t;

  private:
    
    // Size of element matrix
    unsigned int _size;

    // True if form contains integrals over interior of domain
    bool _interior;

    // True if form contains integrals over boundary of domain
    bool _boundary;

    // List of function pairs
    NewArray<FunctionPair> functions;

  };

}

#endif
