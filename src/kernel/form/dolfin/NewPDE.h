// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __NEW_PDE_H
#define __NEW_PDE_H

#include <dolfin/constants.h>
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
    NewPDE();

    /// Destructor
    virtual ~NewPDE();

    /// Return size of element matrix
    virtual unsigned int size() const = 0;

    /// Return map from local to global degree of freedom
    virtual unsigned int dof(unsigned int i, const Cell& cell) const = 0;
   
    /// Return true if form contains integrals over interior of domain (default)
    virtual bool interior() const;

    /// Return true if form contains integrals over boundary of domain (default)
    virtual bool boundary() const;
 
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

  protected:

    /// Add function pair (local and global)
    void add(NewArray<real>& w, Function& f);

    /// Update functions
    void updateFunctions(const Cell& cell);

    /// Update affine map from reference triangle
    void updateTriLinMap(const Cell& cell);
    
    /// Update affine map from reference tetrahedron
    void updateTetLinMap(const Cell& cell);

    // Determinant of Jacobian of map
    real det;

    // Jacobian of map
    real f00, f01, f02, f10, f11, f12, f20, f21, f22;

    // Inverse of Jacobian of map
    real g00, g01, g02, g10, g11, g12, g20, g21, g22;

    // Current time
    real t;

    /// Function pair of local and global functions
    class FunctionPair
    {
    public:

      /// Create empty function pair
      FunctionPair();

      /// Create function pair from given functions
      FunctionPair(NewArray<real>& w, Function& f);
      
      /// Update local values on given cell
      void update(const Cell& cell, real t);
      
    private:

      NewArray<real>* w;
      Function* f;
      
    };

    // List of function pairs
    NewArray<FunctionPair> functions;

  };

}

#endif
