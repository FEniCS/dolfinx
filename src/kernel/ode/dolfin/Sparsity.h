// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __SPARSITY_H
#define __SPARSITY_H

#include <dolfin/NewArray.h>

namespace dolfin
{

  class ODE;
  class Vector;
  class Matrix;

  class Sparsity
  {
  public:
    
    // Constructor
    Sparsity(unsigned int N);
    
    // Destructor
    ~Sparsity();
    
    /// Clear sparsity (no dependencies)
    void clear();
    
    /// Clear sparsity for given component
    void clear(unsigned int i);

    /// Set sparsity (number of dependencies for component i)
    void setsize(unsigned int i, unsigned int size);
    
    /// Set sparsity (component i depends on component j)
    void set(unsigned int i, unsigned int j, bool checknew = false);
    
    /// Set sparsity defined by a sparse matrix
    void set(const Matrix& A);

    /// Set sparsity to transpose of given sparsity
    void transp(const Sparsity& sparsity);
    
    /// Automatically detect dependencies
    void detect(ODE& ode);

    /// Check if the dependency pattern is sparse (inline optimized)
    inline bool sparse() const { return !pattern.empty(); }

    /// Get dependencies for given component
    NewArray<unsigned int>& row(unsigned int i);

    /// Get dependencies for given component
    const NewArray<unsigned int>& row(unsigned int i) const;

    /// Show sparsity (dependences)
    void show() const;
    
  private:
    
    // Check given dependency
    bool checkdep(ODE& ode, Vector& u, real f0, unsigned int i, unsigned int j);

    // Number of components
    unsigned int N;
    
    // Increment of automatic detection of sparsity
    real increment;

    // The sparsity pattern
    NewArray< NewArray<unsigned int> > pattern;

  };

}

#endif
