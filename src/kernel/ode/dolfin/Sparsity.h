// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __SPARSITY_H
#define __SPARSITY_H

#include <dolfin/GenericSparsity.h>

namespace dolfin {

  class Matrix;
  class ODE;

  class Sparsity {
  public:
    
    // Constructor
    Sparsity(unsigned int N);

    // Destructor
    ~Sparsity();

    /// Clear sparsity (no dependencies)
    void clear();

    /// Set sparsity (number of dependencies for component i)
    void setsize(unsigned int i, unsigned int size);
    
    /// Set sparsity (component i depends on component j)
    void set(unsigned int i, unsigned int j);
    
    /// Set sparsity defined by a sparse matrix
    void set(const Matrix& A);

    /// Set sparsity to transpose of given sparsity
    void transp(const Sparsity& sparsity);
    
    /// Try to automatically detect dependencies
    void guess(ODE& ode);

    /// Show sparsity (dependences)
    void show() const;
    
    // Iterator over a given row
    class Iterator {
    public:
      
      /// Constructor
      Iterator(unsigned int i, const Sparsity& sparsity);

      /// Destructor
      ~Iterator();
      
      /// Increment
      Iterator& operator++();

      /// Return index for current position
      unsigned int operator*() const;

      /// Return index for current position
      operator unsigned int() const;

      /// Check if we have reached end of the row
      bool end() const;
      
    private:
      
      GenericSparsity::Iterator* iterator;
      
    };

    /// Friends
    friend class Iterator;
    friend class ComponentIterator;
    
  private:

    GenericSparsity::Iterator* createIterator(unsigned int i) const;
    
    GenericSparsity* sparsity;

    unsigned int N;

  };

}

#endif
