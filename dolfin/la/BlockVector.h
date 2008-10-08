// Copyright (C) 2008 Kent-Andre Mardal.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-08-25

#ifndef __BLOCKVECTOR_H
#define __BLOCKVECTOR_H

#include <map>
#include "Vector.h"

namespace dolfin
{
  class SubVector; 

  class BlockVector 
  {
    private: 
      bool owner; 
      uint n; 
      Vector** vectors;  

    public:

      /// Constructor  
      BlockVector(uint n_=0, bool owner=false);  

      /// Destructor
      virtual ~BlockVector(); 

      /// Return copy of tensor
      virtual BlockVector* copy() const;

      SubVector operator() (uint i); 

      // Set function 
      void set(uint i, Vector& v);

      // Get functions (const and non-const) 
      const Vector& getc(uint i) const; 
            Vector& get(uint); 

      /// Add multiple of given vector (AXPY operation)
      void axpy(double a, const BlockVector& x);

      /// Return inner product with given vector
      double inner(const BlockVector& x) const;

      /// Return norm of vector
      double norm(NormType type) const;

      /// Return minimum value of vector
      double min() const;

      /// Return maximum value of vector
      double max() const;

      /// Multiply vector by given number
      const BlockVector& operator*= (double a);

      /// Divide vector by given number
      const BlockVector& operator/= (double a);

      /// Add given vector
      const BlockVector& operator+= (const BlockVector& x);

      /// Subtract given vector
      const BlockVector& operator-= (const BlockVector& x);

      /// Assignment operator
      const BlockVector& operator= (const BlockVector& x);

      /// Assignment operator
      const BlockVector& operator= (double a);

      /// Number of vectors
      uint size() const; 

      /// Display vectors 
      virtual void disp(uint precision=2) const;
  }; 

  class SubVector
  {
  public:
    SubVector(uint n, BlockVector& bv);
    ~SubVector(); 

    const SubVector& operator= (Vector& v); 

  private: 
    uint n; 
    BlockVector& bv; 
  }; 

}

#endif 

