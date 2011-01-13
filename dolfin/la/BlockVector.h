// Copyright (C) 2008 Kent-Andre Mardal.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg, 2008.
// Modified by Garth N. Wells, 2011.
//
// First added:  2008-08-25
// Last changed: 2011-01-13
//

#ifndef __BLOCKVECTOR_H
#define __BLOCKVECTOR_H

#include <vector>
#include <boost/shared_ptr.hpp>

namespace dolfin
{

  // Forward declarations
  class GenericVector;
  class SubVector;

  class BlockVector
  {
  public:

    /// Constructor
    BlockVector(uint n = 0);

    /// Destructor
    virtual ~BlockVector();

    /// Return copy of tensor
    virtual BlockVector* copy() const;

    SubVector operator() (uint i);

    /// Set function
    void set(uint i, GenericVector& v);

    /// Get functions (const)
    const GenericVector& get(uint i) const;

    /// Get functions (non-const)
    GenericVector& get(uint);

    /// Add multiple of given vector (AXPY operation)
    void axpy(double a, const BlockVector& x);

    /// Return inner product with given vector
    double inner(const BlockVector& x) const;

    /// Return norm of vector
    double norm(std::string norm_type) const;

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

    /// Return informal string representation (pretty-print)
    std::string str(bool verbose) const;

  private:

      //bool owner;
      //uint n;
      std::vector<boost::shared_ptr<GenericVector> > vectors;

  };

  class SubVector
  {
  public:

    SubVector(uint n, BlockVector& bv);
    ~SubVector();

    const SubVector& operator= (GenericVector& v);

  private:

    uint n;
    BlockVector& bv;

  };

}

#endif
