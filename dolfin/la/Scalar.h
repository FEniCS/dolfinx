// Copyright (C) 2007-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2007-2009.
// Modified by Ola Skavhaug, 2007.
//
// First added:  2007-03-15
// Last changed: 2010-03-25

#ifndef __SCALAR_H
#define __SCALAR_H

#include <numeric>
#include <vector>
#include <dolfin/main/MPI.h>
#include "uBLASFactory.h"
#include "GenericTensor.h"

namespace dolfin
{

  class GenericSparsityPattern;

  /// This class represents a real-valued scalar quantity and
  /// implements the GenericTensor interface for scalars.

  class Scalar : public GenericTensor
  {
  public:

    /// Create zero scalar
    Scalar() : GenericTensor(), value(0.0)
    {}

    /// Destructor
    virtual ~Scalar()
    {}

    //--- Implementation of the GenericTensor interface ---

    /// Resize tensor to given dimensions
    virtual void resize(uint rank, const uint*)
    { assert(rank == 0); value = 0.0; }

    /// Initialize zero tensor using sparsity pattern
    void init(const GenericSparsityPattern&)
    { value = 0.0; }

    /// Return copy of tensor
    virtual Scalar* copy() const
    { Scalar* s = new Scalar(); s->value = value; return s; }

    /// Return tensor rank (number of dimensions)
    uint rank() const
    { return 0; }

    /// Return size of given dimension
    uint size(uint) const
    { error("The size() function is not available for scalars."); return 0; }

    /// Get block of values
    void get(double* block, const uint*, const uint * const *) const
    { block[0] = value; }

    /// Set block of values
    void set(const double* block, const uint*, const uint * const *)
    { value = block[0]; }

    /// Add block of values
    void add(const double* block, const uint*, const uint * const *)
    { value += block[0]; }

    /// Set all entries to zero and keep any sparse structure
    void zero()
    { value = 0.0; }

    /// Finalize assembly of tensor
    void apply(std::string)
    {
      if (MPI::num_processes() > 1)
      {
        // Get values from other processes
        std::vector<double> values(MPI::num_processes());
        values[MPI::process_number()] = value;
        MPI::gather(values);

        // Sum contribution from each process
        value = std::accumulate(values.begin(), values.end(), 0.0);
      }
    }

    /// Return informal string representation (pretty-print)
    std::string str(bool) const
    {
      std::stringstream s;
      s << "<Scalar value " << value << ">";
      return s.str();
    }

    //--- Scalar interface ---

    /// Cast to real
    operator double() const
    { return value; }

    /// Assignment from real
    const Scalar& operator= (double value)
    { this->value = value; return *this; }

    //--- Special functions

    /// Return a factory for the default linear algebra backend
    inline LinearAlgebraFactory& factory() const
    { return dolfin::uBLASFactory<>::instance(); }

    /// Get value
    double getval() const
    { return value; }

  private:

    // Value of scalar
    double value;

  };

}

#endif
