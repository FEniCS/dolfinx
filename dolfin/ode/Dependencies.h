// Copyright (C) 2003-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2005-01-06
// Last changed: 2008-10-07

#ifndef __DEPENDENCIES_H
#define __DEPENDENCIES_H

#include <dolfin/common/types.h>
#include <dolfin/common/Array.h>

namespace dolfin
{

  class ODE;

  /// This class keeps track of the dependencies between different
  /// components of an ODE system. For a large ODE, it is important
  /// that the sparsity of the dependency pattern matches the sparsity
  /// of the ODE. By default, a dense pattern is created (but not
  /// stored).
  
  class Dependencies
  {
  public:
    
    /// Constructor
    Dependencies(uint N);
    
    /// Destructor
    ~Dependencies();
    
    /// Specify number of dependencies for component i
    void setsize(uint i, uint size);
    
    /// Add dependency (component i depends on component j)
    void set(uint i, uint j, bool checknew = false);
    
    /// Set dependencies to transpose of given dependencies
    void transp(const Dependencies& dependencies);
    
    /// Automatically detect dependencies
    void detect(ODE& ode);

    /// Check if the dependency pattern is sparse (inline optimized)
    bool sparse() const;

    /// Get dependencies for given component (inline optimized)
    inline Array<uint>& operator[] (uint i) { return ( _sparse ? sdep[i] : ddep ); }

    /// Get dependencies for given component (inline optimized)
    inline const Array<uint>& operator[] (uint i) const { return ( _sparse ? sdep[i] : ddep ); }

    /// Display dependencies
    void disp() const;
    
  private:
    
    // Check given dependency
    bool check_dependency(ODE& ode, real* u, real f0, uint i, uint j);

    // Make pattern sparse
    void make_sparse();

    // Number of components
    uint N;
    
    // Increment for automatic detection of sparsity
    real increment;

    // True if sparsity pattern is sparse
    bool _sparse;
    
    // Sparse dependency pattern
    Array< Array<uint> > sdep;

    // Dense dependency pattern
    Array<uint> ddep;
    
  };

}

#endif
