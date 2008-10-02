// Copyright (C) 2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-04-10
// Last changed: 2007-08-20

#ifndef __SUB_DOMAIN_H
#define __SUB_DOMAIN_H

#include <dolfin/common/types.h>
#include <dolfin/common/simple_array.h>
#include "MeshFunction.h"

namespace dolfin
{

  /// This class defines the interface for definition of sub domains.
  /// Alternatively, sub domains may be defined by a Mesh and a
  /// MeshFunction<uint> over the mesh.

  class SubDomain
  {
  public:

    /// Constructor
    SubDomain();

    /// Destructor
    virtual ~SubDomain();

    /// Return true for points inside the sub domain (used for subclassing through SWIG interface)
    virtual bool inside(const simple_array<double>& x, bool on_boundary) const { return inside(x.data, on_boundary); }

    /// Return true for points inside the sub domain
    virtual bool inside(const double* x, bool on_boundary) const;
    
    /// Map coordinate x in domain H to coordinate y in domain G (used for subclassing through SWIG interface)
    virtual void map(const simple_array<double>& x, simple_array<double>& y) const { map(x.data, y.data); }

    /// Map coordinate x in domain H to coordinate y in domain G (used for periodic boundary conditions)
    virtual void map(const double* x, double* y) const;
    
    /// Set sub domain markers for given sub domain
    void mark(MeshFunction<uint>& sub_domains, uint sub_domain) const;

  };

}

#endif
