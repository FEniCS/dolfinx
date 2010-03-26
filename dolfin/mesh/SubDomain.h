// Copyright (C) 2007-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-04-10
// Last changed: 2010-03-02

#ifndef __SUB_DOMAIN_H
#define __SUB_DOMAIN_H

#include <dolfin/common/types.h>
#include "MeshFunction.h"

namespace dolfin
{

  template<class T> class Array;

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

    /// Return true for points inside the subdomain
    virtual bool inside(const Array<double>& x, bool on_boundary) const;

    /// Map coordinate x in domain H to coordinate y in domain G (used for
    /// periodic boundary conditions)
    virtual void map(const Array<double>& x, Array<double>&) const;

    /// Snap coordinate to boundary of sub domain
    virtual void snap(Array<double>& x) const {}

    /// Set sub domain markers for given subdomain
    void mark(MeshFunction<uint>& sub_domains, uint sub_domain) const;

    /// Return geometric dimension
    uint geometric_dimension() const;

  private:

    // Friends
    friend class DirichletBC;
    friend class PeriodicBC;

    // Geometric dimension, needed for SWIG interface, will be set before
    // calls to inside() and map()
    mutable uint _geometric_dimension;

  };

}

#endif
