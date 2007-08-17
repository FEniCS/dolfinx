// Copyright (C) 2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-04-10
// Last changed: 2007-08-17

#ifndef __SUB_DOMAIN_H
#define __SUB_DOMAIN_H

#include <dolfin/constants.h>
#include <dolfin/MeshFunction.h>

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

    /// Return true for points inside the sub domain
    virtual bool inside(const real* x, bool on_boundary) const;

    /// Map coordinate x in domain H to coordinate y in domain G (used for periodic boundary conditions)
    virtual void map(const real* x, real* y) const;
    
    /// Set sub domain markers for given sub domain
    void mark(MeshFunction<uint>& sub_domains, uint sub_domain) const;

  };

}

#endif
