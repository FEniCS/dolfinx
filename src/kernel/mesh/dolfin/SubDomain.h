// Copyright (C) 2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-04-10
// Last changed: 2007-04-24

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
    virtual bool inside(const real* x, bool on_boundary) const = 0;

    /// Set sub domain markers for given sub domain
    void mark(MeshFunction<uint>& sub_domains, uint sub_domain) const;

  };

}

#endif
