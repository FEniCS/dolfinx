// Copyright (C) 2007-2008 Anders Logg
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2007-04-10
// Last changed: 2011-01-25

#ifndef __SUB_DOMAIN_H
#define __SUB_DOMAIN_H

#include <dolfin/common/types.h>

namespace dolfin
{

  template <class T> class MeshFunction;
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

    /// Set sub domain markers (uint) for given subdomain
    void mark(MeshFunction<unsigned int>& sub_domains, unsigned int sub_domain) const;

    /// Set sub domain markers (int) for given subdomain
    void mark(MeshFunction<int>& sub_domains, int sub_domain) const;

    /// Set sub domain markers (double) for given subdomain
    void mark(MeshFunction<double>& sub_domains, double sub_domain) const;

    /// Set sub domain markers (bool) for given subdomain
    void mark(MeshFunction<bool>& sub_domains, bool sub_domain) const;

    /// Return geometric dimension
    uint geometric_dimension() const;

  private:

    /// Set sub domain markers for given subdomain
    template<class T>
    void mark_meshfunction(MeshFunction<T>& sub_domains, T sub_domain) const;

    // Friends
    friend class DirichletBC;
    friend class PeriodicBC;

    // Geometric dimension, needed for SWIG interface, will be set before
    // calls to inside() and map()
    mutable uint _geometric_dimension;

  };

}

#endif
