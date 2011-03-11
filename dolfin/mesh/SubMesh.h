// Copyright (C) 2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-02-11
// Last changed: 2009-02-11

#ifndef __SUB_MESH_H
#define __SUB_MESH_H

#include "MeshFunction.h"
#include "Mesh.h"

namespace dolfin
{

  class SubDomain;

  /// A SubMesh is a mesh defined as a subset of a given mesh. It
  /// provides a convenient way to create matching meshes for
  /// multiphysics applications by creating meshes for subdomains as
  /// subsets of a single global mesh.

  class SubMesh : public Mesh
  {
  public:

    /// Create subset of given mesh marked by sub domain
    SubMesh(const Mesh& mesh, const SubDomain& sub_domain);

    /// Create subset of given mesh marked by mesh function
    SubMesh(const Mesh& mesh, const MeshFunction<unsigned int>& sub_domains, uint sub_domain);

    /// Destructor
    ~SubMesh();

  private:

    /// Create sub mesh
    void init(const Mesh& mesh, const MeshFunction<uint>& sub_domains, uint sub_domain);

  };

}

#endif
