// Copyright (C) 2008 Anders Logg and Ola Skavhaug.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Niclas Jansson 2009.
//
// First added:  2008-08-12
// Last changed: 2009-11-04

#ifndef __DOF_MAP_BUILDER_H
#define __DOF_MAP_BUILDER_H

#include <set>
#include <boost/unordered_set.hpp>
#include <dolfin/common/Set.h>

namespace dolfin
{

  class DofMap;
  class Mesh;
  class UFC;
  class UFCMesh;

  /// Documentation of class

  class DofMapBuilder
  {
    // FIXME: Test which 'set' is most efficient

    //typedef std::set<dolfin::uint> set;
    //typedef std::set<dolfin::uint>::const_iterator set_iterator;
    //typedef Set<dolfin::uint> set;
    //typedef Set<dolfin::uint>::const_iterator set_iterator;
    //typedef std::tr1::unordered_set<dolfin::uint> set;
    //typedef std::tr1::unordered_set<dolfin::uint>::const_iterator set_iterator;
    typedef boost::unordered_set<dolfin::uint> set;
    typedef boost::unordered_set<dolfin::uint>::const_iterator set_iterator;

    typedef std::vector<dolfin::uint>::const_iterator vector_it;

  public:

    static void build(DofMap& dofmap, const Mesh& dolfin_mesh,
                      const UFCMesh& ufc_mesh, bool distributed);


  private:

    // Build distributed dof map
    static void build_distributed(DofMap& dofmap, const Mesh& mesh);

    static void compute_ownership(set& owned_dofs, set& shared_owned_dofs,
                                  set& shared_unowned_dofs,
                                  const DofMap& dofmap,
                                  const Mesh& mesh);

    static void parallel_renumber(const set& owned_dofs, const set& shared_owned_dofs,
                                  const set& shared_unowned_dofs,
                                  DofMap& dofmap, const Mesh& mesh);


  };

}

#endif
