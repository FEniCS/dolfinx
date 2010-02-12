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
  class UFC;
  class Mesh;

  /// Documentation of class

  class DofMapBuilder
  {
    // FIXME: Test which 'set' is most efficient

    //typedef std::set<dolfin::uint> set;
    //typedef std::set<dolfin::uint>::const_iterator set_iterator;
    //typedef Set<dolfin::uint> set;
    //typedef Set<dolfin::uint>::const_iterator set_iterator;
    typedef boost::unordered_set<dolfin::uint> set;
    typedef boost::unordered_set<dolfin::uint>::const_iterator set_iterator;

    typedef std::vector<dolfin::uint>::const_iterator vector_it;

  public:

    /// Build dof map
    static void parallel_build(DofMap& dofmap, const Mesh& mesh);

  private:

    static void compute_ownership(set& owned_dofs, set& shared_dofs,
                                  set& forbidden_dofs, 
                                  std::map<uint, std::vector<uint> >& dof2index,
                                  const DofMap& dofmap, const Mesh& mesh);

    static void parallel_renumber(const set& owned_dofs, const set& shared_dofs,
                                  const set& forbidden_dofs, 
                                  const std::map<uint, 
                                  std::vector<uint> >& dof2index,
                                  DofMap& dofmap, const Mesh& mesh);


  };

}

#endif


