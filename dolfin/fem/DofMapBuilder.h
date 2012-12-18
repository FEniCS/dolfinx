// Copyright (C) 2008-2012 Anders Logg and Ola Skavhaug
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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Niclas Jansson 2009
// Modified by Garth Wells 2009-2012
//
// First added:  2008-08-12
// Last changed: 2012-11-05

#ifndef __DOF_MAP_BUILDER_H
#define __DOF_MAP_BUILDER_H

#include <set>
#include <map>
#include <boost/unordered_set.hpp>
#include <boost/unordered_map.hpp>
#include <dolfin/common/types.h>
#include <dolfin/common/Set.h>

namespace ufc
{
  class dofmap;
}

namespace dolfin
{

  class DofMap;
  class Mesh;
  class UFC;
  class UFCMesh;

  /// Documentation of class

  class DofMapBuilder
  {

    // FIXME: Test which 'map' is most efficient
    typedef std::map<DolfinIndex, DolfinIndex> map;
    typedef std::map<DolfinIndex, DolfinIndex>::const_iterator map_iterator;

    // FIXME: Test which 'set' is most efficient
    typedef std::set<std::size_t> set;
    typedef std::set<std::size_t>::const_iterator set_iterator;

    //typedef boost::unordered_set<dolfin::std::size_t> set;
    //typedef boost::unordered_set<dolfin::std::size_t>::const_iterator set_iterator;

    typedef std::vector<std::size_t>::const_iterator vector_it;
    typedef boost::unordered_map<std::size_t, std::vector<std::size_t> > vec_map;

  public:

    // Build dofmap. The restriction may be a null pointer in which
    // case it is ignored.
    static void build(DofMap& dofmap,
                      const Mesh& dolfin_mesh,
                      const UFCMesh& ufc_mesh,
                      boost::shared_ptr<const Restriction> restriction,
                      bool reorder,
                      bool distributed);

  private:

    // Build distributed dof map
    static void build_distributed(DofMap& dofmap,
                                  const DofMapBuilder::set& global_dofs,
                                  const Mesh& mesh);

    static void compute_ownership(set& owned_dofs, set& shared_owned_dofs,
                                  set& shared_unowned_dofs,
                                  vec_map& shared_dof_processes,
                                  const DofMap& dofmap,
                                  const DofMapBuilder::set& global_dofs,
                                  const Mesh& mesh);

    static void parallel_renumber(const set& owned_dofs,
                                  const set& shared_owned_dofs,
                                  const set& shared_unowned_dofs,
                                  const vec_map& shared_dof_processes,
                                  DofMap& dofmap, const Mesh& mesh);

    /// Compute set of global dofs (e.g. Reals associated with global
    /// Lagrnage multipliers) based on UFC numbering. Global dofs
    /// are not associated with any mesh entity
    static set compute_global_dofs(const DofMap& dofmap,
                                   const Mesh& dolfin_mesh);


    // Iterate recursively over all sub-dof maps to find global
    // degrees of freedom
    static void compute_global_dofs(set& global_dofs, std::size_t& offset,
                            boost::shared_ptr<const ufc::dofmap> dofmap,
                            const Mesh& dolfin_mesh, const UFCMesh& ufc_mesh);


  };

}

#endif
