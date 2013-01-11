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
// Modified by Niclas Jansson 2009.
// Modified by Garth Wells 2009-2012
// Modified by Mikael Mortensen 2012.
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

  /// Documentation of class

  class DofMapBuilder
  {

    // FIXME: Test which 'map' is most efficient
    typedef std::map<dolfin::la_index, dolfin::la_index> map;
    typedef std::map<dolfin::la_index, dolfin::la_index>::const_iterator map_iterator;

    // FIXME: Test which 'set' is most efficient

    //typedef std::set<std::size_t> set;
    //typedef std::set<std::size_t>::const_iterator set_iterator;

    typedef boost::unordered_set<std::size_t> set;
    typedef boost::unordered_set<std::size_t>::const_iterator set_iterator;

    typedef std::vector<std::size_t>::const_iterator vector_it;
    typedef boost::unordered_map<std::size_t, std::vector<std::size_t> > vec_map;

    typedef std::pair<std::size_t, std::size_t> facet_data;
    typedef std::map<std::size_t, std::size_t> periodic_map;
    typedef std::vector<facet_data> vector_of_pairs;
    typedef periodic_map::iterator periodic_map_iterator;
    typedef std::vector<std::pair<facet_data, facet_data> > facet_pair_type;

  public:

    // Build dofmap. The restriction may be a null pointer in which
    // case it is ignored.
    static void build(DofMap& dofmap, const Mesh& dolfin_mesh,
        boost::shared_ptr<const Restriction> restriction,
        const std::map<std::size_t, std::pair<std::size_t, std::size_t> > slave_to_master_facets);

    // Build dofmap. The restriction may be a null pointer in which
    // case it is ignored.
    static void build_sub_map(DofMap& sub_dofmap,
                              const DofMap& parent_dofmap,
                              const std::vector<std::size_t>& component,
                              const Mesh& mesh);

  private:

    // Build UFC-based dofmap
    static void build_ufc(DofMap& dofmap, map& restricted_dofs_inverse,
                         const Mesh& mesh,
                         boost::shared_ptr<const Restriction> restriction);

    // Re-order dofmap locally for dof spatial locality
    static void reorder_local(DofMap& dofmap, const Mesh& mesh);

    // Re-order distributed dof map for process locality
    static void reorder_distributed(DofMap& dofmap,
                                  const Mesh& mesh,
                                  boost::shared_ptr<const Restriction> restriction,
                                  const map& restricted_dofs_inverse);

    static void compute_dof_ownership(set& owned_dofs, set& shared_owned_dofs,
                                  set& shared_unowned_dofs,
                                  vec_map& shared_dof_processes,
                                  DofMap& dofmap,
                                  const DofMapBuilder::set& global_dofs,
                                  const Mesh& mesh,
                                  boost::shared_ptr<const Restriction> restriction,
                                  const map& restricted_dofs_inverse);

    static void parallel_renumber(const set& owned_dofs,
                                  const set& shared_owned_dofs,
                                  const set& shared_unowned_dofs,
                                  const vec_map& shared_dof_processes,
                                  DofMap& dofmap,
                                  const Mesh& mesh,
                                  boost::shared_ptr<const Restriction> restriction,
                                  const map& restricted_dofs_inverse);

    // Compute set of global dofs (e.g. Reals associated with global
    // Lagrange multipliers) based on UFC numbering. Global dofs
    // are not associated with any mesh entity.
    static set compute_global_dofs(const DofMap& dofmap,
                                   const Mesh& dolfin_mesh);

    // Iterate recursively over all sub-dof maps to find global
    // degrees of freedom
    static void compute_global_dofs(set& global_dofs, std::size_t& offset,
                            boost::shared_ptr<const ufc::dofmap> dofmap,
                            const Mesh& dolfin_mesh);

    // Iterate recursively over all sub-dof maps to build a global
    // map from slave dofs to master dofs. Build also a map of all
    // processes that shares the master dofs
    static void extract_dof_pairs(const DofMap& dofmap, const Mesh& mesh,
        periodic_map& _slave_master_map,
        std::map<std::size_t, boost::unordered_set<std::size_t> >& _master_processes);

    // Make all necessary modifications to dofmap due to periodicity of the mesh
    static void periodic_modification(DofMap& dofmap, const Mesh& dolfin_mesh,
      DofMapBuilder::set& global_dofs);

    // Recursively extract UFC sub-dofmap and compute offset
    static ufc::dofmap* extract_ufc_sub_dofmap(const ufc::dofmap& ufc_dofmap,
                                               std::size_t& offset,
                                               const std::vector<std::size_t>& component,
                                               const Mesh& mesh);

  };
}

#endif
