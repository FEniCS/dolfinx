// Copyright (C) 2008-2013 Anders Logg, Ola Skavhaug and Garth N. Wells
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
#include <boost/array.hpp>
#include <boost/unordered_map.hpp>
#include <boost/unordered_set.hpp>
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

    // Build modified global entity indices that account for periodic bcs
    static std::size_t build_constrained_vertex_indices(const Mesh& mesh,
        const std::map<std::size_t, std::pair<std::size_t, std::size_t> >& slave_to_master_vertices,
        std::vector<std::size_t>& modified_global_indices);

    // Re-order local dofmap for dof spatial locality. Re-ordering is
    // optional, but re-ordering can make other algorithms
    // (e.g. matrix-vector products) significantly faster.
    static void reorder_local(DofMap& dofmap, const Mesh& mesh);

    // Re-order distributed dof map for process locality
    static void reorder_distributed(DofMap& dofmap,
                                   const Mesh& mesh,
                                   boost::shared_ptr<const Restriction> restriction,
                                   const map& restricted_dofs_inverse);

    // Compute which process 'owns' each degree of freedom
    //   dof_ownership[0] -> all dofs owned by this process (will intersect dof_ownership[1])
    //   dof_ownership[1] -> dofs shared with other processes and owned by this process
    //   dof_ownership[2] -> dofs shared with other processes and owned by another process
    static void compute_dof_ownership(boost::array<set, 3>& dof_ownership,
                                  vec_map& shared_dof_processes,
                                  DofMap& dofmap,
                                  const DofMapBuilder::set& global_dofs,
                                  const Mesh& mesh,
                                  boost::shared_ptr<const Restriction> restriction,
                                  const map& restricted_dofs_inverse);

    // Re-order distributed dofmap for process locality based on ownership data
    static void parallel_renumber(const boost::array<set, 3>& dof_ownership,
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

    // Recursively extract UFC sub-dofmap and compute offset
    static boost::shared_ptr<ufc::dofmap>
        extract_ufc_sub_dofmap(const ufc::dofmap& ufc_dofmap,
                               std::size_t& offset,
                               const std::vector<std::size_t>& component,
                               const Mesh& mesh);

  };
}

#endif
