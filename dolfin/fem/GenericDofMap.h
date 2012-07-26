// Copyright (C) 2010-2011 Anders Logg and Garth N. Wells
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
// Modified by Joachim B Haga, 2012
//
// First added:  2010-05-26
// Last changed: 2012-02-29

#ifndef __GENERIC_DOF_MAP_H
#define __GENERIC_DOF_MAP_H

#include <map>
#include <utility>
#include <vector>
#include <boost/multi_array.hpp>
#include <boost/unordered_map.hpp>
#include <boost/unordered_set.hpp>

#include <dolfin/common/types.h>
#include <dolfin/common/Variable.h>

namespace ufc
{
  class cell;
}

namespace dolfin
{

  class Cell;
  class GenericVector;
  class Mesh;
  template<typename T> class Set;

  /// This class provides a generic interface for dof maps

  class GenericDofMap : public Variable
  {
  public:

    /// True if dof map is a view into another map (is a sub-dofmap)
    virtual bool is_view() const = 0;

    /// Return true iff mesh entities of topological dimension d are needed
    virtual bool needs_mesh_entities(unsigned int d) const = 0;

    /// Return the dimension of the global finite element function space
    virtual unsigned int global_dimension() const = 0;

    /// Return the dimension of the local finite element function space on a
    /// cell
    virtual unsigned int cell_dimension(uint index) const = 0;

    /// Return the maximum dimension of the local finite element function space
    virtual unsigned int max_cell_dimension() const = 0;

    // Return the geometric dimension of the coordinates this dof map provides
    virtual unsigned int geometric_dimension() const = 0;

    /// Return number of facet dofs
    virtual unsigned int num_facet_dofs() const = 0;

    /// Return the ownership range (dofs in this range are owned by this process)
    virtual std::pair<unsigned int, unsigned int> ownership_range() const = 0;

    /// Return map from nonlocal-dofs (that appear in local dof map) to owning process
    virtual const boost::unordered_map<unsigned int, unsigned int>& off_process_owner() const = 0;

    /// Local-to-global mapping of dofs on a cell
    virtual const std::vector<unsigned int>& cell_dofs(uint cell_index) const = 0;

    /// Tabulate the local-to-global mapping of dofs on a cell
    virtual void tabulate_dofs(uint* dofs, const Cell& cell) const = 0;

    /// Tabulate local-local facet dofs
    virtual void tabulate_facet_dofs(uint* dofs, uint local_facet) const = 0;

    /// Tabulate the coordinates of all dofs on a cell (UFC cell version)
    virtual void tabulate_coordinates(boost::multi_array<double, 2>& coordinates,
                                      const ufc::cell& ufc_cell) const = 0;

    /// Tabulate the coordinates of all dofs on a cell (DOLFIN cell version)
    virtual void tabulate_coordinates(boost::multi_array<double, 2>& coordinates,
                                      const Cell& cell) const = 0;

    /// Create a copy of the dof map
    virtual GenericDofMap* copy(const Mesh& mesh) const = 0;

    /// Extract sub dofmap component
    virtual GenericDofMap* extract_sub_dofmap(const std::vector<uint>& component,
                                              const Mesh& mesh) const = 0;

    /// Create a "collapsed" a dofmap (collapses from a sub-dofmap view)
    virtual GenericDofMap* collapse(boost::unordered_map<uint, uint>& collapsed_map,
                                    const Mesh& mesh) const = 0;

    /// Set dof values in vector to a specified value. Parallel layout of
    /// vector must be consistent with dof map range
    virtual void set(GenericVector& x, double value) const = 0;

    /// Return the set of dof indices
    virtual boost::unordered_set<uint> dofs() const = 0;

    /// Return map from shared dofs to the processes (not including the current
    /// process) that share it.
    virtual const boost::unordered_map<uint, std::vector<uint> >& shared_dofs() const = 0;

    /// Return set of all processes that share dofs with the current process.
    virtual const std::set<uint>& neighbours() const = 0;

    /// Re-number based on provided re-numbering map
    virtual void renumber(const std::vector<uint>& renumbering_map) = 0;

    /// Return informal string representation (pretty-print)
    virtual std::string str(bool verbose) const = 0;

  };

}

#endif
