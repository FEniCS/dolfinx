// Copyright (C) 2010-2015 Anders Logg and Garth N. Wells
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
// Modified by Joachim B. Haga, 2012
// Modified by Jan Blechta, 2013
// Modified by Martin Alnes, 2015

#ifndef __GENERIC_DOF_MAP_H
#define __GENERIC_DOF_MAP_H

#include <map>
#include <utility>
#include <vector>
#include <memory>
#include <unordered_map>
#include <unordered_set>

#include <dolfin/common/Array.h>
#include <dolfin/common/ArrayView.h>
#include <dolfin/common/types.h>
#include <dolfin/common/Variable.h>
#include <dolfin/la/IndexMap.h>
#include <dolfin/log/log.h>

namespace ufc
{
  class cell;
}

namespace dolfin
{

  class Cell;
  class GenericVector;
  class Mesh;
  class SubDomain;

  /// This class provides a generic interface for dof maps

  class GenericDofMap : public Variable
  {
  public:

    /// Constructor
    GenericDofMap()
    {}

    /// True if dof map is a view into another map (is a sub-dofmap)
    virtual bool is_view() const = 0;

    /// Return the dimension of the global finite element function
    /// space
    virtual std::size_t global_dimension() const = 0;

    /// Return number of owned (type="owned"), unowned
    /// (type="unowned"), or all (type="all") dofs in the map on this
    /// process
    virtual std::size_t local_dimension(std::string type) const = 0;

    /// Return the dimension of the local finite element function
    /// space on a cell (deprecated API)
    std::size_t cell_dimension(std::size_t index) const
    {
      // TODO: Add deprecation warning
      return num_element_dofs(index);
    }

    /// Return the maximum dimension of the local finite element
    /// function space (deprecated API)
    std::size_t max_cell_dimension() const
    {
      // TODO: Add deprecation warning
      return max_element_dofs();
    }

    /// Return the dimension of the local finite element function
    /// space on a cell
    virtual std::size_t num_element_dofs(std::size_t index) const = 0;

    /// Return the maximum dimension of the local finite element
    /// function space
    virtual std::size_t max_element_dofs() const = 0;

    /// Return the number of dofs for a given entity dimension
    virtual std::size_t num_entity_dofs(std::size_t dim) const = 0;

    /// Return number of facet dofs
    virtual std::size_t num_facet_dofs() const = 0;

    /// Return the ownership range (dofs in this range are owned by
    /// this process)
    virtual std::pair<std::size_t, std::size_t> ownership_range() const = 0;

    /// Return map from nonlocal-dofs (that appear in local dof map)
    /// to owning process
    virtual const std::vector<int>& off_process_owner() const = 0;

    /// Local-to-global mapping of dofs on a cell
    virtual ArrayView<const dolfin::la_index>
    cell_dofs(std::size_t cell_index) const = 0;

    /// Tabulate local-local facet dofs
    virtual void tabulate_facet_dofs(std::vector<std::size_t>& dofs,
                                     std::size_t local_facet) const = 0;

    /// Tabulate the local-to-local mapping of dofs on entity
    /// (dim, local_entity)
    virtual void tabulate_entity_dofs(std::vector<std::size_t>& dofs,
                                      std::size_t dim,
                                      std::size_t local_entity) const = 0;

    /// Create a copy of the dof map
    virtual std::shared_ptr<GenericDofMap> copy() const = 0;

    /// Create a new dof map on new mesh
    virtual std::shared_ptr<GenericDofMap>
      create(const Mesh& new_mesh) const = 0;

    /// Extract sub dofmap component
    virtual std::shared_ptr<GenericDofMap>
        extract_sub_dofmap(const std::vector<std::size_t>& component,
                           const Mesh& mesh) const = 0;

    /// Create a "collapsed" a dofmap (collapses from a sub-dofmap view)
    virtual std::shared_ptr<GenericDofMap>
        collapse(std::unordered_map<std::size_t, std::size_t>& collapsed_map,
                 const Mesh& mesh) const = 0;

    /// Return list of dof indices on this process that belong to mesh
    /// entities of dimension dim
    virtual std::vector<dolfin::la_index> dofs(const Mesh& mesh,
                                               std::size_t dim) const = 0;

    /// Return list of global dof indices on this process
    virtual std::vector<dolfin::la_index> dofs() const = 0;

    /// Set dof entries in vector to a specified value. Parallel
    /// layout of vector must be consistent with dof map range. This
    /// function is typically used to construct the null space of a
    /// matrix operator
    virtual void set(GenericVector& x, double value) const = 0;

    /// Return the map from unowned local dofmap nodes to global dofmap
    /// nodes. Dofmap node is dof index modulo block size.
    virtual const std::vector<std::size_t>& local_to_global_unowned() const = 0;

    /// Index map (const access)
    virtual std::shared_ptr<const IndexMap> index_map() const = 0;

    /// Tabulate map between local (process) and global dof indices
    virtual void tabulate_local_to_global_dofs(std::vector<std::size_t>& local_to_global_map) const = 0;

    /// Return global dof index corresponding to a given local index
    virtual std::size_t local_to_global_index(int local_index) const = 0;

    /// Return map from shared nodes to the processes (not including
    /// the current process) that share it.
    virtual const std::unordered_map<int, std::vector<int>>&
      shared_nodes() const = 0;

    /// Return set of processes that share dofs with the this process
    virtual const std::set<int>& neighbours() const = 0;

    /// Clear any data required to build sub-dofmaps (this is to
    /// reduce memory use)
    virtual void clear_sub_map_data() = 0;

    /// Return informal string representation (pretty-print)
    virtual std::string str(bool verbose) const = 0;

    /// Get block size
    virtual int block_size() const = 0;

    /// Subdomain mapping constrained boundaries, e.g. periodic
    /// conditions
    std::shared_ptr<const SubDomain> constrained_domain;

  };

}

#endif
