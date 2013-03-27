// Copyright (C) 2010-2013 Anders Logg and Garth N. Wells
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
// Modified by Jan Blechta, 2013
//
// First added:  2010-05-26
// Last changed: 2013-03-04

#ifndef __GENERIC_DOF_MAP_H
#define __GENERIC_DOF_MAP_H

#include <map>
#include <utility>
#include <vector>
#include <boost/multi_array.hpp>
#include <boost/shared_ptr.hpp>
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
  class Restriction;
  class SubDomain;

  /// This class provides a generic interface for dof maps

  class GenericDofMap : public Variable
  {
  public:

    /// Constructor
    GenericDofMap() : block_size(1) {}

    /// True if dof map is a view into another map (is a sub-dofmap)
    virtual bool is_view() const = 0;

    /// Return the dimension of the global finite element function space
    virtual std::size_t global_dimension() const = 0;

    /// Return the dimension of the local finite element function space
    /// on a cell
    virtual std::size_t cell_dimension(std::size_t index) const = 0;

    /// Return the maximum dimension of the local finite element
    /// function space
    virtual std::size_t max_cell_dimension() const = 0;

    /// Return the number of dofs for a given entity dimension
    virtual std::size_t num_entity_dofs(std::size_t dim) const = 0;

    /// Return the geometric dimension of the coordinates this dof map
    // provides
    virtual std::size_t geometric_dimension() const = 0;

    /// Return number of facet dofs
    virtual std::size_t num_facet_dofs() const = 0;

    /// Restriction if any. If the dofmap is not restricted, a null
    /// pointer is returned.
    virtual boost::shared_ptr<const Restriction> restriction() const = 0;

    /// Return the ownership range (dofs in this range are owned by
    /// this process)
    virtual std::pair<std::size_t, std::size_t> ownership_range() const = 0;

    /// Return map from nonlocal-dofs (that appear in local dof map)
    /// to owning process
    virtual const boost::unordered_map<std::size_t, unsigned int>&
      off_process_owner() const = 0;

    /// Local-to-global mapping of dofs on a cell
    virtual const std::vector<dolfin::la_index>& cell_dofs(std::size_t cell_index) const = 0;

    /// Tabulate local-local facet dofs
    virtual void tabulate_facet_dofs(std::vector<std::size_t>& dofs,
                                     std::size_t local_facet) const = 0;

    /// Tabulate the local-to-local mapping of dofs on entity (dim, local_entity)
    virtual void tabulate_entity_dofs(std::vector<std::size_t>& dofs,
				      std::size_t dim, std::size_t local_entity) const = 0;

    /// Return a map between vertices and dofs
    virtual std::vector<dolfin::la_index> dof_to_vertex_map(Mesh& mesh) const = 0;

    /// Return a map between vertices and dofs
    virtual std::vector<std::size_t> vertex_to_dof_map(Mesh& mesh) const = 0;

    /// Tabulate the coordinates of all dofs on a cell (UFC cell version)
    virtual void tabulate_coordinates(boost::multi_array<double, 2>& coordinates,
                                      const ufc::cell& ufc_cell) const = 0;

    /// Tabulate the coordinates of all dofs owned by this process
    virtual std::vector<double> tabulate_all_coordinates(const Mesh& mesh) const = 0;

    /// Create a copy of the dof map
    virtual boost::shared_ptr<GenericDofMap> copy() const = 0;

    /// Create a new dof map on new mesh
    virtual boost::shared_ptr<GenericDofMap> create(const Mesh& new_mesh) const = 0;

    /// Extract sub dofmap component
    virtual boost::shared_ptr<GenericDofMap>
        extract_sub_dofmap(const std::vector<std::size_t>& component,
                           const Mesh& mesh) const = 0;

    /// Create a "collapsed" a dofmap (collapses from a sub-dofmap view)
    virtual boost::shared_ptr<GenericDofMap>
        collapse(boost::unordered_map<std::size_t, std::size_t>& collapsed_map,
                 const Mesh& mesh) const = 0;

    /// Set dof entries in vector to a specified value. Parallel layout
    /// of vector must be consistent with dof map range.
    virtual void set(GenericVector& x, double value) const = 0;

    /// Set dof entries in vector to the value*x[i], where x[i] is the
    /// spatial coordinate of the dof. Parallel layout of vector must
    /// be consistent with dof map range.
    virtual void set_x(GenericVector& x, double value, std::size_t component,
                       const Mesh& mesh) const = 0;

    /// Return map from shared dofs to the processes (not including the current
    /// process) that share it.
    virtual const boost::unordered_map<std::size_t, std::vector<unsigned int> >&
      shared_dofs() const = 0;

    /// Return set of processes that share dofs with the this process
    virtual const std::set<std::size_t>& neighbours() const = 0;

    /// Return informal string representation (pretty-print)
    virtual std::string str(bool verbose) const = 0;

    /// Subdomain mapping constrained boundaries, e.g. periodic conditions
    boost::shared_ptr<const SubDomain> constrained_domain;

    /// Dofmap block size, e.g. 3 for 3D elasticity with a suitable ordered dofmao
    std::size_t block_size;

  };

}

#endif
