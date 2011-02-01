// Copyright (C) 2010 Anders Logg and Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2010-05-26
// Last changed:

#ifndef __GENERIC_DOF_MAP_H
#define __GENERIC_DOF_MAP_H

#include <map>
#include <utility>
#include <vector>
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
  class Mesh;
  template<class T> class Set;

  /// This class provides a generic interface for dof maps

  class GenericDofMap : public Variable
  {
  public:

    /// Return a string identifying the dof map
    virtual std::string signature() const = 0;

    /// Return true iff mesh entities of topological dimension d are needed
    virtual bool needs_mesh_entities(unsigned int d) const = 0;

    /// Return the dimension of the global finite element function space
    virtual unsigned int global_dimension() const = 0;

    /// Return the dimension of the local (process) finite element function space
    virtual unsigned int local_dimension() const = 0;

    /// Return the dimension of the local finite element function space on a
    /// cell
    virtual unsigned int dimension(uint index) const = 0;

    /// Return the maximum dimension of the local finite element function space
    virtual unsigned int max_local_dimension() const = 0;

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
    virtual void tabulate_coordinates(double** coordinates,
                                      const ufc::cell& ufc_cell) const = 0;

    /// Tabulate the coordinates of all dofs on a cell (DOLFIN cell version)
    virtual void tabulate_coordinates(double** coordinates,
                                      const Cell& cell) const = 0;

    /// Extract sub dofmap component
    virtual GenericDofMap* extract_sub_dofmap(const std::vector<uint>& component,
                                              const Mesh& dolfin_mesh) const = 0;

    /// "Collapse" a sub dofmap
    virtual GenericDofMap* collapse(std::map<uint, uint>& collapsed_map,
                                    const Mesh& dolfin_mesh) const = 0;

    /// Return the set of dof indices
    virtual boost::unordered_set<dolfin::uint> dofs() const = 0;

    /// Re-number based on provided re-numbering map
    virtual void renumber(const std::vector<uint>& renumbering_map) = 0;

    /// Return informal string representation (pretty-print)
    virtual std::string str(bool verbose) const = 0;

  };

}

#endif
