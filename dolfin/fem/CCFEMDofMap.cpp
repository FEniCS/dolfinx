// Copyright (C) 2013 Anders Logg
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
// First added:  2013-09-19
// Last changed: 2013-10-22

#include <dolfin/common/NoDeleter.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/function/CCFEMFunctionSpace.h>
#include "CCFEMDofMap.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
CCFEMDofMap::CCFEMDofMap()
{
  clear();
}
//-----------------------------------------------------------------------------
CCFEMDofMap::CCFEMDofMap(const CCFEMDofMap& dofmap)
{
  _global_dimension = dofmap._global_dimension;
  _dofmaps = dofmap._dofmaps;
  _current_part = dofmap._current_part;
}
//-----------------------------------------------------------------------------
CCFEMDofMap::~CCFEMDofMap()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
std::size_t CCFEMDofMap::num_parts() const
{
  return _dofmaps.size();
}
//-----------------------------------------------------------------------------
std::shared_ptr<const GenericDofMap> CCFEMDofMap::part(std::size_t i) const
{
  dolfin_assert(i < _dofmaps.size());
  return _dofmaps[i];
}
//-----------------------------------------------------------------------------
void CCFEMDofMap::set_current_part(std::size_t part) const
{
  dolfin_assert(part < num_parts());
  _current_part = part; // mutable
}
//-----------------------------------------------------------------------------
void CCFEMDofMap::add(std::shared_ptr<const GenericDofMap> dofmap)
{
  _dofmaps.push_back(dofmap);
  log(PROGRESS, "Added dofmap to CCFEM dofmap; dofmap has %d part(s).",
      _dofmaps.size());
}
//-----------------------------------------------------------------------------
void CCFEMDofMap::add(const GenericDofMap& dofmap)
{
  add(reference_to_no_delete_pointer(dofmap));
}
//-----------------------------------------------------------------------------
void CCFEMDofMap::build(const CCFEMFunctionSpace& function_space)
{
  begin(PROGRESS, "Building CCFEM dofmap.");

  // Compute global dimension
  begin(PROGRESS, "Computing total dimension.");
  _global_dimension = 0;
  for (std::size_t i = 0; i < num_parts(); i++)
  {
    const std::size_t d = _dofmaps[i]->global_dimension();
    _global_dimension += d;
    log(PROGRESS, "dim(V_%d) = %d", i, d);
  }
  end();
  log(PROGRESS, "Total global dimension is %d.", _global_dimension);

  // For now, we build the simplest possible dofmap by reusing the
  // dofmaps for each part and adding offsets in between.

  // Build dofmap
  _dofmap.clear();
  dolfin::la_index offset = 0;
  for (std::size_t part = 0; part < num_parts(); part++)
  {
    log(PROGRESS, "Computing dofs for part %d.", part);

    dolfin_assert(_dofmaps[part]);
    std::vector<std::vector<dolfin::la_index> > dofmap_part;

    // Get mesh on current part
    const Mesh& mesh = *function_space.part(part)->mesh();

    // Add all dofs for current part with offset
    for (CellIterator cell(mesh); !cell.end(); ++cell)
    {
      // Get dofs from dofmap on part
      const std::vector<dolfin::la_index>& dofs
        = _dofmaps[part]->cell_dofs(cell->index());

      // Compute new dofs by adding offset
      std::vector<dolfin::la_index> new_dofs;
      for (std::size_t i = 0; i < dofs.size(); i++)
        new_dofs.push_back(dofs[i] + offset);

      // Store dofs for cell
      dofmap_part.push_back(new_dofs);
    }

    // Store dofs for part
    _dofmap.push_back(dofmap_part);

    // Increase offset
    offset += _dofmaps[part]->global_dimension();
  }

  end();
}
//-----------------------------------------------------------------------------
void CCFEMDofMap::clear()
{
  _global_dimension = 0;
  _dofmaps.clear();
  _current_part = 0;
  _dofmap.clear();
}
//-----------------------------------------------------------------------------
// Implementation of the GenericDofMap interface
//-----------------------------------------------------------------------------
bool CCFEMDofMap::is_view() const
{
  return false;
}
//-----------------------------------------------------------------------------
std::size_t CCFEMDofMap::global_dimension() const
{
  return _global_dimension;
}
//-----------------------------------------------------------------------------
std::size_t CCFEMDofMap::cell_dimension(std::size_t index) const
{
  dolfin_assert(_current_part < _dofmaps.size() && _dofmaps[_current_part]);
  return _dofmaps[_current_part]->cell_dimension(index);
}
//-----------------------------------------------------------------------------
std::size_t CCFEMDofMap::max_cell_dimension() const
{
  dolfin_assert(_current_part < _dofmaps.size() && _dofmaps[_current_part]);
  return _dofmaps[_current_part]->max_cell_dimension();
}
//-----------------------------------------------------------------------------
std::size_t CCFEMDofMap::num_entity_dofs(std::size_t dim) const
{
  dolfin_assert(_current_part < _dofmaps.size() && _dofmaps[_current_part]);
  return _dofmaps[_current_part]->num_entity_dofs(dim);
}
//-----------------------------------------------------------------------------
std::size_t CCFEMDofMap::geometric_dimension() const
{
  dolfin_assert(_current_part < _dofmaps.size() && _dofmaps[_current_part]);
  return _dofmaps[_current_part]->geometric_dimension();
}
//-----------------------------------------------------------------------------
std::size_t CCFEMDofMap::num_facet_dofs() const
{
  dolfin_assert(_current_part < _dofmaps.size() && _dofmaps[_current_part]);
  return _dofmaps[_current_part]->num_facet_dofs();
}
//-----------------------------------------------------------------------------
std::shared_ptr<const Restriction> CCFEMDofMap::restriction() const
{
  // FIXME: Restrictions are unhandled but we need to return something
  dolfin_assert(_current_part < _dofmaps.size() && _dofmaps[_current_part]);
  return _dofmaps[_current_part]->restriction();
}
//-----------------------------------------------------------------------------
std::pair<std::size_t, std::size_t> CCFEMDofMap::ownership_range() const
{
  // FIXME: Does not run in parallel
  return std::make_pair<std::size_t, std::size_t>(0, global_dimension());
}
//-----------------------------------------------------------------------------
const boost::unordered_map<std::size_t, unsigned int>&
CCFEMDofMap::off_process_owner() const
{
  dolfin_assert(_current_part < _dofmaps.size() && _dofmaps[_current_part]);
  return _dofmaps[_current_part]->off_process_owner();
}
//-----------------------------------------------------------------------------
const std::vector<dolfin::la_index>&
CCFEMDofMap::cell_dofs(std::size_t cell_index) const
{
  dolfin_assert(cell_index < _dofmap[_current_part].size());
  return _dofmap[_current_part][cell_index];
}
//-----------------------------------------------------------------------------
void CCFEMDofMap::tabulate_facet_dofs(std::vector<std::size_t>& dofs,
                                      std::size_t local_facet) const
{
  dolfin_assert(_current_part < _dofmaps.size() && _dofmaps[_current_part]);
  return _dofmaps[_current_part]->tabulate_facet_dofs(dofs, local_facet);
}
//-----------------------------------------------------------------------------
void CCFEMDofMap::tabulate_entity_dofs(std::vector<std::size_t>& dofs,
                                       std::size_t dim,
                                       std::size_t local_entity) const
{
  dolfin_assert(_current_part < _dofmaps.size() && _dofmaps[_current_part]);
  return _dofmaps[_current_part]->tabulate_entity_dofs(dofs, dim, local_entity);
}
//-----------------------------------------------------------------------------
std::vector<dolfin::la_index>
CCFEMDofMap::dof_to_vertex_map(const Mesh& mesh) const
{
  dolfin_assert(_current_part < _dofmaps.size() && _dofmaps[_current_part]);
  return _dofmaps[_current_part]->dof_to_vertex_map(mesh);
}
//-----------------------------------------------------------------------------
std::vector<std::size_t>
CCFEMDofMap::vertex_to_dof_map(const Mesh& mesh) const
{
  dolfin_assert(_current_part < _dofmaps.size() && _dofmaps[_current_part]);
  return _dofmaps[_current_part]->vertex_to_dof_map(mesh);
}
//-----------------------------------------------------------------------------
void
CCFEMDofMap::tabulate_coordinates(boost::multi_array<double, 2>& coordinates,
                                  const std::vector<double>& vertex_coordinates,
                                  const Cell& cell) const
{
  dolfin_assert(_current_part < _dofmaps.size() && _dofmaps[_current_part]);
  _dofmaps[_current_part]->tabulate_coordinates(coordinates, vertex_coordinates,
                                                cell);
}
//-----------------------------------------------------------------------------
std::vector<double>
CCFEMDofMap::tabulate_all_coordinates(const Mesh& mesh) const
{
  dolfin_assert(_current_part < _dofmaps.size() && _dofmaps[_current_part]);
  return _dofmaps[_current_part]->tabulate_all_coordinates(mesh);
}
//-----------------------------------------------------------------------------
std::shared_ptr<GenericDofMap> CCFEMDofMap::copy() const
{
  return std::shared_ptr<GenericDofMap>(new CCFEMDofMap(*this));
}
//-----------------------------------------------------------------------------
std::shared_ptr<GenericDofMap>
CCFEMDofMap::create(const Mesh& new_mesh) const
{
  dolfin_not_implemented();
  return copy(); // need to return something
}
//-----------------------------------------------------------------------------
std::shared_ptr<GenericDofMap>
CCFEMDofMap::extract_sub_dofmap(const std::vector<std::size_t>& component,
                                const Mesh& mesh) const
{
  dolfin_not_implemented();
  return copy(); // need to return something
}
//-----------------------------------------------------------------------------
std::shared_ptr<GenericDofMap>
CCFEMDofMap::collapse(boost::unordered_map<std::size_t, std::size_t>& collapsed_map,
                      const Mesh& mesh) const
{
  dolfin_not_implemented();
  return copy(); // need to return something
}
//-----------------------------------------------------------------------------
std::vector<dolfin::la_index> CCFEMDofMap::dofs() const
{
  dolfin_not_implemented();
  return std::vector<dolfin::la_index>();
}
//-----------------------------------------------------------------------------
void CCFEMDofMap::set(GenericVector& x, double value) const
{
  dolfin_not_implemented();
}
//-----------------------------------------------------------------------------
void CCFEMDofMap::set_x(GenericVector& x, double value, std::size_t component,
           const Mesh& mesh) const
{
  dolfin_not_implemented();
}
//-----------------------------------------------------------------------------
const boost::unordered_map<std::size_t,
                           std::vector<unsigned int> >& CCFEMDofMap::shared_dofs() const
{
  dolfin_not_implemented();

  // Need to return a reference to something
  dolfin_assert(_current_part < _dofmaps.size() && _dofmaps[_current_part]);
  return _dofmaps[_current_part]->shared_dofs();
}
//-----------------------------------------------------------------------------
const std::set<std::size_t>& CCFEMDofMap::neighbours() const
{
  dolfin_not_implemented();

  // Need to return a reference to something
  dolfin_assert(_current_part < _dofmaps.size() && _dofmaps[_current_part]);
  return _dofmaps[_current_part]->neighbours();
}
//-----------------------------------------------------------------------------
std::string CCFEMDofMap::str(bool verbose) const
{
  std::stringstream s;
  s << "<CCFEMDofMap with "
    << num_parts()
    << " parts and total global dimension "
    << global_dimension()
    << ">"
    << std::endl;
  return s.str();
}
//-----------------------------------------------------------------------------
