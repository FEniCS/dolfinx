// Copyright (C) 2007-2013 Anders Logg and Garth N. Wells
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
// Modified by Martin Alnes, 2008, 2013
// Modified by Kent-Andre Mardal, 2009
// Modified by Ola Skavhaug, 2009
// Modified by Niclas Jansson, 2009
// Modified by Joachim B Haga, 2012
// Modified by Mikael Mortensen, 2012
//
// First added:  2007-03-01
// Last changed: 2013-01-08

#include <boost/unordered_map.hpp>
#include <dolfin/common/MPI.h>
#include <dolfin/common/NoDeleter.h>
#include <dolfin/common/Set.h>
#include <dolfin/common/types.h>
#include <dolfin/la/GenericVector.h>
#include <dolfin/log/LogStream.h>
#include <dolfin/mesh/BoundaryMesh.h>
#include <dolfin/mesh/MeshDistributed.h>
#include <dolfin/mesh/Restriction.h>
#include <dolfin/parameter/GlobalParameters.h>
#include "DofMapBuilder.h"
#include "UFCCell.h"
#include "DofMap.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
DofMap::DofMap(boost::shared_ptr<const ufc::dofmap> ufc_dofmap,
  const Mesh& mesh) : _ufc_dofmap(ufc_dofmap),  _global_dimension(0),
  _ufc_offset(0)
{
  dolfin_assert(_ufc_dofmap);

  // Call dofmap builder
  const std::map<std::size_t, std::pair<std::size_t, std::size_t> > slave_to_master_facets;
  DofMapBuilder::build(*this, mesh, _restriction, slave_to_master_facets);
}
//-----------------------------------------------------------------------------
DofMap::DofMap(boost::shared_ptr<const ufc::dofmap> ufc_dofmap,
               boost::shared_ptr<const Restriction> restriction)
  : _ufc_dofmap(ufc_dofmap), _restriction(restriction), _global_dimension(0),
    _ufc_offset(0)
{
  dolfin_assert(_ufc_dofmap);
  dolfin_assert(_restriction);

  // Get mesh
  const dolfin::Mesh& mesh = restriction->mesh();

  // Check that we get cell markers, extend later
  if (restriction->dim() != mesh.topology().dim())
  {
    dolfin_error("DofMap.cpp",
                 "create mapping of degrees of freedom",
                 "Only cell-based restriction of function spaces are currently supported. ");
  }

  // Call dofmap builder
  const std::map<std::size_t, std::pair<std::size_t, std::size_t> > slave_to_master_facets;
  DofMapBuilder::build(*this, mesh, restriction, slave_to_master_facets);
}
//-----------------------------------------------------------------------------
DofMap::DofMap(const DofMap& parent_dofmap,
  const std::vector<std::size_t>& component, const Mesh& mesh)
  : _global_dimension(0), _ufc_offset(0), _ownership_range(0, 0)
{
  // Note: Ownership range is set to zero since dofmap is a view

  // Build sub-dofmap
  DofMapBuilder::build_sub_map(*this, parent_dofmap, component, mesh);
}
//-----------------------------------------------------------------------------
DofMap::DofMap(boost::unordered_map<std::size_t, std::size_t>& collapsed_map,
               const DofMap& dofmap_view, const Mesh& mesh)
   :  _ufc_dofmap(dofmap_view._ufc_dofmap), _global_dimension(0), _ufc_offset(0)
{
  dolfin_assert(_ufc_dofmap);

  // Check for dimensional consistency between the dofmap and mesh
  check_dimensional_consistency(*_ufc_dofmap, mesh);

  // Check that mesh has been ordered
  if (!mesh.ordered())
  {
     dolfin_error("DofMap.cpp",
                  "create mapping of degrees of freedom",
                  "Mesh is not ordered according to the UFC numbering convention. "
                  "Consider calling mesh.order()");
  }

  // Check dimensional consistency between UFC dofmap and the mesh
  check_provided_entities(*_ufc_dofmap, mesh);

  // Build dof map
  const std::map<std::size_t, std::pair<std::size_t, std::size_t> > slave_to_master_facets;
  DofMapBuilder::build(*this, mesh, _restriction, slave_to_master_facets);

  // Dimension checks
  dolfin_assert(dofmap_view._dofmap.size() == mesh.num_cells());
  dolfin_assert(global_dimension() == dofmap_view.global_dimension());
  dolfin_assert(_dofmap.size() == mesh.num_cells());

  // FIXME: Could we use a std::vector instead of std::map if the collapsed
  //        dof map is contiguous (0, . . . , n)?

  // Build map from collapsed dof index to original dof index
  collapsed_map.clear();
  for (std::size_t i = 0; i < mesh.num_cells(); ++i)
  {
    const std::vector<dolfin::la_index>& view_cell_dofs = dofmap_view._dofmap[i];
    const std::vector<dolfin::la_index>& cell_dofs = _dofmap[i];
    dolfin_assert(view_cell_dofs.size() == cell_dofs.size());

    for (std::size_t j = 0; j < view_cell_dofs.size(); ++j)
      collapsed_map[cell_dofs[j]] = view_cell_dofs[j];
  }
}
//-----------------------------------------------------------------------------
DofMap::DofMap(const DofMap& dofmap)
{
  // Copy data
  _dofmap = dofmap._dofmap;
  _ufc_dofmap = dofmap._ufc_dofmap;
  ufc_map_to_dofmap = dofmap.ufc_map_to_dofmap;
  _global_dimension = dofmap._global_dimension;
  _ufc_offset = dofmap._ufc_offset;
  _ownership_range = dofmap._ownership_range;
  _off_process_owner = dofmap._off_process_owner;
  _shared_dofs = dofmap._shared_dofs;
  _neighbours = dofmap._neighbours;
  //_slave_master_map = dofmap._slave_master_map;
  //_master_processes = dofmap._master_processes;
}
//-----------------------------------------------------------------------------
DofMap::~DofMap()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
std::size_t DofMap::global_dimension() const
{
  return _global_dimension;
}
//-----------------------------------------------------------------------------
std::size_t DofMap::cell_dimension(std::size_t cell_index) const
{
  dolfin_assert(cell_index < _dofmap.size());
  return _dofmap[cell_index].size();
}
//-----------------------------------------------------------------------------
std::size_t DofMap::max_cell_dimension() const
{
  dolfin_assert(_ufc_dofmap);
  return _ufc_dofmap->max_local_dimension();
}
//-----------------------------------------------------------------------------
std::size_t DofMap::geometric_dimension() const
{
  dolfin_assert(_ufc_dofmap);
  return _ufc_dofmap->geometric_dimension();
}
//-----------------------------------------------------------------------------
std::size_t DofMap::num_facet_dofs() const
{
  dolfin_assert(_ufc_dofmap);
  return _ufc_dofmap->num_facet_dofs();
}
//-----------------------------------------------------------------------------
boost::shared_ptr<const dolfin::Restriction> DofMap::restriction() const
{
  return _restriction;
}
//-----------------------------------------------------------------------------
std::pair<std::size_t, std::size_t> DofMap::ownership_range() const
{
  if (_ownership_range.first == 0 && _ownership_range.second == 0)
  {
    dolfin_error("DofMap.cpp",
                 "access ownership range of degree of freedom mapping",
                 "Cannot determine ownership range for sub-dofmaps");
  }

  return _ownership_range;
}
//-----------------------------------------------------------------------------
const boost::unordered_map<std::size_t, std::size_t>& DofMap::off_process_owner() const
{
  return _off_process_owner;
}
//-----------------------------------------------------------------------------
const boost::unordered_map<std::size_t, std::vector<std::size_t> >& DofMap::shared_dofs() const
{
  return _shared_dofs;
}
//-----------------------------------------------------------------------------
const std::set<std::size_t>& DofMap::neighbours() const
{
  return _neighbours;
}
//-----------------------------------------------------------------------------
void DofMap::tabulate_facet_dofs(std::size_t* dofs, std::size_t local_facet) const
{
  dolfin_assert(_ufc_dofmap);
  _ufc_dofmap->tabulate_facet_dofs(dofs, local_facet);
}
//-----------------------------------------------------------------------------
void DofMap::tabulate_coordinates(boost::multi_array<double, 2>& coordinates,
                                  const ufc::cell& ufc_cell) const
{
  // FIXME: This is a hack because UFC wants a double pointer for coordinates
  dolfin_assert(_ufc_dofmap);

  // Check dimensions
  if (coordinates.shape()[0] != cell_dimension(ufc_cell.index) ||
      coordinates.shape()[1] != _ufc_dofmap->geometric_dimension())
  {
    boost::multi_array<double, 2>::extent_gen extents;
    const std::size_t cell_dim = cell_dimension(ufc_cell.index);
    coordinates.resize(extents[cell_dim][_ufc_dofmap->geometric_dimension()]);
  }

  // Set vertex coordinates
  const std::size_t num_points = coordinates.size();
  std::vector<double*> coords(num_points);
  for (std::size_t i = 0; i < num_points; ++i)
    coords[i] = &(coordinates[i][0]);

  // Tabulate coordinates
  _ufc_dofmap->tabulate_coordinates(&coords[0], ufc_cell);
}
//-----------------------------------------------------------------------------
void DofMap::tabulate_coordinates(boost::multi_array<double, 2>& coordinates,
                                  const Cell& cell) const
{
  UFCCell ufc_cell(cell);
  tabulate_coordinates(coordinates, ufc_cell);
}
//-----------------------------------------------------------------------------
boost::shared_ptr<GenericDofMap> DofMap::copy() const
{
  return boost::shared_ptr<GenericDofMap>(new DofMap(*this));
}
//-----------------------------------------------------------------------------
boost::shared_ptr<GenericDofMap> DofMap::create(const Mesh& new_mesh) const
{
  // Get underlying UFC dof map
  boost::shared_ptr<const ufc::dofmap> ufc_dof_map(_ufc_dofmap);
  return boost::shared_ptr<GenericDofMap>(new DofMap(ufc_dof_map, new_mesh));
}
//-----------------------------------------------------------------------------
DofMap* DofMap::extract_sub_dofmap(const std::vector<std::size_t>& component,
                                   const Mesh& mesh) const
{
  return new DofMap(*this, component, mesh);
}
//-----------------------------------------------------------------------------
DofMap* DofMap::collapse(boost::unordered_map<std::size_t, std::size_t>& collapsed_map,
                         const Mesh& mesh) const
{
  return new DofMap(collapsed_map, *this, mesh);
}
//-----------------------------------------------------------------------------
void DofMap::set(GenericVector& x, double value) const
{
  std::vector<std::vector<dolfin::la_index> >::const_iterator cell_dofs;
  for (cell_dofs = _dofmap.begin(); cell_dofs != _dofmap.end(); ++cell_dofs)
  {
    std::vector<double> _value(cell_dofs->size(), value);
    x.set(_value.data(), cell_dofs->size(), cell_dofs->data());
  }
  x.apply("add");
}
//-----------------------------------------------------------------------------
void DofMap::set_x(GenericVector& x, double value, std::size_t component,
                   const Mesh& mesh) const
{
  std::vector<double> x_values;
  boost::multi_array<double, 2> coordinates;
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    // Get local-to-global map
    const std::vector<dolfin::la_index>& dofs = cell_dofs(cell->index());

    // Tabulate dof coordinates
    tabulate_coordinates(coordinates, *cell);
    dolfin_assert(coordinates.shape()[0] == dofs.size());
    dolfin_assert(component < coordinates.shape()[1]);

    // Copy coordinate (it may be possible to avoid this)
    x_values.resize(dofs.size());
    for (std::size_t i = 0; i < coordinates.shape()[0]; ++i)
      x_values[i] = value*coordinates[i][component];

    // Set x[component] values in vector
    x.set(x_values.data(), dofs.size(), dofs.data());
  }
}
//-----------------------------------------------------------------------------
void DofMap::check_provided_entities(const ufc::dofmap& dofmap,
                                     const Mesh& mesh)
{
  // Check that we have all mesh entities
  for (std::size_t d = 0; d <= mesh.topology().dim(); ++d)
  {
    if (dofmap.needs_mesh_entities(d) && mesh.num_entities(d) == 0)
    {
      dolfin_error("DofMap.cpp",
                   "initialize mapping of degrees of freedom",
                   "Missing entities of dimension %d. Try calling mesh.init(%d)", d, d);
    }
  }
}
//-----------------------------------------------------------------------------
boost::unordered_set<std::size_t> DofMap::dofs() const
{
  // Build set of dofs
  boost::unordered_set<std::size_t> dof_list;
  std::vector<std::vector<dolfin::la_index> >::const_iterator cell_dofs;
  for (cell_dofs = _dofmap.begin(); cell_dofs != _dofmap.end(); ++cell_dofs)
    dof_list.insert(cell_dofs->begin(), cell_dofs->end());

  return dof_list;
}
//-----------------------------------------------------------------------------
std::string DofMap::str(bool verbose) const
{
  // TODO: Display information on parallel stuff

  // Prefix with process number if running in parallel
  std::stringstream prefix;
  if (MPI::num_processes() > 1)
    prefix << "Process " << MPI::process_number() << ": ";

  std::stringstream s;
  s << prefix.str() << "<DofMap of global dimension " << global_dimension() << ">" << std::endl;
  if (verbose)
  {
    // Cell loop
    for (std::size_t i = 0; i < _dofmap.size(); ++i)
    {
      s << prefix.str() << "Local cell index, cell dofmap dimension: " << i << ", " << _dofmap[i].size() << std::endl;

      // Local dof loop
      for (std::size_t j = 0; j < _dofmap[i].size(); ++j)
        s << prefix.str() <<  "  " << "Local, global dof indices: " << j << ", " << _dofmap[i][j] << std::endl;
    }
  }

  return s.str();
}
//-----------------------------------------------------------------------------
void DofMap::check_dimensional_consistency(const ufc::dofmap& dofmap,
                                            const Mesh& mesh)
{
  // Check geometric dimension
  if (dofmap.geometric_dimension() != mesh.geometry().dim())
  {
    dolfin_error("DofMap.cpp",
                 "create mapping of degrees of freedom",
                 "Geometric dimension of the UFC dofmap (dim = %d) and the mesh (dim = %d) do not match",
                 dofmap.geometric_dimension(),
                 mesh.geometry().dim());
  }

  // Check topological dimension
  if (dofmap.topological_dimension() != mesh.topology().dim())
  {
    dolfin_error("DofMap.cpp",
                 "create mapping of degrees of freedom",
                 "Topological dimension of the UFC dofmap (dim = %d) and the mesh (dim = %d) do not match",
                 dofmap.topological_dimension(),
                 mesh.topology().dim());
  }
}
//-----------------------------------------------------------------------------
