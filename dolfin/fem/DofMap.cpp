// Copyright (C) 2007-2012 Anders Logg and Garth N. Wells
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
// Modified by Martin Alnes, 2008
// Modified by Kent-Andre Mardal, 2009
// Modified by Ola Skavhaug, 2009
// Modified by Niclas Jansson, 2009
// Modified by Joachim B Haga, 2012
//
// First added:  2007-03-01
// Last changed: 2012-11-05

#include <dolfin/common/MPI.h>
#include <dolfin/common/NoDeleter.h>
#include <dolfin/common/Set.h>
#include <dolfin/common/types.h>
#include <dolfin/la/GenericVector.h>
#include <dolfin/log/LogStream.h>
#include <dolfin/mesh/BoundaryMesh.h>
#include <dolfin/mesh/MeshPartitioning.h>
#include <dolfin/mesh/Restriction.h>
#include <dolfin/parameter/GlobalParameters.h>
#include "DofMapBuilder.h"
#include "UFCCell.h"
#include "UFCMesh.h"
#include "DofMap.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
DofMap::DofMap(boost::shared_ptr<const ufc::dofmap> ufc_dofmap,
               const Mesh& dolfin_mesh)
  : _ufc_dofmap(ufc_dofmap->create()),
    _global_dimension(0), _ufc_offset(0), _is_view(false),
    _distributed(MPI::num_processes() > 1)
{
  dolfin_assert(_ufc_dofmap);

  // Check for dimensional consistency between the dofmap and mesh
  check_dimensional_consistency(*_ufc_dofmap, dolfin_mesh);

  // Check that mesh has been ordered
  if (!dolfin_mesh.ordered())
  {
     dolfin_error("DofMap.cpp",
                  "create mapping of degrees of freedom",
                  "Mesh is not ordered according to the UFC numbering convention. "
                  "Consider calling mesh.order()");
  }

  // Generate and number all mesh entities
  const std::size_t D = dolfin_mesh.topology().dim();
  for (std::size_t d = 1; d <= D; ++d)
  {
    if (_ufc_dofmap->needs_mesh_entities(d) || (_distributed && d == (D - 1)))
    {
      dolfin_mesh.init(d);
      if (_distributed)
        MeshPartitioning::number_entities(dolfin_mesh, d);
    }
  }

  // Create the UFC mesh
  const UFCMesh ufc_mesh(dolfin_mesh);

  // Initialize the UFC dofmap
  init_ufc_dofmap(*_ufc_dofmap, ufc_mesh, dolfin_mesh);

  // Build dof map
  const bool reorder = dolfin::parameters["reorder_dofs_serial"];
  DofMapBuilder::build(*this, dolfin_mesh, ufc_mesh, _restriction,
                       reorder, _distributed);
}
//-----------------------------------------------------------------------------
DofMap::DofMap(boost::shared_ptr<const ufc::dofmap> ufc_dofmap,
               boost::shared_ptr<const Restriction> restriction)
  : _ufc_dofmap(ufc_dofmap->create()),
    _restriction(restriction),
    _global_dimension(0), _ufc_offset(0), _is_view(false),
    _distributed(MPI::num_processes() > 1)
{
  info("Creating restricted dofmap.");
  warning("Restricted function space is an experimental feature.");
  not_working_in_parallel("Restricted function space");

  dolfin_assert(_ufc_dofmap);
  dolfin_assert(_restriction);

  // Get mesh
  const dolfin::Mesh& dolfin_mesh(restriction->mesh());

  // Check for dimensional consistency between the dofmap and mesh
  check_dimensional_consistency(*_ufc_dofmap, dolfin_mesh);

  // Check that mesh has been ordered
  if (!dolfin_mesh.ordered())
  {
     dolfin_error("DofMap.cpp",
                  "create mapping of degrees of freedom",
                  "Mesh is not ordered according to the UFC numbering convention. "
                  "Consider calling mesh.order()");
  }

  // Check that we get cell markers, extend later
  const std::size_t D = dolfin_mesh.topology().dim();
  if (restriction->dim() != D)
  {
    dolfin_error("DofMap.cpp",
                 "create mapping of degrees of freedom",
                 "Only cell-based restricted function spaces are currently supported. ");
  }

  // Generate and number all mesh entities
  for (std::size_t d = 1; d <= D; ++d)
  {
    if (_ufc_dofmap->needs_mesh_entities(d) || (_distributed && d == (D - 1)))
    {
      dolfin_mesh.init(d);
      if (_distributed)
        MeshPartitioning::number_entities(dolfin_mesh, d);
    }
  }

  // Create the UFC mesh
  const UFCMesh ufc_mesh(dolfin_mesh);

  // Initialize the UFC dofmap
  init_ufc_dofmap(*_ufc_dofmap, ufc_mesh, dolfin_mesh);

  // Build restricted dof map
  const bool reorder = dolfin::parameters["reorder_dofs_serial"];
  DofMapBuilder::build(*this, dolfin_mesh, ufc_mesh, restriction,
                       reorder, _distributed);
}
//-----------------------------------------------------------------------------
DofMap::DofMap(const DofMap& parent_dofmap, const std::vector<std::size_t>& component,
               const Mesh& mesh, bool distributed) :
  _global_dimension(0), _ufc_offset(0),
  _ownership_range(0, 0), _is_view(true),
  _distributed(distributed)
{
  // Note: Ownership range is set to zero since dofmap is a view

  dolfin_assert(!component.empty());

  // Create UFC mesh
  const UFCMesh ufc_mesh(mesh);

  // Initialise offset from parent
  std::size_t offset = parent_dofmap._ufc_offset;

  // Get parent UFC dof map
  const ufc::dofmap& parent_ufc_dofmap = *(parent_dofmap._ufc_dofmap);

  // Extract ufc sub-dofmap from parent and get offset
  _ufc_dofmap.reset(extract_ufc_sub_dofmap(parent_ufc_dofmap, offset,
                                           component, ufc_mesh, mesh));
  dolfin_assert(_ufc_dofmap);

  // Check for dimensional consistency between the dofmap and mesh
  check_dimensional_consistency(*_ufc_dofmap, mesh);

  // Set UFC offset
  this->_ufc_offset = offset;

  // Initialise UFC dofmap
  init_ufc_dofmap(*_ufc_dofmap, ufc_mesh, mesh);

  // Resize dofmap data structure
  _dofmap.resize(mesh.num_cells());

  // Hack to get around UFC using 32-bit integers
  std::vector<unsigned int> tmp_dof_holder;

  // Build sub-map based on UFC dofmap
  UFCCell ufc_cell(mesh);
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    const std::size_t cell_index = cell->index();

    // Update to current cell
    ufc_cell.update(*cell);

    // Resize list for cell
    _dofmap[cell_index].resize(_ufc_dofmap->local_dimension(ufc_cell));
    tmp_dof_holder.resize(_ufc_dofmap->local_dimension(ufc_cell));

    // Tabulate sub-dofs on cell (using UFC map)
    _ufc_dofmap->tabulate_dofs(&tmp_dof_holder[0], ufc_mesh, ufc_cell);
    std::copy(tmp_dof_holder.begin(), tmp_dof_holder.end(), _dofmap[cell_index].begin());

    // Add UFC offset
    for (std::size_t i = 0; i < _dofmap[cell_index].size(); ++i)
      _dofmap[cell_index][i] += offset;
  }

  // Set global dimension
  _global_dimension = _ufc_dofmap->global_dimension();

  // Modify dofmap for non-UFC numbering
  ufc_map_to_dofmap.clear();
  _off_process_owner.clear();
  _shared_dofs.clear();
  _neighbours.clear();
  if (!parent_dofmap.ufc_map_to_dofmap.empty())
  {
    boost::unordered_map<std::size_t, std::size_t>::const_iterator ufc_to_current_dof;
    std::vector<std::vector<DolfinIndex> >::iterator cell_map;
    std::vector<DolfinIndex>::iterator dof;
    for (cell_map = _dofmap.begin(); cell_map != _dofmap.end(); ++cell_map)
    {
      for (dof = cell_map->begin(); dof != cell_map->end(); ++dof)
      {
        // Get dof index
        ufc_to_current_dof = parent_dofmap.ufc_map_to_dofmap.find(*dof);
        dolfin_assert(ufc_to_current_dof != parent_dofmap.ufc_map_to_dofmap.end());

        // Add to ufc-to-current dof map
        ufc_map_to_dofmap.insert(*ufc_to_current_dof);

        // Set dof index
        *dof = ufc_to_current_dof->second;

        // Add to off-process dof owner map
        boost::unordered_map<std::size_t, std::size_t>::const_iterator parent_off_proc = parent_dofmap._off_process_owner.find(*dof);
        if (parent_off_proc != parent_dofmap._off_process_owner.end())
          _off_process_owner.insert(*parent_off_proc);

        // Add to shared-dof process map, and update the set of neighbours
        boost::unordered_map<std::size_t, std::vector<std::size_t> >::const_iterator parent_shared = parent_dofmap._shared_dofs.find(*dof);
        if (parent_shared != parent_dofmap._shared_dofs.end())
        {
          _shared_dofs.insert(*parent_shared);
          _neighbours.insert(parent_shared->second.begin(), parent_shared->second.end());
        }
      }
    }
  }
}
//-----------------------------------------------------------------------------
DofMap::DofMap(boost::unordered_map<std::size_t, std::size_t>& collapsed_map,
               const DofMap& dofmap_view, const Mesh& mesh, bool distributed)
             :
  _ufc_dofmap(dofmap_view._ufc_dofmap->create()),
  _global_dimension(0), _ufc_offset(0),
  _is_view(false), _distributed(distributed)
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

  // Create the UFC mesh
  const UFCMesh ufc_mesh(mesh);

  // Initialize the UFC dofmap
  init_ufc_dofmap(*_ufc_dofmap, ufc_mesh, mesh);

  // Build dof map
  const bool reorder = dolfin::parameters["reorder_dofs_serial"];
  DofMapBuilder::build(*this, mesh, ufc_mesh, _restriction, reorder, _distributed);

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
    const std::vector<DolfinIndex>& view_cell_dofs = dofmap_view._dofmap[i];
    const std::vector<DolfinIndex>& cell_dofs = _dofmap[i];
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
  _ufc_dofmap.reset(dofmap._ufc_dofmap->create());
  ufc_map_to_dofmap = dofmap.ufc_map_to_dofmap;
  _global_dimension = dofmap._global_dimension;
  _ufc_offset = dofmap._ufc_offset;
  _ownership_range = dofmap._ownership_range;
  _off_process_owner = dofmap._off_process_owner;
  _shared_dofs = dofmap._shared_dofs;
  _neighbours = dofmap._neighbours;
  _is_view = dofmap. _is_view;
  _distributed = dofmap._distributed;
}
//-----------------------------------------------------------------------------
DofMap::~DofMap()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
bool DofMap::needs_mesh_entities(std::size_t d) const
{
  dolfin_assert(_ufc_dofmap);
  return _ufc_dofmap->needs_mesh_entities(d);
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
  if (is_view())
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
void DofMap::tabulate_facet_dofs(unsigned int* dofs, std::size_t local_facet) const
{
  dolfin_assert(_ufc_dofmap);
  _ufc_dofmap->tabulate_facet_dofs(dofs, local_facet);
}
//-----------------------------------------------------------------------------
void DofMap::tabulate_coordinates(boost::multi_array<double, 2>& coordinates,
                                  const ufc::cell& ufc_cell) const
{
  // FIXME: This is a hack because UFC wants a double pointer for coordinates

  // Check dimensions
  if (coordinates.shape()[0] != cell_dimension(ufc_cell.index) ||
      coordinates.shape()[1] != geometric_dimension())
  {
    boost::multi_array<double, 2>::extent_gen extents;
    const std::size_t cell_dim = cell_dimension(ufc_cell.index);
    coordinates.resize(extents[cell_dim][geometric_dimension()]);
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
boost::shared_ptr<GenericDofMap> DofMap::build(const Mesh& new_mesh) const
{
  // Get copy of underlying UFC dof mapo
  boost::shared_ptr<const ufc::dofmap> ufc_dof_map(_ufc_dofmap->create());
  return boost::shared_ptr<GenericDofMap>(new DofMap(ufc_dof_map, new_mesh));
}
//-----------------------------------------------------------------------------
DofMap* DofMap::extract_sub_dofmap(const std::vector<std::size_t>& component,
                                   const Mesh& mesh) const
{
  return new DofMap(*this, component, mesh, _distributed);
}
//-----------------------------------------------------------------------------
DofMap* DofMap::collapse(boost::unordered_map<std::size_t, std::size_t>& collapsed_map,
                         const Mesh& mesh) const
{
  return new DofMap(collapsed_map, *this, mesh, _distributed);
}
//-----------------------------------------------------------------------------
void DofMap::set(GenericVector& x, double value) const
{
  std::vector<std::vector<DolfinIndex> >::const_iterator cell_dofs;
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
    const std::vector<DolfinIndex>& dofs = cell_dofs(cell->index());

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
ufc::dofmap* DofMap::extract_ufc_sub_dofmap(const ufc::dofmap& ufc_dofmap,
                                            std::size_t& offset,
                                            const std::vector<std::size_t>& component,
                                            const ufc::mesh ufc_mesh,
                                            const Mesh& dolfin_mesh)
{
  // Check if there are any sub systems
  if (ufc_dofmap.num_sub_dofmaps() == 0)
  {
    dolfin_error("DofMap.cpp",
                 "extract subsystem of degree of freedom mapping",
                 "There are no subsystems");
  }

  // Check that a sub system has been specified
  if (component.empty())
  {
    dolfin_error("DofMap.cpp",
                 "extract subsystem of degree of freedom mapping",
                 "No system was specified");
  }

  // Check the number of available sub systems
  if (component[0] >= ufc_dofmap.num_sub_dofmaps())
  {
    dolfin_error("DofMap.cpp",
                 "extract subsystem of degree of freedom mapping",
                 "Requested subsystem (%d) out of range [0, %d)",
                 component[0], ufc_dofmap.num_sub_dofmaps());
  }

  // Add to offset if necessary
  for (std::size_t i = 0; i < component[0]; i++)
  {
    // Extract sub dofmap
    boost::scoped_ptr<ufc::dofmap> ufc_tmp_dofmap(ufc_dofmap.create_sub_dofmap(i));
    dolfin_assert(ufc_tmp_dofmap);

    // Initialise
    init_ufc_dofmap(*ufc_tmp_dofmap, ufc_mesh, dolfin_mesh);

    // Get offset
    offset += ufc_tmp_dofmap->global_dimension();
  }

  // Create UFC sub-system
  ufc::dofmap* sub_dofmap = ufc_dofmap.create_sub_dofmap(component[0]);
  dolfin_assert(sub_dofmap);

  // Return sub-system if sub-sub-system should not be extracted, otherwise
  // recursively extract the sub sub system
  if (component.size() == 1)
    return sub_dofmap;
  else
  {
    std::vector<std::size_t> sub_component;
    for (std::size_t i = 1; i < component.size(); ++i)
      sub_component.push_back(component[i]);

    ufc::dofmap* sub_sub_dofmap = extract_ufc_sub_dofmap(*sub_dofmap, offset,
                                                     sub_component, ufc_mesh,
                                                     dolfin_mesh);
    delete sub_dofmap;
    return sub_sub_dofmap;
  }
}
//-----------------------------------------------------------------------------
void DofMap::init_ufc_dofmap(ufc::dofmap& dofmap,
                             const ufc::mesh ufc_mesh,
                             const Mesh& dolfin_mesh)
{
  // Check that we have all mesh entities
  for (std::size_t d = 0; d <= dolfin_mesh.topology().dim(); ++d)
  {
    if (dofmap.needs_mesh_entities(d) && dolfin_mesh.num_entities(d) == 0)
      dolfin_error("DofMap.cpp",
                   "initialize mapping of degrees of freedom",
                   "Missing entities of dimension %d. Try calling mesh.init(%d)", d, d);
  }

  // Initialize UFC dof map
  const bool init_cells = dofmap.init_mesh(ufc_mesh);
  if (init_cells)
  {
    UFCCell ufc_cell(dolfin_mesh);
    for (CellIterator cell(dolfin_mesh); !cell.end(); ++cell)
    {
      ufc_cell.update(*cell);
      dofmap.init_cell(ufc_mesh, ufc_cell);
    }
    dofmap.init_cell_finalize();
  }
}
//-----------------------------------------------------------------------------
boost::unordered_set<std::size_t> DofMap::dofs() const
{
  // Build set of dofs
  boost::unordered_set<std::size_t> dof_list;
  std::vector<std::vector<DolfinIndex> >::const_iterator cell_dofs;
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
