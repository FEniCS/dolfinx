// Copyright (C) 2007-2010 Anders Logg and Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Martin Alnes, 2008
// Modified by Kent-Andre Mardal, 2009
// Modified by Ola Skavhaug, 2009
// Modified by Niclas Jansson, 2009
//
// First added:  2007-03-01
// Last changed: 2010-04-29

#include <boost/scoped_array.hpp>

#include <dolfin/common/Set.h>
#include <dolfin/main/MPI.h>
#include <dolfin/mesh/MeshPartitioning.h>
#include <dolfin/common/NoDeleter.h>
#include <dolfin/common/types.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/MeshData.h>
#include <dolfin/main/MPI.h>
#include "UFC.h"
#include "UFCCell.h"
#include "DofMapBuilder.h"
#include "DofMap.h"

#include <dolfin/mesh/BoundaryMesh.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
DofMap::DofMap(boost::shared_ptr<ufc::dof_map> ufc_dofmap,
               Mesh& dolfin_mesh)
  : _ufc_dofmap(ufc_dofmap), _ufc_offset(0),
    _parallel(MPI::num_processes() > 1)
{
  // Generate and number all mesh entities
  for (uint d = 1; d <= dolfin_mesh.topology().dim(); ++d)
  {
    if (_ufc_dofmap->needs_mesh_entities(d) ||
    	(_parallel && d == (dolfin_mesh.topology().dim() - 1)))
    {
      dolfin_mesh.init(d);
      if (_parallel)
        MeshPartitioning::number_entities(dolfin_mesh, d);
    }
  }

  // Initialize
  init(dolfin_mesh, true);
}
//-----------------------------------------------------------------------------
DofMap::DofMap(boost::shared_ptr<ufc::dof_map> ufc_dofmap,
               const Mesh& dolfin_mesh)
  : _ufc_dofmap(ufc_dofmap), _ufc_offset(0),
    _parallel(MPI::num_processes() > 1)
{
  // Initialize
  init(dolfin_mesh, true);
}
//-----------------------------------------------------------------------------
DofMap::DofMap(boost::shared_ptr<std::vector<dolfin::uint> > map,
               boost::shared_ptr<ufc::dof_map> ufc_dofmap,
               const Mesh& dolfin_mesh)
  : _map(map), _ufc_dofmap(ufc_dofmap), _ufc_offset(0),
    _parallel(MPI::num_processes() > 1)

{
  // Initialize
  init(dolfin_mesh, false);
}
//-----------------------------------------------------------------------------
DofMap::~DofMap()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void DofMap::tabulate_dofs(uint* dofs, const ufc::cell& ufc_cell,
                           uint cell_index) const
{
  // Lookup pretabulated values or ask the ufc::dof_map to tabulate the values
  if (_map.get())
  {
    // FIXME: Add assertion to test that this process has the dof.
    // FIXME: This will only work for problems where local_dimension is the
    // FIXME: same for all cells since the offset will not be computed correctly.
    const uint n = local_dimension(ufc_cell);
    const uint offset = n*cell_index;
    for (uint i = 0; i < n; i++)
      dofs[i] = (*_map)[offset + i];
    // FIXME: Maybe std::copy be used to speed this up?
    //std::copy(&(*map)[offset], &(*map)[offset+n], dofs);
  }
  else
  {
    // Tabulate UFC dof map
    _ufc_dofmap->tabulate_dofs(dofs, _ufc_mesh, ufc_cell);

    // Add offset if necessary
    if (_ufc_offset > 0)
    {
      const uint local_dim = local_dimension(ufc_cell);
      for (uint i = 0; i < local_dim; i++)
        dofs[i] += _ufc_offset;
    }
  }
}
//-----------------------------------------------------------------------------
void DofMap::tabulate_dofs(uint* dofs, const Cell& cell) const
{
  UFCCell ufc_cell(cell);
  tabulate_dofs(dofs, ufc_cell, cell.index());
}
//-----------------------------------------------------------------------------
void DofMap::tabulate_facet_dofs(uint* dofs, uint local_facet) const
{
  _ufc_dofmap->tabulate_facet_dofs(dofs, local_facet);
}
//-----------------------------------------------------------------------------
void DofMap::tabulate_coordinates(double** coordinates, const Cell& cell) const
{
  UFCCell ufc_cell(cell);
  tabulate_coordinates(coordinates, ufc_cell);
}
//-----------------------------------------------------------------------------
DofMap* DofMap::extract_sub_dofmap(const std::vector<uint>& component,
                                   const Mesh& dolfin_mesh) const
{
  // Reset offset
  uint ufc_offset = 0;

  // Recursively extract UFC sub dofmap
  boost::shared_ptr<ufc::dof_map>
    ufc_sub_dof_map(extract_sub_dofmap(*_ufc_dofmap, ufc_offset, component, _ufc_mesh, dolfin_mesh));
  info(DBG, "Extracted dof map for sub system: %s", ufc_sub_dof_map->signature());
  info(DBG, "Offset for sub system: %d", ufc_offset);

  // Create dofmap
  DofMap* sub_dofmap = 0;
  if (_map.get())
  {
    if (_ufc_to_map.size() == 0)
      error("Cannnot yet extract sub dofmaps of a sub DofMap after renumbering yet.");

    const uint max_local_dim = ufc_sub_dof_map->max_local_dimension();
    const uint num_cells = dolfin_mesh.num_cells();

     // Create vector for new map
    boost::shared_ptr<std::vector<uint> > sub_map(new std::vector<uint>);
    sub_map->resize(max_local_dim*num_cells);

    // Create new dof map (this will initialise the UFC dof map)
    sub_dofmap = new DofMap(sub_map, ufc_sub_dof_map, dolfin_mesh);

    // Build sub-map vector
    UFCCell ufc_cell(dolfin_mesh);
    uint* ufc_dofs = new uint[ufc_sub_dof_map->max_local_dimension()];
    for (CellIterator cell(dolfin_mesh); !cell.end(); ++cell)
    {
      // Update to current cell
      ufc_cell.update(*cell);

     // Tabulate sub-dofs on cell (UFC map)
     ufc_sub_dof_map->tabulate_dofs(ufc_dofs, _ufc_mesh, ufc_cell);

     const uint cell_index = cell->index();
     const uint sub_local_dim = sub_dofmap->local_dimension(ufc_cell);
     const uint cell_offset = sub_local_dim*cell_index;
     for (uint i = 0; i < sub_local_dim; i++)
       (*sub_dofmap->_map)[cell_offset + i] = _ufc_to_map.find(ufc_dofs[i] + ufc_offset)->second;
    }
    delete [] ufc_dofs;
  }
  else
    sub_dofmap = new DofMap(ufc_sub_dof_map, dolfin_mesh);

  // Set offset
  sub_dofmap->_ufc_offset = ufc_offset;

  return sub_dofmap;
}
//-----------------------------------------------------------------------------
DofMap* DofMap::collapse(std::map<uint, uint>& collapsed_map,
                         const Mesh& dolfin_mesh) const
{
  // Create a new DofMap
  DofMap* collapsed_dof_map = 0;
  if (_map.get())
    error("Cannot yet collapse renumbered dof maps.");
  else
    collapsed_dof_map = new DofMap(_ufc_dofmap, dolfin_mesh);

  assert(collapsed_dof_map->global_dimension() == this->global_dimension());

  // Clear map
  collapsed_map.clear();

  // Build map from collapsed to original dofs
  UFCCell ufc_cell(dolfin_mesh);
  uint* dofs = new uint[this->max_local_dimension()];
  uint* collapsed_dofs = new uint[collapsed_dof_map->max_local_dimension()];
  for (CellIterator cell(dolfin_mesh); !cell.end(); ++cell)
  {
    // Update to current cell
    ufc_cell.update(*cell);

   // Tabulate dofs
   this->tabulate_dofs(dofs, ufc_cell, cell->index());
   collapsed_dof_map->tabulate_dofs(collapsed_dofs, ufc_cell, cell->index());

    // Add to map
    for (uint i = 0; i < collapsed_dof_map->local_dimension(ufc_cell); ++i)
      collapsed_map[collapsed_dofs[i]] = dofs[i];
  }
  delete [] dofs;
  delete [] collapsed_dofs;

  return collapsed_dof_map;
}
//-----------------------------------------------------------------------------
ufc::dof_map* DofMap::extract_sub_dofmap(const ufc::dof_map& ufc_dofmap,
                                         uint& offset,
                                         const std::vector<uint>& component,
                                         const ufc::mesh ufc_mesh,
                                         const Mesh& dolfin_mesh)
{
  // Check if there are any sub systems
  if (ufc_dofmap.num_sub_dof_maps() == 0)
    error("Unable to extract sub system (there are no sub systems).");

  // Check that a sub system has been specified
  if (component.size() == 0)
    error("Unable to extract sub system (no sub system specified).");

  // Check the number of available sub systems
  if (component[0] >= ufc_dofmap.num_sub_dof_maps())
    error("Unable to extract sub system %d (only %d sub systems defined).",
                  component[0], ufc_dofmap.num_sub_dof_maps());

  // Add to offset if necessary
  for (uint i = 0; i < component[0]; i++)
  {
    // Extract sub dofmap
    boost::shared_ptr<ufc::dof_map> _ufc_dofmap(ufc_dofmap.create_sub_dof_map(i));

    // Initialise
    init_ufc_dofmap(*_ufc_dofmap, ufc_mesh, dolfin_mesh);

    // Get offset
    offset += _ufc_dofmap->global_dimension();
  }

  // Create UFC sub system
  ufc::dof_map* sub_dof_map = ufc_dofmap.create_sub_dof_map(component[0]);

  // Return sub system if sub sub system should not be extracted
  if (component.size() == 1)
    return sub_dof_map;

  // Otherwise, recursively extract the sub sub system
  std::vector<uint> sub_component;
  for (uint i = 1; i < component.size(); i++)
    sub_component.push_back(component[i]);
  ufc::dof_map* sub_sub_dof_map = extract_sub_dofmap(*sub_dof_map, offset,
                                                     sub_component, ufc_mesh,
                                                     dolfin_mesh);
  delete sub_dof_map;

  return sub_sub_dof_map;
}
//-----------------------------------------------------------------------------
void DofMap::init(const Mesh& dolfin_mesh, bool build_map)
{
  // Initialize the UFC mesh
  init_ufc_mesh(_ufc_mesh, dolfin_mesh);

  // Initialize the UFC dofmap
  init_ufc_dofmap(*_ufc_dofmap, _ufc_mesh, dolfin_mesh);

  // Build (renumber) dofmap when running in parallel
  if (build_map && _parallel)
    DofMapBuilder::parallel_build(*this, dolfin_mesh);
}
//-----------------------------------------------------------------------------
void DofMap::init_ufc_mesh(UFCMesh& ufc_mesh, const Mesh& dolfin_mesh)
{
  // Check that mesh has been ordered
  if (!dolfin_mesh.ordered())
     error("Mesh is not ordered according to the UFC numbering convention, consider calling mesh.order().");

  // Initialize UFC mesh data (must be done after entities are created)
  ufc_mesh.init(dolfin_mesh);
}
//-----------------------------------------------------------------------------
void DofMap::init_ufc_dofmap(ufc::dof_map& dofmap,
                             const ufc::mesh ufc_mesh,
                             const Mesh& dolfin_mesh)
{
  // Check that we have all mesh entities
  for (uint d = 0; d <= dolfin_mesh.topology().dim(); ++d)
  {
    if (dofmap.needs_mesh_entities(d) && dolfin_mesh.num_entities(d) == 0)
      error("Unable to create function space, missing entities of dimension %d. Try calling mesh.init(%d).", d, d);
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
dolfin::Set<dolfin::uint> DofMap::dofs(const Mesh& mesh, bool sort) const
{
  dolfin::Set<uint> dof_list;

  UFCCell ufc_cell(mesh);
  std::vector<uint> dofs(max_local_dimension());
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    // Update to current cell
    ufc_cell.update(*cell);

    // Tabulate dofs and insert int Set
    tabulate_dofs(&dofs[0], ufc_cell, cell->index());

    for (uint i = 0; i < local_dimension(ufc_cell); ++i)
      dof_list.insert(dofs[i]);
  }

  if(sort)
    dof_list.sort();

  return dof_list;
}
//-----------------------------------------------------------------------------
std::string DofMap::str(bool verbose) const
{
  // TODO: Display information on renumbering?
  // TODO: Display information on parallel stuff?

  if (_map.get())
  {
    warning("DofMap::str has not been updated for re-numbered dof maps.");
    return std::string();
  }

  std::stringstream s;

  if (verbose)
  {
    s << str(false) << std::endl << std::endl;

    // Display UFC dof_map information
    s << "  ufc::dof_map" << std::endl;
    s << "  ------------" << std::endl;

    s << "    Signature:               " << _ufc_dofmap->signature() << std::endl;
    s << "    Global dimension:        " << _ufc_dofmap->global_dimension() << std::endl;
    s << "    Maximum local dimension: " << _ufc_dofmap->max_local_dimension() << std::endl;
    s << "    Geometric dimension:     " << _ufc_dofmap->geometric_dimension() << std::endl;
    s << "    Number of sub dofmaps:   " << _ufc_dofmap->num_sub_dof_maps() << std::endl;
    s << "    Number of facet dofs:    " << _ufc_dofmap->num_facet_dofs() << std::endl;

    // FIXME: Mostly commented out after removal of Mesh member variable

    /*
    for (uint d = 0; d <= _dolfin_mesh->topology().dim(); d++)
      s << "    Number of entity dofs (dim " << d << "): "
        << _ufc_dofmap->num_entity_dofs(d) << std::endl;
    for (uint d = 0; d <= _dolfin_mesh->topology().dim(); d++)
      s << "    Needs mesh entities (dim " << d << "):   "
        << _ufc_dofmap->needs_mesh_entities(d) << std::endl;
    s << std::endl;

    // Display mesh information
    s << "  Mesh info" << std::endl;
    s << "  ---------" << std::endl;
    s << "    Geometric dimension:   " << _dolfin_mesh->geometry().dim() << std::endl;
    s << "    Topological dimension: " << _dolfin_mesh->topology().dim() << std::endl;
    s << "    Number of vertices:    " << _dolfin_mesh->num_vertices() << std::endl;
    s << "    Number of edges:       " << _dolfin_mesh->num_edges() << std::endl;
    s << "    Number of faces:       " << _dolfin_mesh->num_faces() << std::endl;
    s << "    Number of facets:      " << _dolfin_mesh->num_facets() << std::endl;
    s << "    Number of cells:       " << _dolfin_mesh->num_cells() << std::endl;
    s << std::endl;

    s << "  tabulate_entity_dofs"
      << std::endl;
    s << "  --------------------"
      << std::endl;
    {
      uint tdim = _dolfin_mesh->topology().dim();
      for (uint d = 0; d <= tdim; d++)
      {
        uint num_dofs = _ufc_dofmap->num_entity_dofs(d);
        if (num_dofs)
        {
          uint num_entities = _dolfin_mesh->type().num_entities(d);
          uint* dofs = new uint[num_dofs];
          for (uint i = 0; i < num_entities; i++)
          {
            s << "    Entity (" << d << ", " << i << "):  ";
            _ufc_dofmap->tabulate_entity_dofs(dofs, d, i);
            for (uint j = 0; j < num_dofs; j++)
            {
              s << dofs[j];
              if (j < num_dofs-1)
                s << ", ";
            }
            s << std::endl;
          }
          delete [] dofs;
        }
      }
      s << std::endl;
    }

    s << "  tabulate_facet_dofs output" << std::endl;
    s << "  --------------------------" << std::endl;
    {
      uint tdim = _dolfin_mesh->topology().dim();
      uint num_dofs = _ufc_dofmap->num_facet_dofs();
      uint num_facets = _dolfin_mesh->type().num_entities(tdim-1);
      uint* dofs = new uint[num_dofs];
      for (uint i = 0; i < num_facets; i++)
      {
        s << "Facet " << i << ":  ";
        _ufc_dofmap->tabulate_facet_dofs(dofs, i);
        for (uint j = 0; j<num_dofs; j++)
        {
          s << dofs[j];
          if (j < num_dofs-1)
            s << ", ";
        }
        s << std::endl;
      }
      delete [] dofs;
      s << std::endl;
    }

    s << "  tabulate_dofs" << std::endl;
    s << "  -------------" << std::endl;
    {
      uint tdim = _dolfin_mesh->topology().dim();
      uint max_num_dofs = _ufc_dofmap->max_local_dimension();
      uint* dofs = new uint[max_num_dofs];
      CellIterator cell(*_dolfin_mesh);
      UFCCell ufc_cell(*cell);
      for (; !cell.end(); ++cell)
      {
        ufc_cell.update(*cell);
        uint num_dofs = _ufc_dofmap->local_dimension(ufc_cell);

        _ufc_dofmap->tabulate_dofs(dofs, _ufc_mesh, ufc_cell);

        s << "    Cell " << ufc_cell.entity_indices[tdim][0] << ":  ";
        for (uint j = 0; j < num_dofs; j++)
        {
          s << dofs[j];
          if (j < num_dofs-1)
            s << ", ";
        }
        s << std::endl;
      }
      delete [] dofs;
      s << std::endl;
    }

    s << "  tabulate_coordinates" << std::endl;
    s << "  --------------------" << std::endl;
    {
      const uint tdim = _dolfin_mesh->topology().dim();
      const uint gdim = _ufc_dofmap->geometric_dimension();
      const uint max_num_dofs = _ufc_dofmap->max_local_dimension();
      double** coordinates = new double*[max_num_dofs];
      for (uint k = 0; k < max_num_dofs; k++)
        coordinates[k] = new double[gdim];
      CellIterator cell(*_dolfin_mesh);
      UFCCell ufc_cell(*cell);
      for (; !cell.end(); ++cell)
      {
        ufc_cell.update(*cell);
        uint num_dofs = _ufc_dofmap->local_dimension(ufc_cell);

        _ufc_dofmap->tabulate_coordinates(coordinates, ufc_cell);

        s << "    Cell " << ufc_cell.entity_indices[tdim][0] << ":  ";
        for (uint j = 0; j < num_dofs; j++)
        {
          s << "(";
          for (uint k = 0; k < gdim; k++)
          {
            s << coordinates[j][k];
            if (k < gdim-1)
              s << ", ";
          }
          s << ")";
          if (j < num_dofs-1)
            s << ",  ";
        }
        s << std::endl;
      }
      for (uint k = 0; k < gdim; k++)
        delete [] coordinates[k];
      delete [] coordinates;
      s << std::endl;
    }

    */

  }
  else
  {
    s << "<DofMap>";
  }

  return s.str();
}
//-----------------------------------------------------------------------------
