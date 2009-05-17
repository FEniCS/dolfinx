// Copyright (C) 2007-2009 Anders Logg and Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Martin Alnes, 2008
// Modified by Kent-Andre Mardal, 2009
// Modified by Ola Skavhaug, 2009
//
// First added:  2007-03-01
// Last changed: 2009-05-17

#include <dolfin/main/MPI.h>
#include <dolfin/mesh/MeshPartitioning.h>
#include <dolfin/common/NoDeleter.h>
#include <dolfin/common/Timer.h>
#include <dolfin/common/types.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/MeshData.h>
#include <dolfin/main/MPI.h>
#include "UFC.h"
#include "UFCCell.h"
#include "DofMapBuilder.h"
#include "DofMap.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
DofMap::DofMap(boost::shared_ptr<ufc::dof_map> dof_map, boost::shared_ptr<Mesh> mesh)
  : dof_map(0), dof_map_size(0), cell_map(0),
    ufc_dof_map(dof_map), _offset(0),
    dolfin_mesh(mesh), parallel(MPI::num_processes() > 1)
{
  // Generate and number all mesh entities
  for (uint d = 1; d <= mesh->topology().dim(); ++d)
  {
    if (ufc_dof_map->needs_mesh_entities(d))
    {
      mesh->init(d);
      if (parallel) 
        MeshPartitioning::number_entities(*mesh, d);
    }
  }

  // Initialize dof map
  init();
}
//-----------------------------------------------------------------------------
DofMap::DofMap(boost::shared_ptr<ufc::dof_map> dof_map, boost::shared_ptr<const Mesh> mesh)
  : dof_map(0), dof_map_size(0), cell_map(0),
    ufc_dof_map(dof_map), _offset(0),
    dolfin_mesh(mesh), parallel(MPI::num_processes() > 1)
{
  // Check that we have all mesh entities (const so we can't generate them)
  for (uint d = 0; d <= mesh->topology().dim(); ++d)
  {
    if (ufc_dof_map->needs_mesh_entities(d) && mesh->num_entities(d) == 0)
      error("Unable to create function space, missing entities of dimension %d. Try calling mesh.init(%d).", d, d);
  }

  // Initialize dof map
  init();
}
//-----------------------------------------------------------------------------
DofMap::~DofMap()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
DofMap* DofMap::extract_sub_dofmap(const std::vector<uint>& component,
                                   uint& offset) const
{
  // Check that dof map has not be re-ordered
  if (dof_map)
    error("Dof map has been re-ordered. Don't yet know how to extract sub dof maps.");

  // Reset offset
  offset = 0;

  // Recursively extract sub dofmap
  boost::shared_ptr<ufc::dof_map> sub_dof_map(extract_sub_dofmap(*ufc_dof_map, offset, component));
  info(2, "Extracted dof map for sub system: %s", sub_dof_map->signature());
  info(2, "Offset for sub system: %d", offset);

  // Create dofmap
  DofMap* dofmap = new DofMap(sub_dof_map, dolfin_mesh);

  // Set offset
  dofmap->_offset = offset;

  return dofmap;
}
//-----------------------------------------------------------------------------
ufc::dof_map* DofMap::extract_sub_dofmap(const ufc::dof_map& dof_map,
                                         uint& offset,
                                         const std::vector<uint>& component) const
{
  // Check if there are any sub systems
  if (dof_map.num_sub_dof_maps() == 0)
    error("Unable to extract sub system (there are no sub systems).");

  // Check that a sub system has been specified
  if (component.size() == 0)
    error("Unable to extract sub system (no sub system specified).");

  // Check the number of available sub systems
  if (component[0] >= dof_map.num_sub_dof_maps())
    error("Unable to extract sub system %d (only %d sub systems defined).",
                  component[0], dof_map.num_sub_dof_maps());

  // Add to offset if necessary
  for (uint i = 0; i < component[0]; i++)
  {
    boost::shared_ptr<ufc::dof_map> _ufc_dof_map(dof_map.create_sub_dof_map(i)); 
    DofMap dof_map_test(_ufc_dof_map, dolfin_mesh);
    offset += _ufc_dof_map->global_dimension();
  }

  // Create sub system
  ufc::dof_map* sub_dof_map = dof_map.create_sub_dof_map(component[0]);

  // Return sub system if sub sub system should not be extracted
  if (component.size() == 1)
    return sub_dof_map;

  // Otherwise, recursively extract the sub sub system
  std::vector<uint> sub_component;
  for (uint i = 1; i < component.size(); i++)
    sub_component.push_back(component[i]);
  ufc::dof_map* sub_sub_dof_map = extract_sub_dofmap(*sub_dof_map, offset, sub_component);
  delete sub_dof_map;

  return sub_sub_dof_map;
}
//-----------------------------------------------------------------------------
void DofMap::init()
{
  Timer timer("Init dof map");

  // Check that mesh has been ordered
  if (!dolfin_mesh->ordered())
    error("Mesh is not ordered according to the UFC numbering convention, consider calling mesh.order().");

  // Initialize UFC mesh data (must be done after entities are created)
  ufc_mesh.init(*dolfin_mesh);

  // Initialize UFC dof map
  const bool init_cells = ufc_dof_map->init_mesh(ufc_mesh);
  if (init_cells)
  {
    CellIterator cell(*dolfin_mesh);
    UFCCell ufc_cell(*cell);
    for (; !cell.end(); ++cell)
    {
      ufc_cell.update(*cell);
      ufc_dof_map->init_cell(ufc_mesh, ufc_cell);
    }
    ufc_dof_map->init_cell_finalize();
  }
}
//-----------------------------------------------------------------------------
void DofMap::tabulate_dofs(uint* dofs, const ufc::cell& ufc_cell, uint cell_index) const
{
  // Either lookup pretabulated values (if build() has been called)
  // or ask the ufc::dof_map to tabulate the values

  if (dof_map)
  {
    // FIXME: This will only work for problem where local_dimension is the
    //        same for all cells
    const uint n = local_dimension(ufc_cell);
    uint offset = 0;
    offset = n*cell_index;
    for (uint i = 0; i < n; i++)
      dofs[i] = dof_map[offset + i];
    // FIXME: Maybe memcpy() can be used to speed this up? Test this!
    //memcpy(dofs, dof_map[cell_index], sizeof(uint)*local_dimension());
  }
  else
    ufc_dof_map->tabulate_dofs(dofs, ufc_mesh, ufc_cell);
}
//-----------------------------------------------------------------------------
void DofMap::build(UFC& ufc)
{
  error("DofMap::build(UFC& ufc) needs to be fixed since the mesh is const in DofMap.");
  //DofMapBuilder::build(*this, ufc, *dolfin_mesh);
}
//-----------------------------------------------------------------------------
void DofMap::build(const MeshFunction<bool>& meshfunction)
{
  // FIXME: This should be moved to DofMapBuilder

  // Allocate dof map
  const uint n = ufc_dof_map->max_local_dimension();
  uint* dofs = new uint[n];
  dof_map = new int[n*dolfin_mesh->num_cells()];
  cell_map = new int[dolfin_mesh->num_cells()];
  int* restriction_mapping = new int[n*dolfin_mesh->num_cells()];

  // dof_map, restriction_mapping, and cell_map are initialized to -1 to indicate that an error
  // has occured when used outside the subdomain described by meshfunction
  for (uint i=0; i<n*dolfin_mesh->num_cells(); i++)
    dof_map[i]  = -1;
  for (uint i=0; i<ufc_dof_map->global_dimension(); i++)
    restriction_mapping[i] = -1;
  for (uint i=0; i<dolfin_mesh->num_cells(); i++) 
    cell_map[i] = -1;

  CellIterator cell(*dolfin_mesh);
  UFCCell ufc_cell(*cell);
  bool use_cell = false;

  // restriction maping R
  std::map<int,int> R;
  std::map<int,int>::iterator iter;
  uint dof_counter = 0;
  uint cell_counter = 0;

  // Loop over all cells
  for (; !cell.end(); ++cell)
  {
    ufc_cell.update(*cell);
    use_cell = meshfunction.get(cell->index());
    ufc_dof_map->init_cell(ufc_mesh, ufc_cell);
    ufc_dof_map->tabulate_dofs(dofs, ufc_mesh, ufc_cell);

    // If the cell is marked by meshfunction then run through all dofs[k]
    // and check if they have been used before or not.
    // If the dof is new then it is set to dof_counter.
    // The reason why this simple algorithm works is that all already have
    // an unique global numbers. We only need to leave out some of them
    // and have the other numbered in increasing order like 1,2,3,4 (without gaps like 1,3,4).
    if (use_cell) 
    {
      for (uint k=0; k<n; k++) 
      {
        // the dofs[k] is new
        if (R.find(dofs[k]) == R.end())
        {
          R[dofs[k]] = dof_counter;
          dof_counter++;
        }
        cell_map[cell->index()] = cell_counter;
        dof_map[cell->index()*n + k] = R[dofs[k]];
        restriction_mapping [dofs[k]] = R[dofs[k]];
      }
      cell_counter++;
    }
  }
  dof_map_size = dof_counter;

  delete [] dofs;
  delete [] restriction_mapping;
}
//-----------------------------------------------------------------------------
std::map<dolfin::uint, dolfin::uint> DofMap::get_map() const
{
  return map;
}
//-----------------------------------------------------------------------------
dolfin::uint DofMap::offset() const
{
  return _offset;
}
//-----------------------------------------------------------------------------
void DofMap::disp() const
{
  cout << "DofMap" << endl;
  cout << "------" << endl;

  // Begin indentation
  begin("");

  // Display UFC dof_map information
  cout << "ufc::dof_map info" << endl;
  cout << "-----------------" << endl;
  begin("");

  cout << "Signature:               " << ufc_dof_map->signature() << endl;
  cout << "Global dimension:        " << ufc_dof_map->global_dimension() << endl;
  cout << "Maximum local dimension: " << ufc_dof_map->max_local_dimension() << endl;
  cout << "Geometric dimension:     " << ufc_dof_map->geometric_dimension() << endl;
  cout << "Number of subdofmaps:    " << ufc_dof_map->num_sub_dof_maps() << endl;
  cout << "Number of facet dofs:    " << ufc_dof_map->num_facet_dofs() << endl;

  for(uint d = 0; d <= dolfin_mesh->topology().dim(); d++)
    cout << "Number of entity dofs (dim " << d << "): " << ufc_dof_map->num_entity_dofs(d) << endl;
  for(uint d = 0; d <= dolfin_mesh->topology().dim(); d++)
    cout << "Needs mesh entities (dim " << d << "):   " << ufc_dof_map->needs_mesh_entities(d) << endl;
  cout << endl;
  end();

  // Display mesh information
  cout << "Mesh info" << endl;
  cout << "---------" << endl;
  begin("");
  cout << "Geometric dimension:   " << dolfin_mesh->geometry().dim() << endl;
  cout << "Topological dimension: " << dolfin_mesh->topology().dim() << endl;
  cout << "Number of vertices:    " << dolfin_mesh->num_vertices() << endl;
  cout << "Number of edges:       " << dolfin_mesh->num_edges() << endl;
  cout << "Number of faces:       " << dolfin_mesh->num_faces() << endl;
  cout << "Number of facets:      " << dolfin_mesh->num_facets() << endl;
  cout << "Number of cells:       " << dolfin_mesh->num_cells() << endl;
  cout << endl;
  end();

  cout << "Local cell dofs associated with cell entities (tabulate_entity_dofs output):" << endl;
  cout << "----------------------------------------------------------------------------" << endl;
  begin("");
  {
    uint tdim = dolfin_mesh->topology().dim();
    for(uint d=0; d<=tdim; d++)
    {
      uint num_dofs = ufc_dof_map->num_entity_dofs(d);
      if(num_dofs)
      {
        uint num_entities = dolfin_mesh->type().num_entities(d);
        uint* dofs = new uint[num_dofs];
        for(uint i=0; i<num_entities; i++)
        {
          cout << "Entity (" << d << ", " << i << "):  ";
          ufc_dof_map->tabulate_entity_dofs(dofs, d, i);
          for(uint j=0; j<num_dofs; j++)
          {
            cout << dofs[j];
            if(j < num_dofs-1) cout << ", ";
          }
          cout << endl;
        }
        delete [] dofs;
      }
    }
    cout << endl;
  }
  end();

  cout << "Local cell dofs associated with facets (tabulate_facet_dofs output):" << endl;
  cout << "--------------------------------------------------------------------" << endl;
  begin("");
  {
    uint tdim = dolfin_mesh->topology().dim();
    uint num_dofs = ufc_dof_map->num_facet_dofs();
    uint num_facets = dolfin_mesh->type().num_entities(tdim-1);
    uint* dofs = new uint[num_dofs];
    for(uint i=0; i<num_facets; i++)
    {
      cout << "Facet " << i << ":  ";
      ufc_dof_map->tabulate_facet_dofs(dofs, i);
      for(uint j=0; j<num_dofs; j++)
      {
        cout << dofs[j];
        if(j < num_dofs-1) cout << ", ";
      }
      cout << endl;
    }
    delete [] dofs;
    cout << endl;
  }
  end();

  cout << "tabulate_dofs output" << endl;
  cout << "--------------------" << endl;
  begin("");
  {
    uint tdim = dolfin_mesh->topology().dim();
    uint max_num_dofs = ufc_dof_map->max_local_dimension();
    uint* dofs = new uint[max_num_dofs];
    CellIterator cell(*dolfin_mesh);
    UFCCell ufc_cell(*cell);
    for (; !cell.end(); ++cell)
    {
      ufc_cell.update(*cell);
      uint num_dofs = ufc_dof_map->local_dimension(ufc_cell);

      ufc_dof_map->tabulate_dofs(dofs, ufc_mesh, ufc_cell);

      cout << "Cell " << ufc_cell.entity_indices[tdim][0] << ":  ";
      for(uint j = 0; j < num_dofs; j++)
      {
        cout << dofs[j];
        if(j < num_dofs-1) cout << ", ";
      }
      cout << endl;
    }
    delete [] dofs;
    cout << endl;
  }
  end();

  cout << "tabulate_coordinates output" << endl;
  cout << "---------------------------" << endl;
  begin("");
  {
    uint tdim = dolfin_mesh->topology().dim();
    uint gdim = ufc_dof_map->geometric_dimension();
    uint max_num_dofs = ufc_dof_map->max_local_dimension();
    double** coordinates = new double*[max_num_dofs];
    for(uint k=0; k<max_num_dofs; k++)
    {
      coordinates[k] = new double[gdim];
    }
    CellIterator cell(*dolfin_mesh);
    UFCCell ufc_cell(*cell);
    for (; !cell.end(); ++cell)
    {
      ufc_cell.update(*cell);
      uint num_dofs = ufc_dof_map->local_dimension(ufc_cell);

      ufc_dof_map->tabulate_coordinates(coordinates, ufc_cell);

      cout << "Cell " << ufc_cell.entity_indices[tdim][0] << ":  ";
      for(uint j=0; j<num_dofs; j++)
      {
        cout << "(";
        for(uint k=0; k<gdim; k++)
        {
          cout << coordinates[j][k];
          if(k < gdim-1) cout << ", ";
        }
        cout << ")";
        if(j < num_dofs-1) cout << ",  ";
      }
      cout << endl;
    }
    for(uint k=0; k<gdim; k++)
      delete [] coordinates[k];
    delete [] coordinates;
    cout << endl;
  }

  end();

  // TODO: Display information on renumbering?
  // TODO: Display information on parallel stuff?

  // End indentation
  end();
}
//-----------------------------------------------------------------------------

