// Copyright (C) 2007-2008 Anders Logg and Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Martin Alnes, 2008
//
// First added:  2007-03-01
// Last changed: 2008-10-14

#include <dolfin/common/NoDeleter.h>
#include <dolfin/common/Timer.h>
#include <dolfin/common/types.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/MeshData.h>
#include <dolfin/common/Array.h>
#include <dolfin/elements/ElementLibrary.h>
#include <dolfin/main/MPI.h>
#include "UFCCell.h"
#include "SubSystem.h"
#include "UFC.h"
#include "DofMapBuilder.h"
#include "DofMap.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
DofMap::DofMap(ufc::dof_map& dof_map, Mesh& mesh)
  : dof_map(0), 
    ufc_dof_map(&dof_map, NoDeleter<ufc::dof_map>()), dolfin_mesh(mesh), 
    num_cells(mesh.numCells()), partitions(0), _offset(0)
{
  init();
}
//-----------------------------------------------------------------------------
DofMap::DofMap(std::tr1::shared_ptr<ufc::dof_map> dof_map, Mesh& mesh)
  : dof_map(0), ufc_dof_map(dof_map), dolfin_mesh(mesh), num_cells(mesh.numCells()), 
    partitions(0), _offset(0)
{
  init();
}
//-----------------------------------------------------------------------------
DofMap::DofMap(ufc::dof_map& dof_map, Mesh& mesh, MeshFunction<uint>& partitions)
  : dof_map(0), ufc_dof_map(&dof_map, NoDeleter<ufc::dof_map>()), 
    dolfin_mesh(mesh), num_cells(mesh.numCells()), 
    partitions(&partitions), _offset(0)
{
  init();
}
//-----------------------------------------------------------------------------
DofMap::DofMap(std::tr1::shared_ptr<ufc::dof_map> dof_map, Mesh& mesh, MeshFunction<uint>& partitions)
  : dof_map(0), ufc_dof_map(dof_map), 
    dolfin_mesh(mesh), num_cells(mesh.numCells()), 
    partitions(&partitions), _offset(0)
{
  init();
}
//-----------------------------------------------------------------------------
DofMap::DofMap(const std::string signature, Mesh& mesh) 
  : dof_map(0), dolfin_mesh(mesh), num_cells(mesh.numCells()),
    partitions(0), _offset(0)
{
  // Create ufc dof map from signature
  std::tr1::shared_ptr<ufc::dof_map> _ufc_dof_map(ElementLibrary::create_dof_map(signature));
  ufc_dof_map.swap(_ufc_dof_map);

  if (!ufc_dof_map)
    error("Unable to find dof map in library: \"%s\".",signature.c_str());

  init();
}
//-----------------------------------------------------------------------------
DofMap::DofMap(const std::string signature, Mesh& mesh,
               MeshFunction<uint>& partitions)
  : dof_map(0), dolfin_mesh(mesh),
    num_cells(mesh.numCells()), partitions(&partitions), _offset(0)
{
  // Create ufc dof map from signature
  std::tr1::shared_ptr<ufc::dof_map> _ufc_dof_map(ElementLibrary::create_dof_map(signature));
  ufc_dof_map.swap(_ufc_dof_map);

  if (!ufc_dof_map)
    error("Unable to find dof map in library: \"%s\".",signature.c_str());

  init();
}
//-----------------------------------------------------------------------------
DofMap::~DofMap()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
DofMap* DofMap::extract_sub_dofmap(const Array<uint>& sub_system, uint& offset) const
{
  // Check that dof map has not be re-ordered
  if (dof_map)
    error("Dof map has been re-ordered. Don't yet know how to extract sub dof maps.");

  // Reset offset
  offset = 0;

  // Recursively extract sub dofmap
  std::tr1::shared_ptr<ufc::dof_map> sub_dof_map(extract_sub_dofmap(*ufc_dof_map, offset, sub_system));
  message(2, "Extracted dof map for sub system: %s", sub_dof_map->signature());
  message(2, "Offset for sub system: %d", offset);

  // Create dofmap
  DofMap* dofmap = 0;
  if (partitions)
    dofmap = new DofMap(sub_dof_map, dolfin_mesh, *partitions);
  else
    dofmap = new DofMap(sub_dof_map, dolfin_mesh);

  // Set offset
  dofmap->_offset = offset;

  return dofmap;
}
//-----------------------------------------------------------------------------
ufc::dof_map* DofMap::extract_sub_dofmap(const ufc::dof_map& dof_map, uint& offset,
                                         const Array<uint>& sub_system) const
{
  // Check if there are any sub systems
  if (dof_map.num_sub_dof_maps() == 0)
    error("Unable to extract sub system (there are no sub systems).");

  // Check that a sub system has been specified
  if (sub_system.size() == 0)
    error("Unable to extract sub system (no sub system specified).");
  
  // Check the number of available sub systems
  if (sub_system[0] >= dof_map.num_sub_dof_maps())
    error("Unable to extract sub system %d (only %d sub systems defined).",
                  sub_system[0], dof_map.num_sub_dof_maps());

  // Add to offset if necessary
  for (uint i = 0; i < sub_system[0]; i++)
  {
    ufc::dof_map* ufc_dof_map = dof_map.create_sub_dof_map(i);
    if (partitions)
      DofMap dof_map_test(*ufc_dof_map, dolfin_mesh, *partitions);
    else
      DofMap dof_map_test(*ufc_dof_map, dolfin_mesh);
    offset += ufc_dof_map->global_dimension();
    delete ufc_dof_map;
  }
  
  // Create sub system
  ufc::dof_map* sub_dof_map = dof_map.create_sub_dof_map(sub_system[0]);
  
  // Return sub system if sub sub system should not be extracted
  if (sub_system.size() == 1)
    return sub_dof_map;

  // Otherwise, recursively extract the sub sub system
  Array<uint> sub_sub_system;
  for (uint i = 1; i < sub_system.size(); i++)
    sub_sub_system.push_back(sub_system[i]);
  ufc::dof_map* sub_sub_dof_map = extract_sub_dofmap(*sub_dof_map, offset, sub_sub_system);
  delete sub_dof_map;

  return sub_sub_dof_map;
}
//-----------------------------------------------------------------------------
void DofMap::init()
{
  Timer timer("Init dof map");

  //dolfin_debug("Initializing dof map...");

  // Order vertices, so entities will be created correctly according to convention
  dolfin_mesh.order();

  // Initialize mesh entities used by dof map
  for (uint d = 0; d <= dolfin_mesh.topology().dim(); d++)
    if (ufc_dof_map->needs_mesh_entities(d))
      dolfin_mesh.init(d);
  
  // Initialize UFC mesh data (must be done after entities are created)
  ufc_mesh.init(dolfin_mesh);

  // Initialize UFC dof map
  const bool init_cells = ufc_dof_map->init_mesh(ufc_mesh);
  if (init_cells)
  {
    CellIterator cell(dolfin_mesh);
    UFCCell ufc_cell(*cell);
    for (; !cell.end(); ++cell)
    {
      ufc_cell.update(*cell);
      ufc_dof_map->init_cell(ufc_mesh, ufc_cell);
    }
    ufc_dof_map->init_cell_finalize();
  }

  //dolfin_debug("Dof map initialized");
}
//-----------------------------------------------------------------------------
void DofMap::tabulate_dofs(uint* dofs, ufc::cell& ufc_cell, uint cell_index) const
{
  // Either lookup pretabulated values (if build() has been called)
  // or ask the ufc::dof_map to tabulate the values

  if (dof_map)
  {
    const uint n = local_dimension();
    const uint offset = n*cell_index;
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
  DofMapBuilder::build(*this, ufc, dolfin_mesh);
}
//-----------------------------------------------------------------------------
std::map<dolfin::uint, dolfin::uint> DofMap::getMap() const
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

  cout << "Signature:            " << ufc_dof_map->signature() << endl;
  cout << "Global dimension:     " << ufc_dof_map->global_dimension() << endl;
  cout << "Local dimension:      " << ufc_dof_map->local_dimension() << endl;
  cout << "Geometric dimension:  " << ufc_dof_map->geometric_dimension() << endl;
  cout << "Number of subdofmaps: " << ufc_dof_map->num_sub_dof_maps() << endl;
  cout << "Number of facet dofs: " << ufc_dof_map->num_facet_dofs() << endl;

  for(uint d=0; d<=dolfin_mesh.topology().dim(); d++)
  {
    cout << "Number of entity dofs (dim " << d << "): " << ufc_dof_map->num_entity_dofs(d) << endl;
  }
  for(uint d=0; d<=dolfin_mesh.topology().dim(); d++)
  {
    cout << "Needs mesh entities (dim " << d << "):   " << ufc_dof_map->needs_mesh_entities(d) << endl;
  }
  cout << endl;
  end();

  // Display mesh information
  cout << "Mesh info" << endl;
  cout << "---------" << endl;
  begin("");
  cout << "Geometric dimension:   " << dolfin_mesh.geometry().dim() << endl;
  cout << "Topological dimension: " << dolfin_mesh.topology().dim() << endl;
  cout << "Number of vertices:    " << dolfin_mesh.numVertices() << endl;
  cout << "Number of edges:       " << dolfin_mesh.numEdges() << endl;
  cout << "Number of faces:       " << dolfin_mesh.numFaces() << endl;
  cout << "Number of facets:      " << dolfin_mesh.numFacets() << endl;
  cout << "Number of cells:       " << dolfin_mesh.numCells() << endl;
  cout << endl;
  end();

  cout << "Local cell dofs associated with cell entities (tabulate_entity_dofs output):" << endl;
  cout << "----------------------------------------------------------------------------" << endl;
  begin("");
  {
    uint tdim = dolfin_mesh.topology().dim();
    for(uint d=0; d<=tdim; d++)
    {
      uint num_dofs = ufc_dof_map->num_entity_dofs(d);
      if(num_dofs)
      {
        uint num_entities = dolfin_mesh.type().numEntities(d);
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
    uint tdim = dolfin_mesh.topology().dim();
    uint num_dofs = ufc_dof_map->num_facet_dofs();
    uint num_facets = dolfin_mesh.type().numEntities(tdim-1);
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
    uint tdim = dolfin_mesh.topology().dim();
    uint num_dofs = ufc_dof_map->local_dimension();
    uint* dofs = new uint[num_dofs];
    CellIterator cell(dolfin_mesh);
    UFCCell ufc_cell(*cell);
    for (; !cell.end(); ++cell)
    {
      ufc_cell.update(*cell);
 
      ufc_dof_map->tabulate_dofs(dofs, ufc_mesh, ufc_cell);
 
      cout << "Cell " << ufc_cell.entity_indices[tdim][0] << ":  ";
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

  cout << "tabulate_coordinates output" << endl;
  cout << "---------------------------" << endl;
  begin("");
  {
    uint tdim = dolfin_mesh.topology().dim();
    uint gdim = ufc_dof_map->geometric_dimension();
    uint num_dofs = ufc_dof_map->local_dimension();
    double** coordinates = new double*[num_dofs];
    for(uint k=0; k<num_dofs; k++)
    {
      coordinates[k] = new double[gdim];
    }
    CellIterator cell(dolfin_mesh);
    UFCCell ufc_cell(*cell);
    for (; !cell.end(); ++cell)
    {
      ufc_cell.update(*cell);

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

