// Copyright (C) 2012 Chris N Richardson
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
// Modified by Garth N. Wells, 2012
//
// First added:  2012-06-01
// Last changed: 2012-09-24

#ifdef HAS_HDF5

#include <cstdio>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>

#include <dolfin/common/types.h>
#include <dolfin/common/constants.h>
#include <dolfin/common/MPI.h>
#include <dolfin/common/NoDeleter.h>
#include <dolfin/function/Function.h>
#include <dolfin/la/GenericVector.h>
#include <dolfin/log/log.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/LocalMeshData.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshPartitioning.h>
#include <dolfin/mesh/MeshEntityIterator.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/mesh/Vertex.h>

#include "HDF5File.h"
#include "HDF5Interface.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
HDF5File::HDF5File(const std::string filename) : GenericFile(filename, "H5")
{
  // Do nothing

  // FIXME: Create file here in constructor?
  // Not all instatiations of HDF5File create a new file.
  // Could possibly open file descriptor here.
}
//-----------------------------------------------------------------------------
HDF5File::~HDF5File()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void HDF5File::create()
{
  // Create a new file - used by XDMFFile
  HDF5Interface::create(filename);
}
//-----------------------------------------------------------------------------
std::string HDF5File::search_list(std::vector<std::string> &list_of_strings, 
                                  const std::string &search_term) const
{
  // Search through a list of names for a name beginning with search_term
  for(std::vector<std::string>::iterator list_iterator = list_of_strings.begin();
      list_iterator != list_of_strings.end();
      ++list_iterator)
  {
    if(list_iterator->find(search_term) != std::string::npos)
      return *list_iterator;
  }
  return std::string("");
}
//-----------------------------------------------------------------------------
// Mesh input not yet supported
void HDF5File::operator>> (Mesh& input_mesh)
{

  dolfin_error("HDF5File.cpp",
               "read mesh from file",
               "Mesh input is not supported yet");

}

//-----------------------------------------------------------------------------
void HDF5File::operator<< (const Mesh& mesh)
{
  // Mesh output with true global indices - not currently useable for visualisation
  write_mesh(mesh, true);
}
//-----------------------------------------------------------------------------
void HDF5File::write_mesh(const Mesh& mesh, bool true_topology_indices)
{
  // Clear file when writing to file for the first time
  if(counter == 0)
    HDF5Interface::create(filename);
  counter++;

  // Get local mesh data
  const uint cell_dim = mesh.topology().dim();
  const uint num_local_cells = mesh.num_cells();
  const uint num_local_vertices = mesh.num_vertices();
  const CellType::Type _cell_type = mesh.type().cell_type();
  const std::string cell_type = CellType::type2string(_cell_type);

  // Get cell offset and local cell range
  const uint cell_offset = MPI::global_offset(num_local_cells, true);
  const std::pair<uint, uint> cell_range(cell_offset, cell_offset + num_local_cells);

  // Get vertex offset and local vertex range
  const uint vertex_offset = MPI::global_offset(num_local_vertices, true);
  const std::pair<uint, uint> vertex_range(vertex_offset, vertex_offset + num_local_vertices);

  // FIXME: This is a bit clumsy because of lack of good support in DOLFIN
  //        for local/global indices. Replace when support in DOLFIN is
  //        improved
  // Get global vertex indices
  MeshFunction<uint> v_indices(mesh, 0);
  if (MPI::num_processes() == 1)
  {
    for (VertexIterator v(mesh); !v.end(); ++v)
      v_indices[*v] = v->index();
  }
  else
    v_indices = mesh.parallel_data().global_entity_indices(0);

  // Get vertex indices
  std::vector<uint> vertex_indices;
  std::vector<double> vertex_coords;
  vertex_indices.reserve(2*num_local_vertices);
  vertex_coords.reserve(3*num_local_vertices);
  const uint process_number = MPI::process_number();
  for (VertexIterator v(mesh); !v.end(); ++v)
  {
    // Vertex global index and process number
    vertex_indices.push_back(v_indices[*v]);
    vertex_indices.push_back(process_number);

    // Vertex coordinates
    const Point p = v->point();
    vertex_coords.push_back(p.x());
    vertex_coords.push_back(p.y());
    vertex_coords.push_back(p.z());
  }

  // Write vertex data to HDF5 file if not already there
  const std::string coord_dataset = mesh_coords_dataset_name(mesh);
  if (!dataset_exists(coord_dataset))
  {
    write(mesh_index_dataset_name(mesh), vertex_indices, 2);
    write(coord_dataset, vertex_coords, 3);
  }

  // Get cell connectivity
  // NOTE: For visualisation via XDMF, the vertex indices correspond
  //       to the local vertex position, and not the true vertex indices.
  std::vector<uint> topological_data;
  if (true_topology_indices)
  {
    // Build connectivity using true vertex indices
    for (CellIterator cell(mesh); !cell.end(); ++cell)
      for (VertexIterator v(*cell); !v.end(); ++v)
        topological_data.push_back(v_indices[*v]);
  }
  else
  {
    // Build connectivity using contiguous vertex indices
    for (CellIterator cell(mesh); !cell.end(); ++cell)
      for (VertexIterator v(*cell); !v.end(); ++v)
        topological_data.push_back(v->index() + vertex_range.first);
  }

  // Write connectivity to HDF5 file if not already there
  const std::string topology_dataset = mesh_topology_dataset_name(mesh);
  if (!dataset_exists(topology_dataset))
  {
    write(topology_dataset, topological_data, cell_dim + 1);
    uint topology_indicator = (true_topology_indices ? 1 : 0);
    HDF5Interface::add_attribute(filename, topology_dataset, "true_indexing", topology_indicator);
    HDF5Interface::add_attribute(filename, topology_dataset, "celltype", cell_type);
  }
}
//-----------------------------------------------------------------------------
void HDF5File::operator<< (const GenericVector& x)
{
  // Get local range;
  std::pair<uint, uint> range = x.local_range(0);

  // Get all local data
  std::vector<double> data;
  x.get_local(data);

  // Overwrite any existing file
  if (counter == 0)
    HDF5Interface::create(filename);

  // Write to HDF5 file
  const std::string name = "/Vector/" + boost::lexical_cast<std::string>(counter);
  write(name.c_str(),data, 1);

  // Increment counter
  counter++;
}
//-----------------------------------------------------------------------------
void HDF5File::operator>> (GenericVector& input)
{
  // Read vector from file, assuming partitioning is already known.
  // FIXME: should abort if not input is not allocated
  const std::pair<uint, uint> range = input.local_range(0);
  std::vector<double> data(range.second - range.first);
  HDF5Interface::read(filename, "/Vector/0", data, range, 1);
  input.set_local(data);
}
//-----------------------------------------------------------------------------
void HDF5File::write(const std::string dataset_name,
                     const std::vector<double>& data,
                     const uint width)
{
  // Write data contiguously from each process 
  uint num_items = data.size()/width;
  uint offset = MPI::global_offset(num_items,true);
  std::pair<uint,uint> range(offset, offset + num_items);
  HDF5Interface::write(filename, dataset_name, data, range, width);
}
//-----------------------------------------------------------------------------
void HDF5File::write(const std::string dataset_name,
                     const std::vector<uint>& data,
                     const uint width)
{
  // Write data contiguously from each process 
  uint num_items = data.size()/width;
  uint offset = MPI::global_offset(num_items,true);
  std::pair<uint,uint> range(offset, offset + num_items);

  HDF5Interface::write(filename, dataset_name, data, range, width);
}
//-----------------------------------------------------------------------------
void HDF5File::write(const std::string dataset_name,
                     const std::vector<int>& data,
                     const uint width)
{
  // Write data contiguously from each process 
  uint num_items = data.size()/width;
  uint offset = MPI::global_offset(num_items,true);
  std::pair<uint,uint> range(offset, offset + num_items);

  HDF5Interface::write(filename, dataset_name, data, range, width);
}

//-----------------------------------------------------------------------------
bool HDF5File::dataset_exists(const std::string &dataset_name)
{
  // Check for existence of dataset - used by XDMFFile
  return HDF5Interface::dataset_exists(filename, dataset_name);
}

//-----------------------------------------------------------------------------
// Work out the names to use for topology and coordinate datasets
// These routines need MPI to work out the hash
std::string HDF5File::mesh_coords_dataset_name(const Mesh& mesh) const
{
  std::stringstream dataset_name;
  dataset_name << "/Mesh/Coordinates_" << std::setfill('0')
          << std::hex << std::setw(8) << mesh.coordinates_hash();
  return dataset_name.str();
}
//-----------------------------------------------------------------------------
std::string HDF5File::mesh_index_dataset_name(const Mesh& mesh) const
{
  std::stringstream dataset_name;
  dataset_name << "/Mesh/GlobalIndex_" << std::setfill('0')
          << std::hex << std::setw(8) << mesh.coordinates_hash();
  return dataset_name.str();
}
//-----------------------------------------------------------------------------
std::string HDF5File::mesh_topology_dataset_name(const Mesh& mesh) const
{
  std::stringstream dataset_name;
  dataset_name << "/Mesh/Topology_" << std::setfill('0')
          << std::hex << std::setw(8) << mesh.topology_hash();
  return dataset_name.str();
}
//-----------------------------------------------------------------------------

#endif
