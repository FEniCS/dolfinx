// Copyright (C) 2003-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Niclas Jansson, 2008.
//
// First added:  2003-10-21
// Last changed: 2008-10-29

#ifdef HAS_MPI
#include <mpi.h>
#endif

#ifdef HAS_PARMETIS
#include <parmetis.h>
#endif

#include <map>
#include <cstring>

#include <dolfin/log/log.h>
#include <dolfin/main/MPI.h>
#include <dolfin/mesh/CellType.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshData.h>
#include <dolfin/mesh/Vertex.h>
#include <dolfin/mesh/DynamicMeshEditor.h>
#include "PXMLMesh.h"

using namespace dolfin;

#ifdef HAS_MPI

//-----------------------------------------------------------------------------
PXMLMesh::PXMLMesh(Mesh& mesh)
  : XMLObject(), _mesh(mesh), state(OUTSIDE), f(0), a(0),
    first_vertex(0), last_vertex(0), current_vertex(0)
{
  dolfin_debug("Creating parallel XML parser");
}
//-----------------------------------------------------------------------------
PXMLMesh::~PXMLMesh()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void PXMLMesh::startElement(const xmlChar *name, const xmlChar **attrs)
{
  switch (state)
  {
  case OUTSIDE:
    
    if (xmlStrcasecmp(name, (xmlChar *) "mesh") == 0)
    {
      readMesh(name, attrs);
      state = INSIDE_MESH;
    }
    
    break;

  case INSIDE_MESH:
    
    if (xmlStrcasecmp(name, (xmlChar *) "vertices") == 0)
    {
      readVertices(name, attrs);
      state = INSIDE_VERTICES;
    }
    else if (xmlStrcasecmp(name, (xmlChar *) "cells") == 0)
    {
      state = INSIDE_CELLS;
    }
    else if (xmlStrcasecmp(name, (xmlChar *) "data") == 0)
    {
      error("Unable to read auxiliary mesh data in parallel, not implemented (yet).");
      state = INSIDE_DATA;
    }

    break;
    
  case INSIDE_VERTICES:
    
    if (xmlStrcasecmp(name, (xmlChar *) "vertex") == 0)
      readVertex(name, attrs);

    break;
    
  case INSIDE_CELLS:
    
    if (xmlStrcasecmp(name, (xmlChar *) "interval") == 0)
      readInterval(name, attrs);
    else if (xmlStrcasecmp(name, (xmlChar *) "triangle") == 0)
      readTriangle(name, attrs);
    else if (xmlStrcasecmp(name, (xmlChar *) "tetrahedron") == 0)
      readTetrahedron(name, attrs);
    
    break;

  case INSIDE_DATA:
    
    if (xmlStrcasecmp(name, (xmlChar *) "meshfunction") == 0)
    {
      readMeshFunction(name, attrs);
      state = INSIDE_MESH_FUNCTION;
    }
    if (xmlStrcasecmp(name, (xmlChar *) "array") == 0)
    {
      readArray(name, attrs);
      state = INSIDE_ARRAY;
    }

    break;

  case INSIDE_MESH_FUNCTION:
    
    if (xmlStrcasecmp(name, (xmlChar *) "entity") == 0)
      readMeshEntity(name, attrs);

    break;

  case INSIDE_ARRAY:
    
    if (xmlStrcasecmp(name, (xmlChar *) "element") == 0)
      readArrayElement(name, attrs);

    break;

  default:
    ;
  }
}
//-----------------------------------------------------------------------------
void PXMLMesh::endElement(const xmlChar *name)
{
  switch (state)
  {
  case INSIDE_MESH:
    
    if (xmlStrcasecmp(name, (xmlChar *) "mesh") == 0)
    {
      closeMesh();
      state = DONE;
    }
    
    break;
    
  case INSIDE_VERTICES:
    
    if (xmlStrcasecmp(name, (xmlChar *) "vertices") == 0)
    {
      partitionVertices();
      state = INSIDE_MESH;    
    }

    break;

  case INSIDE_CELLS:
	 
    if (xmlStrcasecmp(name, (xmlChar *) "cells") == 0)
    {
      state = INSIDE_MESH;
    }

    break;

  case INSIDE_DATA:

    if (xmlStrcasecmp(name, (xmlChar *) "data") == 0)
    {
      state = INSIDE_MESH;
    }

    break;

  case INSIDE_MESH_FUNCTION:

    if (xmlStrcasecmp(name, (xmlChar *) "meshfunction") == 0)
    {
      state = INSIDE_DATA;
    }

    break;

  case INSIDE_ARRAY:

    if (xmlStrcasecmp(name, (xmlChar *) "array") == 0)
    {
      state = INSIDE_DATA;
    }

    break;

  default:
    ;
  }
}
//-----------------------------------------------------------------------------
void PXMLMesh::open(std::string filename)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
bool PXMLMesh::close()
{
  return state == DONE;
}
//-----------------------------------------------------------------------------
void PXMLMesh::readMesh(const xmlChar *name, const xmlChar **attrs)
{
  // Parse values
  std::string type = parseString(name, attrs, "celltype");
  gdim = parseUnsignedInt(name, attrs, "dim");
  
  // Create cell type to get topological dimension
  CellType* cell_type = CellType::create(type);
  tdim = cell_type->dim();
  // Get number of entities for topological dimension 0
  num_cvert = cell_type->numEntities(0);
  delete cell_type;

  // Open mesh for editing
  editor.open(_mesh, CellType::string2type(type), tdim, gdim);
}
//-----------------------------------------------------------------------------
void PXMLMesh::readVertices(const xmlChar *name, const xmlChar **attrs)
{
  // Parse the number of vertices
  const uint num_vertices = parseUnsignedInt(name, attrs, "size");

  // Get process number and number of processes
  const uint num_processes = MPI::num_processes();
  const uint process_number = MPI::process_number();

  // Compute number of vertices per process and remainder
  const uint n = num_vertices / num_processes;
  const uint r = num_vertices % num_processes;

  // Distribute remainder evenly among first r processes
  num_local = 0;
  if (process_number < r)
  {
    num_local    = n + 1;
    first_vertex = process_number*n + process_number;
  }
  else
  {
    num_local    = n;
    first_vertex = process_number*n + r;
  }
  last_vertex = first_vertex + num_local - 1;

  // Set number of vertices
  dolfin_debug2("Reading %d vertices out of %d", num_local, num_vertices);
  
  global_number = new uint[num_local];
  vertex_buffer = new double[gdim * num_local];

  gp = &global_number[0];
  vp = &vertex_buffer[0];



}
//-----------------------------------------------------------------------------
void PXMLMesh::readVertex(const xmlChar *name, const xmlChar **attrs)
{
  // Read index
  const uint v = parseUnsignedInt(name, attrs, "index");

  // FIXME: We could optimize here so that we don't need to
  // FIXME: parse the entire mesh on each processor

  // Skip vertices not in range for this process
  if (v < first_vertex || v > last_vertex)
    return;
  
  // Handle differently depending on geometric dimension
  switch ( gdim )
  {
  case 1:
    {
      *(vp++) = parseReal(name, attrs, "x");
    }
    break;
  case 2:
    {
      *(vp++) = parseReal(name, attrs, "x");
      *(vp++) = parseReal(name, attrs, "y");
    }
    break;
  case 3:
    {
      *(vp++) = parseReal(name, attrs, "x");
      *(vp++) = parseReal(name, attrs, "y");
      *(vp++) = parseReal(name, attrs, "z");
    }
    break;
  default:
    error("Dimension of mesh must be 1, 2 or 3.");
  }

  // Store global vertex numbering 
  *(gp++) = v;
}
//-----------------------------------------------------------------------------
void PXMLMesh::readInterval(const xmlChar *name, const xmlChar **attrs)
{
  // Check dimension
  if (tdim != 1)
    error("Mesh entity (interval) does not match dimension of mesh (%d).", tdim);

  // Parse values
  uint c  = parseUnsignedInt(name, attrs, "index");
  uint v0 = parseUnsignedInt(name, attrs, "v0");
  uint v1 = parseUnsignedInt(name, attrs, "v1");
  
  // Add cell
  editor.addCell(c, v0, v1);
}
//-----------------------------------------------------------------------------
void PXMLMesh::readTriangle(const xmlChar *name, const xmlChar **attrs)
{
  // Check dimension
  if (tdim != 2)  
    error("Mesh entity (triangle) does not match dimension of mesh (%d).", tdim);

  // Parse values
  uint v0 = parseUnsignedInt(name, attrs, "v0");
  uint v1 = parseUnsignedInt(name, attrs, "v1");
  uint v2 = parseUnsignedInt(name, attrs, "v2");
  
  // Skip triangle if vertex v0 is not local
  if (!is_local(v0))
    return;

  // Mark which (local) vertices a cell is using
  used_vertex.insert(v0);
  if (is_local(v1))
    used_vertex.insert(v1);
  if (is_local(v2))
    used_vertex.insert(v2); 

  // Mark vertices which are not present on the processor
  if (!is_local(v1))
    shared_vertex.insert(v1);
  if (!is_local(v2))
    shared_vertex.insert(v2);

  // Add cell to buffer
  cell_buffer.push_back(v0);
  cell_buffer.push_back(v1);
  cell_buffer.push_back(v2);
}
//-----------------------------------------------------------------------------
void PXMLMesh::readTetrahedron(const xmlChar *name, const xmlChar **attrs)
{
  // Check dimension
  if (tdim != 3)
    error("Mesh entity (tetrahedron) does not match dimension of mesh (%d).", tdim);

  // Parse values
  uint v0 = parseUnsignedInt(name, attrs, "v0");
  uint v1 = parseUnsignedInt(name, attrs, "v1");
  uint v2 = parseUnsignedInt(name, attrs, "v2");
  uint v3 = parseUnsignedInt(name, attrs, "v3");

  // Skip tetrahedron if vertex v0 is not local
  if (!is_local(v0))
    return;

  // Mark which (local) vertices a cell is using
  used_vertex.insert(v0);
  if (is_local(v1))
    used_vertex.insert(v1);
  if (is_local(v2))
    used_vertex.insert(v2);
  if (is_local(v3))
    used_vertex.insert(v3);

  // Mark vertices which are not present on the processor
  if (!is_local(v1))
    shared_vertex.insert(v1);
  if (!is_local(v2))
    shared_vertex.insert(v2);
  if (!is_local(v3))
    shared_vertex.insert(v3);
        
  // Add cell to buffer
  cell_buffer.push_back(v0);
  cell_buffer.push_back(v1);
  cell_buffer.push_back(v2);
  cell_buffer.push_back(v3);
}
//-----------------------------------------------------------------------------
void PXMLMesh::readMeshFunction(const xmlChar* name, const xmlChar** attrs)
{
  // Parse values
  const std::string id = parseString(name, attrs, "name");
  const std::string type = parseString(name, attrs, "type");
  const uint dim = parseUnsignedInt(name, attrs, "dim");
  const uint size = parseUnsignedInt(name, attrs, "size");

  // Only uint supported at this point
  if (strcmp(type.c_str(), "uint") != 0)
    error("Only uint-valued mesh data is currently supported.");

  // Check size
  _mesh.init(dim);
  if (_mesh.size(dim) != size)
    error("Wrong number of values for MeshFunction, expecting %d.", _mesh.size(dim));

  // Register data
  f = _mesh.data().createMeshFunction(id);
  dolfin_assert(f);
  f->init(_mesh, dim);

  // Set all values to zero
  *f = 0;
}
//-----------------------------------------------------------------------------
void PXMLMesh::readArray(const xmlChar* name, const xmlChar** attrs)
{
  // Parse values
  const std::string id = parseString(name, attrs, "name");
  const std::string type = parseString(name, attrs, "type");
  const uint size = parseUnsignedInt(name, attrs, "size");

  // Only uint supported at this point
  if (strcmp(type.c_str(), "uint") != 0)
    error("Only uint-valued mesh data is currently supported.");

  // Register data
  a = _mesh.data().createArray(id, size);
  dolfin_assert(a);
}
//-----------------------------------------------------------------------------
void PXMLMesh::readMeshEntity(const xmlChar* name, const xmlChar** attrs)
{
  // Read index
  const uint index = parseUnsignedInt(name, attrs, "index");

  // Read and set value
  dolfin_assert(f);
  dolfin_assert(index < f->size());
  const uint value = parseUnsignedInt(name, attrs, "value");
  f->set(index, value);
}
//-----------------------------------------------------------------------------
void PXMLMesh::readArrayElement(const xmlChar* name, const xmlChar** attrs)
{
  // Read index
  const uint index = parseUnsignedInt(name, attrs, "index");

  // Read and set value
  dolfin_assert(a);
  dolfin_assert(index < a->size());
  const uint value = parseUnsignedInt(name, attrs, "value");
  (*a)[index] = value;
}
//-----------------------------------------------------------------------------
void PXMLMesh::closeMesh()
{

  // Temporary mapping from global to local vertex numbering
  std::map<uint, uint> _global_to_local, __global_to_local;
  
  // Add all local vertices to the mesh
  uint current_vertex = 0;
  uint j = 0;
  for(uint i = 0; i < num_local * gdim; i += gdim, j++)
  {
    if(used_vertex.find(global_number[j]) != used_vertex.end())
    {
	switch(gdim)
	{
	case 1:
	  editor.addVertex(current_vertex, vertex_buffer[i]); 
	  break;
	case 2:
	  editor.addVertex(current_vertex, vertex_buffer[i], 
			   vertex_buffer[i+1]); 
	  break;
	case 3:
	  editor.addVertex(current_vertex, vertex_buffer[i], 
			   vertex_buffer[i+1], vertex_buffer[i+2]); 
	  break;
	}
	// Temporary mapping for all used vertices
	_global_to_local[global_number[j]] = current_vertex++;    
    }
    
    // Temporary mapping for all vertices (used during communication)
    __global_to_local[global_number[j]] = j;
  }

  uint process_number = MPI::process_number();
  uint num_processes = MPI::num_processes();

  Array<uint> send_buff, send_indices, send_orphan;
  Array<double> send_coords;

  // Construct send buffer with missing (shared) vertices global number 
  std::set<uint>::iterator it;
  for (it = shared_vertex.begin(); it != shared_vertex.end(); ++it)
    send_buff.push_back(*it);

  // Number of locally unused vertices
  uint num_orphan = num_local - used_vertex.size();
  
  // Number of locally "missing" vertices
  uint num_shared = send_buff.size();

  // Number of locally missing coordinates
  uint num_coords = gdim * num_shared;

  // Calculate maximum number of shared vertices to be received
  uint max_nsh;
  MPI_Allreduce(&num_shared, &max_nsh, 1, MPI_UNSIGNED, MPI_MAX, MPI_COMM_WORLD);

  // Allocate receive buffers
  uint *recv_shared  = new uint[max_nsh];
  double *recv_coords = new double[num_coords];
  uint *recv_indices = new uint[num_shared];
  uint *recv_orphans = new uint[num_shared];

  // Pointers pointing to the current position in the receive buffers
  double *rcp = &recv_coords[0];
  uint *rip = &recv_indices[0];
  uint *rop = &recv_orphans[0];

  num_orphan = num_shared;
   
  std::set<uint> assigned_orphan;
  std::map<uint, uint> owner_map;

  MPI_Status status;
  int num_recv, src, dest;
  for (uint j = 1; j < num_processes ; j++){
    src = (process_number - j + num_processes) % num_processes;
    dest = (process_number + j) % num_processes;

    // Exchange shared global numbers
    MPI_Sendrecv(&send_buff[0], send_buff.size(), MPI_UNSIGNED, dest, 1,
		 recv_shared, max_nsh, MPI_UNSIGNED, src, 1, 
		 MPI_COMM_WORLD, &status);
    MPI_Get_count(&status, MPI_UNSIGNED, &num_recv);

    for (int k = 0; k < num_recv; k++) 
    {
      // If the receiving processor has the shared vertex, 
      // send back the coordinates
      if (is_local(recv_shared[k]))
      {
	// Determine offset in vertex buffer
	uint v_index = (__global_to_local[recv_shared[k]]) * gdim;
	send_coords.push_back(vertex_buffer[v_index]);
	send_coords.push_back(vertex_buffer[v_index + 1]);
	if (gdim > 2)
	  send_coords.push_back(vertex_buffer[v_index + 2]);	
	send_indices.push_back(recv_shared[k]);

	// If the vertex is unused (orphaned) at the processor, 
	// transfer ownership to the receiving processor
	if (used_vertex.find(recv_shared[k]) == used_vertex.end()  &&
	    assigned_orphan.find(recv_shared[k]) == assigned_orphan.end())
        {	  
	  send_orphan.push_back(num_processes);
	  owner_map[recv_shared[k]] = status.MPI_SOURCE;

	  // Mark orphaned as assigned
	  assigned_orphan.insert(recv_shared[k]);
	}
	else
	  if(owner_map.find(recv_shared[k]) == owner_map.end())
	    send_orphan.push_back(process_number);
	  else
	    send_orphan.push_back(owner_map[recv_shared[k]]);
      }
    }
    
    // Send back coordinates, indices and ownership information
    MPI_Sendrecv(&send_indices[0], send_indices.size(), MPI_UNSIGNED, 
		 src, 1, rip, num_shared, MPI_UNSIGNED, dest, 1,
		 MPI_COMM_WORLD, &status);
    MPI_Get_count(&status, MPI_UNSIGNED, &num_recv);
    num_shared -= num_recv;
    rip += num_recv;

    MPI_Sendrecv(&send_coords[0], send_coords.size(), MPI_DOUBLE, src, 2,
		 rcp, num_coords, MPI_DOUBLE, dest, 2, 
		 MPI_COMM_WORLD, &status);
    MPI_Get_count(&status, MPI_DOUBLE, &num_recv);
    num_coords -= num_recv;
    rcp += num_recv;
    
    MPI_Sendrecv(&send_orphan[0], send_orphan.size(), MPI_UNSIGNED, src, 3,
		 rop, num_orphan, MPI_UNSIGNED, dest, 3, 
		 MPI_COMM_WORLD, &status);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Get_count(&status, MPI_UNSIGNED, &num_recv);
    num_orphan -= num_recv;
    rop += num_recv;
    
    // Clear temporary send buffers
    send_indices.clear();
    send_coords.clear();
    send_orphan.clear();
  }
  
  std::map<uint, uint> new_owner;


  // Add shared vertices
  uint ii = 0;
  for (uint i = 0; i < send_buff.size(); i++, ii += gdim, current_vertex++)
  {    
    
    // Store new mappings
    _global_to_local[recv_indices[i]] = current_vertex;
    
    // Store owner of shared vertex
    if (recv_orphans[i] < num_processes)
      new_owner[current_vertex] = recv_orphans[i];

    switch(gdim)
    {
    case 2:
      editor.addVertex(current_vertex, recv_coords[ii], recv_coords[ii+1]); break;
    case 3:
      editor.addVertex(current_vertex, recv_coords[ii], recv_coords[ii+1], 
			recv_coords[ii+2]); break;
     }
  }

  // Add local cells, using the new global to local mapping for vertex numbers
  uint ci = 0;
  for (uint i = 0; i < cell_buffer.size(); i += num_cvert, ci++)
  {
    switch (num_cvert) 
    {
    case 2:
      editor.addCell(ci, _global_to_local[cell_buffer[i]], 
		     _global_to_local[cell_buffer[i+1]]);
      break;
    case 3:
      editor.addCell(ci, _global_to_local[cell_buffer[i]], 
		     _global_to_local[cell_buffer[i+1]],
		     _global_to_local[cell_buffer[i+2]]);
      break;
    case 4:
      editor.addCell(ci, _global_to_local[cell_buffer[i]], 
		     _global_to_local[cell_buffer[i+1]],
		     _global_to_local[cell_buffer[i+2]],
		     _global_to_local[cell_buffer[i+3]]);
      break;
    }
  }

  editor.close(false);

  delete [] recv_shared;
  delete [] recv_coords;
  delete [] recv_indices;
  delete [] recv_orphans;
  delete [] global_number;
  delete [] vertex_buffer;
  
  // Recreate auxiliary mesh data

  // Set owner of vertices
  std::map<uint, uint>::iterator mit;  
  MeshFunction<uint> *owner = _mesh.data().createMeshFunction("vertex owner");  
  dolfin_assert(owner);
  owner->init(_mesh, 0);
  (*owner) = process_number;

  for (mit = new_owner.begin(); mit != new_owner.end(); ++mit)
    owner->set(mit->first, mit->second);

  // Set global numbering and mapping data for the final mesh
  global_numbering = _mesh.data().createMeshFunction("vertex numbering");
  global_to_local =  _mesh.data().createMapping("global to local");
  global_numbering->init(_mesh, 0);

  for(mit = _global_to_local.begin(); mit  != _global_to_local.end(); ++mit)
  {
    if(used_vertex.find(mit->first) != used_vertex.end()) 
    {
      global_numbering->set(mit->second, mit->first);
      (*global_to_local)[mit->first] = mit->second;
    }
  }
}
//-----------------------------------------------------------------------------
void PXMLMesh::partitionVertices()
{
#ifdef HAS_PARMETIS

  // Duplicate MPI communicator
  MPI_Comm comm; 
  MPI_Comm_dup(MPI_COMM_WORLD, &comm);

  uint size = MPI::num_processes();
  uint rank = MPI::process_number();

  dolfin_debug("Geometric partitioning of vertices");
  
  // Create the vertex distribution array (vtxdist) 
  idxtype *vtxdist = new idxtype[size+1];  
  vtxdist[rank] = static_cast<idxtype>(num_local);
  
  MPI_Allgather(&vtxdist[rank], 1, MPI_INT, vtxdist, 1, 
		MPI_INT, MPI_COMM_WORLD);

  int sum = vtxdist[0];  
  vtxdist[0] = 0;
  for (uint i = 1; i < size + 1; i++)
  {
    const int tmp = vtxdist[i];
    vtxdist[i] = sum;
    sum = tmp + sum;
  }

  idxtype _gdim = static_cast<idxtype>(gdim);
  idxtype *part = new idxtype[num_local];
  float *xdy = new float[gdim * num_local];
  
  for (uint i = 0; i < gdim * num_local; i++)
    xdy[i] = static_cast<float>(vertex_buffer[i]);

  // Call ParMETIS to partition vertex distribution array
  ParMETIS_V3_PartGeom(vtxdist, &_gdim, xdy, part, &comm);

  Array<uint> *send_buff_glb = new Array<uint>[size];
  Array<double> *send_buff_coords = new Array<double>[size];
  std::set<uint> kept_local;

  // Duplicate vertex and global number buffers
  double *tmp_buffer = new double[num_local * gdim];  
  memcpy(&tmp_buffer[0], &vertex_buffer[0], gdim * num_local * sizeof(double));
  delete[] vertex_buffer;

  uint *tmp_glb = new uint[num_local];
  memcpy(&tmp_glb[0], &global_number[0], num_local * sizeof(uint));
  delete[] global_number;

  // Process vertex distribution
  for (uint i = 0; i < num_local; i++) 
  {
    if ( part[i] != (idxtype) rank) 
    {
      send_buff_glb[part[i]].push_back(tmp_glb[i]);
      for (uint j = 0; j < gdim; j++)
	send_buff_coords[part[i]].push_back(tmp_buffer[i*gdim + j]);
    }
    else 
      kept_local.insert(i);
  }
  
  // Determine size of receive buffers 
  int recv_count, recv_size, send_size;  
  for (uint i = 0; i < size; i++) {
    send_size = send_buff_coords[i].size();
    MPI_Reduce(&send_size, &recv_count, 1, MPI_INT, MPI_MAX, i, MPI_COMM_WORLD);
    send_size = send_buff_glb[i].size();
    MPI_Reduce(&send_size, &num_local, 1, MPI_INT, MPI_SUM, i, MPI_COMM_WORLD);
  }
  num_local += kept_local.size();

  // Allocate memory for receive buffers
  uint recv_count_glb = recv_count / gdim;  
  double *recv_buff = new double[recv_count];
  uint* recv_buff_glb = new uint[recv_count_glb];  


  // Allocate new memory for vertex buffer 
  vertex_buffer = new double[num_local * gdim];
  vp = &vertex_buffer[0];

  // Allocate new memory for global numbering
  global_number = new uint[num_local];
  gp = &global_number[0];

  // Add local vertices to new vertex buffer
  std::set<uint>::iterator it;
  for(it = kept_local.begin(); it != kept_local.end(); it++) 
  {
    *(gp++) = tmp_glb[*it];
    local_vertices.insert(tmp_glb[*it]);
    for (uint j = 0; j < gdim; j++)
      *(vp++) = tmp_buffer[(*it) * gdim + j];
  }

  // Exchange vertices and global numbers
  MPI_Status status;
  uint src,dest;
  for (uint i = 1; i < size; i++) 
  {    
    src = (rank - i + size) % size;
    dest = (rank + i) % size;
    
    MPI_Sendrecv(&send_buff_glb[dest][0], send_buff_glb[dest].size(),
		 MPI_UNSIGNED, dest, 0, recv_buff_glb, recv_count_glb, 
		 MPI_UNSIGNED, src, 0, MPI_COMM_WORLD, &status);
    MPI_Get_count(&status, MPI_UNSIGNED, &recv_size);
    
    MPI_Sendrecv(&send_buff_coords[dest][0], send_buff_coords[dest].size(),
		 MPI_DOUBLE, dest, 1, recv_buff, recv_count, MPI_DOUBLE, src,
		 1, MPI_COMM_WORLD, &status);

    for (uint j = 0; j < (uint) recv_size; j++) 
    {
      *(gp++) = recv_buff_glb[j];
      local_vertices.insert(recv_buff_glb[j]);
      for (uint jj = 0; jj < gdim; jj++)
	*(vp++) = recv_buff[j * gdim + jj];
    }    
  }
  
  // Clean up
  for (uint i = 0; i < size; i++)
  {
    send_buff_glb[i].clear();
    send_buff_coords[i].clear();
  }
  delete[] send_buff_glb;
  delete[] send_buff_coords;
  delete[] tmp_glb;
  delete[] tmp_buffer;
  delete[] xdy;
  delete[] part;
  delete[] vtxdist;

#else

  error("Missing ParMETIS for parallel XML parser.");

#endif
}
//-----------------------------------------------------------------------------

#else

//-----------------------------------------------------------------------------
PXMLMesh::PXMLMesh(Mesh& mesh) : _mesh(mesh)
{
  error("Missing MPI for parallel XML parser.");
}
//-----------------------------------------------------------------------------
PXMLMesh::~PXMLMesh() {}
//-----------------------------------------------------------------------------
void PXMLMesh::startElement(const xmlChar *name, const xmlChar **attrs) {}
//-----------------------------------------------------------------------------
void PXMLMesh::endElement(const xmlChar *name) {}
//-----------------------------------------------------------------------------
void PXMLMesh::open(std::string filename) {}
//-----------------------------------------------------------------------------
bool PXMLMesh::close() { return false; }
//-----------------------------------------------------------------------------

#endif
