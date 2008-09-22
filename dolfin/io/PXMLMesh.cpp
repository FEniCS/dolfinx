// Copyright (C) 2003-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Niclas Jansson, 2008.
//
// First added:  2003-10-21
// Last changed: 2008-09-22

#ifdef HAS_MPI
#include <mpi.h>
#endif

#include <map>
#include <cstring>

#include <dolfin/log/log.h>
#include <dolfin/main/MPI.h>
#include <dolfin/mesh/CellType.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshData.h>
#include <dolfin/mesh/Vertex.h>
#include "PXMLMesh.h"

using namespace dolfin;

#ifdef HAS_MPI

//-----------------------------------------------------------------------------
PXMLMesh::PXMLMesh(Mesh& mesh)
  : XMLObject(), _mesh(mesh), state(OUTSIDE), f(0), a(0)
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
      readCells(name, attrs);
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
  uint gdim = parseUnsignedInt(name, attrs, "dim");
  
  // Create cell type to get topological dimension
  CellType* cell_type = CellType::create(type);
  uint tdim = cell_type->dim();
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
  uint num_local = 0;
  if (process_number < r)
  {
    num_local   = n + 1;
    start_index = process_number*n + process_number;
  }
  else
  {
    num_local   = n;
    start_index = process_number*n + r;
  }
  end_index = start_index + num_local - 1;

  num_parsed_v = 0;

  // Set number of vertices
  editor.initVertices(num_local);
  dolfin_debug2("Reading %d vertices out of %d", num_local, num_vertices);
  
  editor.initCells(1);
  global_numbering = _mesh.data().createMeshFunction("vertex numbering");

  local_to_global = _mesh.data().createMap("global to local");
  
  global_numbering->init(_mesh, 0);
}
//-----------------------------------------------------------------------------
void PXMLMesh::readCells(const xmlChar *name, const xmlChar **attrs)
{
  // Parse values

  editor.close();

  MeshFunction<uint> *ghost = _mesh.data().createMeshFunction("ghosted vertices");
  ghost->init(_mesh, 0);
  (*ghost) = 0;


  MeshFunction<uint> geom_partition;
  _mesh.partitionGeom(geom_partition);
  _mesh.distribute(geom_partition);

  global_numbering = _mesh.data().meshFunction("vertex numbering");

  for (VertexIterator v(_mesh); !v.end(); ++v) 
    local_vertex.insert(global_numbering->get(*v));
  it = local_vertex.end();

}
//-----------------------------------------------------------------------------
void PXMLMesh::readVertex(const xmlChar *name, const xmlChar **attrs)
{
  // Read index
  uint v = parseUnsignedInt(name, attrs, "index");

  if(v < start_index || v > end_index) 
    return;
  
  // Handle differently depending on geometric dimension
  switch (_mesh.geometry().dim())
  {
  case 1:
    {
      real x = parseReal(name, attrs, "x");
      editor.addVertex(num_parsed_v, x);
    }
    break;
  case 2:
    {
      real x = parseReal(name, attrs, "x");
      real y = parseReal(name, attrs, "y");
      editor.addVertex(num_parsed_v, x, y);
    }
    break;
  case 3:
    {
      real x = parseReal(name, attrs, "x");
      real y = parseReal(name, attrs, "y");
      real z = parseReal(name, attrs, "z");
      editor.addVertex(num_parsed_v, x, y, z);
    }
    break;
  default:
    error("Dimension of mesh must be 1, 2 or 3.");
  }

  global_numbering->set(num_parsed_v, v); 
  (*local_to_global)[v] = num_parsed_v++;
}
//-----------------------------------------------------------------------------
void PXMLMesh::readInterval(const xmlChar *name, const xmlChar **attrs)
{
  // Check dimension
  if (_mesh.topology().dim() != 1)
    error("Mesh entity (interval) does not match dimension of mesh (%d).",
		 _mesh.topology().dim());

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
  if (_mesh.topology().dim() != 2)
    error("Mesh entity (triangle) does not match dimension of mesh (%d).",
		 _mesh.topology().dim());

  // Parse values
  uint v0 = parseUnsignedInt(name, attrs, "v0");
  uint v1 = parseUnsignedInt(name, attrs, "v1");
  uint v2 = parseUnsignedInt(name, attrs, "v2");
  
  // Skip triangle if no vertices are local
  if(!(local_vertex.find(v2) != it || local_vertex.find(v1) != it ||
       local_vertex.find(v0) != it) || local_vertex.find(v0) == it)
    return;

  used_vertex.insert(v0);
  if (local_vertex.find(v1) != it)
    used_vertex.insert(v1);
  if (local_vertex.find(v2) != it)
    used_vertex.insert(v2); 


  if(!(local_vertex.find(v1) != it && local_vertex.find(v2) != it &&
       local_vertex.find(v0) != it))
  {
    if(local_vertex.find(v1) == it)
      shared_vertex.insert(v1);
    if(local_vertex.find(v2) == it)
      shared_vertex.insert(v2);
  }
  
  // Add cell to buffer
  cell_buffer.push_back(v0);
  cell_buffer.push_back(v1);
  cell_buffer.push_back(v2);
}
//-----------------------------------------------------------------------------
void PXMLMesh::readTetrahedron(const xmlChar *name, const xmlChar **attrs)
{
  // Check dimension
  if (_mesh.topology().dim() != 3)
    error("Mesh entity (tetrahedron) does not match dimension of mesh (%d).",
		 _mesh.topology().dim());

  // Parse values
  uint v0 = parseUnsignedInt(name, attrs, "v0");
  uint v1 = parseUnsignedInt(name, attrs, "v1");
  uint v2 = parseUnsignedInt(name, attrs, "v2");
  uint v3 = parseUnsignedInt(name, attrs, "v3");

  // Skip tetrahedron if no vertices are local
  if(!(local_vertex.find(v3) != it || local_vertex.find(v2) != it ||
       local_vertex.find(v1) != it || local_vertex.find(v0) != it) ||
     local_vertex.find(v0) == it)
    return;

  used_vertex.insert(v0);
  if (local_vertex.find(v1) != it)
    used_vertex.insert(v1);
  if (local_vertex.find(v2) != it)
    used_vertex.insert(v2);
  if (local_vertex.find(v3) != it)
    used_vertex.insert(v3);


  if(!(local_vertex.find(v3) != it && local_vertex.find(v2) != it &&
       local_vertex.find(v1) != it && local_vertex.find(v0) != it))
  {
    if(local_vertex.find(v1) == it)
      shared_vertex.insert(v1);
    if(local_vertex.find(v2) == it)
      shared_vertex.insert(v2);
    if(local_vertex.find(v3) == it)
      shared_vertex.insert(v3);
  }
        
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
  Mesh new_mesh;
  editor.open(new_mesh, _mesh.type().cellType(), 
	      _mesh.topology().dim(), _mesh.geometry().dim());

  uint process_number = MPI::process_number();
  uint num_processes = MPI::num_processes();

  Array<uint> send_buff, send_indices, send_orphan;
  Array<real> send_coords;

  std::set<uint>::iterator it;
  for (it = shared_vertex.begin(); it != shared_vertex.end(); ++it)
      send_buff.push_back(*it);

  uint num_orphan = _mesh.numVertices() - used_vertex.size();
  uint num_shared = send_buff.size();
  uint num_coords = _mesh.geometry().dim() * num_shared;

  editor.initVertices(_mesh.numVertices() + num_shared - num_orphan);

  uint max_nsh;
  MPI_Allreduce(&num_shared, &max_nsh, 1, MPI_UNSIGNED, MPI_MAX, MPI_COMM_WORLD);
  uint *recv_shared  = new uint[max_nsh];
  real *recv_coords = new real[num_coords];
  uint *recv_indices = new uint[num_shared];
  uint *recv_orphans = new uint[num_shared];

  real *rcp = &recv_coords[0];
  uint *rip = &recv_indices[0];
  uint *rop = &recv_orphans[0];
  
  MeshFunction<bool> ghosted(_mesh, 0);
  ghosted = false;  

  global_numbering = _mesh.data().meshFunction("vertex numbering");
  local_to_global =  _mesh.data().mapping("global to local");
  num_orphan = num_shared;
   
  // Exchange shared mesh entities
  MPI_Status status;
  int num_recv, src, dest;
  for (uint j = 1; j < num_processes ; j++){
    src = (process_number - j + num_processes) % num_processes;
    dest = (process_number + j) % num_processes;

    MPI_Sendrecv(&send_buff[0], send_buff.size(), MPI_UNSIGNED, dest, 1,
		 recv_shared, max_nsh, MPI_UNSIGNED, src, 1, 
		 MPI_COMM_WORLD, &status);
    MPI_Get_count(&status, MPI_UNSIGNED, &num_recv);

    for (int k = 0; k < num_recv; k++) 
    {
      if (local_vertex.find(recv_shared[k]) != local_vertex.end())
      {
	Vertex v(_mesh, (*local_to_global)[ recv_shared[k] ]);
	send_coords.push_back(v.point().x());
	send_coords.push_back(v.point().y());
	if (_mesh.geometry().dim() > 2) 
	  send_coords.push_back(v.point().z());
	
	send_indices.push_back(recv_shared[k]);

	if (used_vertex.find(recv_shared[k]) == used_vertex.end() &&
            shared_vertex.find(recv_shared[k]) != shared_vertex.end())
        {
	  
	  send_orphan.push_back(1);
	  shared_vertex.erase(recv_shared[k]);
	}
	else
	  send_orphan.push_back(0);

       	ghosted.set((*local_to_global)[recv_shared[k]], true);
	
      }
    }
    
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
    
    send_indices.clear();
    send_coords.clear();
    send_orphan.clear();
    
  }

  std::map<uint, uint> new_lg, new_gl;
  std::set<uint> new_ghost;

  uint vi = 0;
  for (VertexIterator v(_mesh); !v.end(); ++v)
  {
    if(used_vertex.count(global_numbering->get(*v)))
    {
      new_gl[ global_numbering->get(*v) ] = vi;
      new_lg[ vi ]  = global_numbering->get(*v);
      if (ghosted.get(*v))
	new_ghost.insert(vi);	
      editor.addVertex(vi++, v->point());
    }
  }

  uint ii = 0;
  for (uint i = 0; i < send_buff.size(); i++, ii += _mesh.geometry().dim(), vi++)
  {    
    new_gl[ recv_indices[i] ] = vi;
    new_lg[ vi ] = recv_indices[i];
    if (recv_orphans[i] == 0)
      new_ghost.insert(vi);
    switch(_mesh.geometry().dim())
    {
    case 2:
      editor.addVertex(vi, recv_coords[ii], recv_coords[ii+1]); break;
    case 3:
      editor.addVertex(vi, recv_coords[ii], recv_coords[ii+1], 
		       recv_coords[ii+2]); break;
    }
  }

  uint num_cvert = _mesh.type().numVertices(_mesh.topology().dim());
  editor.initCells(cell_buffer.size() / num_cvert);
  
  uint ci = 0;
  for (uint i = 0; i < cell_buffer.size(); i += num_cvert, ci++)
  {
    switch (num_cvert) 
    {
    case 2:
      editor.addCell(ci, new_gl[cell_buffer[i]], new_gl[cell_buffer[i+1]]);
      break;
    case 3:
      editor.addCell(ci, new_gl[cell_buffer[i]], new_gl[cell_buffer[i+1]],
		     new_gl[cell_buffer[i+2]]);
      break;
    case 4:
      editor.addCell(ci, new_gl[cell_buffer[i]], new_gl[cell_buffer[i+1]],
		     new_gl[cell_buffer[i+2]], new_gl[cell_buffer[i+3]]);
      break;
    }
  }

  editor.close();
  _mesh = new_mesh;
  
  delete[] recv_shared;
  delete[] recv_coords;
  delete[] recv_indices;
  delete[] recv_orphans;
  
  // Recreate auxiliary mesh data
  MeshFunction<uint> *ghost = _mesh.data().createMeshFunction("ghosted vertices");
  dolfin_assert(ghost);
  ghost->init(_mesh, 0);
  (*ghost) = 0;
   
  for (it = new_ghost.begin(); it != new_ghost.end(); ++it)
    ghost->set(*it, 1);

  global_numbering = _mesh.data().createMeshFunction("vertex numbering");
  local_to_global =  _mesh.data().createMap("global to local");
  global_numbering->init(_mesh, 0);

  std::map<uint, uint>::iterator mit;
  for (mit = new_lg.begin(); mit != new_lg.end(); ++mit)
    global_numbering->set(mit->first, mit->second);
  
  for (mit = new_gl.begin(); mit != new_gl.end(); ++mit)
    (*local_to_global)[mit->first] = mit->second;
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
