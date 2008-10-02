// Copyright (C) 2003-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Niclas Jansson, 2008.
//
// First added:  2003-10-21
// Last changed: 2008-09-26

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
  editor.initVertices(num_local);
  dolfin_debug2("Reading %d vertices out of %d", num_local, num_vertices);
  
  editor.initCells(1);

  // Create global numbering (mapping from local vertex to global vertex)
  global_numbering = _mesh.data().createMeshFunction("vertex numbering");
  global_numbering->init(_mesh, 0);

  // Create mapping from global vertex to local vertex
  global_to_local = _mesh.data().createMapping("global to local");

}
//-----------------------------------------------------------------------------
void PXMLMesh::readCells(const xmlChar *name, const xmlChar **attrs)
{

  editor.close();
  
  MeshFunction<uint> *ghost = _mesh.data().createMeshFunction("ghosted vertices");
  ghost->init(_mesh, 0);
  (*ghost) = 0;

  // Geometric partitioning of vertices
  MeshFunction<uint> geom_partition;
  _mesh.partitionGeom(geom_partition);
  _mesh.distribute(geom_partition);

  // Create the set of global numbers in the local part of the partitioned mesh
  global_numbering = _mesh.data().meshFunction("vertex numbering");
  for(VertexIterator v(_mesh); !v.end(); ++v)
    local_vertices.insert( global_numbering->get(*v));
  
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
  switch (_mesh.geometry().dim())
  {
  case 1:
    {
      double x = parseReal(name, attrs, "x");
      editor.addVertex(current_vertex, x);
    }
    break;
  case 2:
    {
      double x = parseReal(name, attrs, "x");
      double y = parseReal(name, attrs, "y");
      editor.addVertex(current_vertex, x, y);
    }
    break;
  case 3:
    {
      double x = parseReal(name, attrs, "x");
      double y = parseReal(name, attrs, "y");
      double z = parseReal(name, attrs, "z");
      editor.addVertex(current_vertex, x, y, z);
    }
    break;
  default:
    error("Dimension of mesh must be 1, 2 or 3.");
  }

  // Store global vertex numbering (mapping from local vertex to global vertex)
  global_numbering->set(current_vertex, v); 

  // FIXME: Is this used outside of PXMLMesh?
  // Yes, mesh partitioning (dual graph) for example

  // Store mapping from global vertex number to local vertex number
  (*global_to_local)[v] = current_vertex++;
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
  if (_mesh.topology().dim() != 3)
    error("Mesh entity (tetrahedron) does not match dimension of mesh (%d).",
		 _mesh.topology().dim());

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
  Mesh new_mesh;
  editor.open(new_mesh, _mesh.type().cellType(), 
	      _mesh.topology().dim(), _mesh.geometry().dim());

  uint process_number = MPI::process_number();
  uint num_processes = MPI::num_processes();

  Array<uint> send_buff, send_indices, send_orphan;
  Array<double> send_coords;

  // Construct send buffer with missing (shared) vertices global number 
  std::set<uint>::iterator it;
  for (it = shared_vertex.begin(); it != shared_vertex.end(); ++it)
      send_buff.push_back(*it);

  // Number of locally unused vertices
  uint num_orphan = _mesh.numVertices() - used_vertex.size();
  
  // Number of locally "missing" vertices
  uint num_shared = send_buff.size();

  // Number of locally missing coordinates
  uint num_coords = _mesh.geometry().dim() * num_shared;

  // Setup mesh editor for the final mesh
  editor.initVertices(_mesh.numVertices() + num_shared - num_orphan);

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
  
  global_numbering = _mesh.data().meshFunction("vertex numbering");
  global_to_local =  _mesh.data().mapping("global to local");
  num_orphan = num_shared;
   
  std::set<uint> assigned_orphan;
  
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
	Vertex v(_mesh, (*global_to_local)[ recv_shared[k] ]);
	send_coords.push_back(v.point().x());
	send_coords.push_back(v.point().y());
	if (_mesh.geometry().dim() > 2) 
	  send_coords.push_back(v.point().z());
	
	send_indices.push_back(recv_shared[k]);

	// If the vertex is unused (orphaned) at the processor, 
	// transfer ownership to the receiving processor
	if (used_vertex.find(recv_shared[k]) == used_vertex.end()  &&
	    assigned_orphan.find(recv_shared[k]) == assigned_orphan.end())
        {	  
	  send_orphan.push_back(1);

	  // Mark orphaned as assigned
	  assigned_orphan.insert(recv_shared[k]);	  
	}
	else
	  send_orphan.push_back(0);
	
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
  
  // New mappings from global to local and local to global vertex number
  std::map<uint, uint> new_lg, new_gl;

  // Set of ghosted vertices in final mesh
  std::set<uint> new_ghost;

  // Add all locally used vertices
  uint vi = 0;
  for (VertexIterator v(_mesh); !v.end(); ++v)
  {
    if (used_vertex.count(global_numbering->get(*v)))
    {

      // Store new mappings 
      new_gl[ global_numbering->get(*v) ] = vi;
      new_lg[ vi ]  = global_numbering->get(*v);

      editor.addVertex(vi++, v->point());
    }
  }

  // Add shared vertices
  uint ii = 0;
  for (uint i = 0; i < send_buff.size(); i++, ii += _mesh.geometry().dim(), vi++)
  {    
    
    // Store new mappings
    new_gl[ recv_indices[i] ] = vi;
    new_lg[ vi ] = recv_indices[i];

    // Mark shared vertices as ghosted
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

  // Add local cells, using the new global to local mapping for vertex numbers
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

  // Mark ghosted vertices in the mesh
  MeshFunction<uint> *ghost = _mesh.data().createMeshFunction("ghosted vertices");
  dolfin_assert(ghost);
  ghost->init(_mesh, 0);
  (*ghost) = 0;
  
  for (it = new_ghost.begin(); it != new_ghost.end(); ++it)
    ghost->set(*it, 1);

  // Set global numbering and mapping data for the final mesh
  global_numbering = _mesh.data().createMeshFunction("vertex numbering");
  global_to_local =  _mesh.data().createMapping("global to local");
  global_numbering->init(_mesh, 0);

  std::map<uint, uint>::iterator mit;
  for (mit = new_lg.begin(); mit != new_lg.end(); ++mit)
    global_numbering->set(mit->first, mit->second);
  
  for (mit = new_gl.begin(); mit != new_gl.end(); ++mit)
    (*global_to_local)[mit->first] = mit->second;
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
