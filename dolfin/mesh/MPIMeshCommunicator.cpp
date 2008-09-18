// Copyright (C) 2007 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Magnus Vikstrøm, 2007.
// Modified by Anders Logg, 2007.
// Modified by Niclas Jansson, 2008.
//
// First added:  2007-05-30
// Last changed: 2008-09-18

#include <dolfin/log/dolfin_log.h>


#include "Mesh.h"
#include "MeshFunction.h"
#include "Vertex.h"
#include "MPIMeshCommunicator.h"
#include <dolfin/main/MPI.h>

#include "MeshData.h"
#include "MeshEditor.h"

#ifdef HAS_MPI
  #include <mpi.h>
  #include <map>
#endif


using namespace dolfin;

#ifdef HAS_MPI

//-----------------------------------------------------------------------------
MPIMeshCommunicator::MPIMeshCommunicator()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
MPIMeshCommunicator::~MPIMeshCommunicator()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void MPIMeshCommunicator::broadcast(const Mesh& mesh)
{
  int process_int;
  MPI_Comm_rank(MPI_COMM_WORLD, &process_int);
  unsigned int this_process = process_int;

  // Define custom MPI datatype?
  //MPI_Datatype mpi_mesh;

  // Mesh geometry

  // Send size
  uint size = mesh.geometry().size();
  //dolfin_debug1("sending geometry size %d", size);
  MPI_Bcast(&size, 1, MPI_UNSIGNED, this_process, MPI_COMM_WORLD);

  // Send dim
  uint dim = mesh.geometry().dim();
  //dolfin_debug1("sending geometry dim %d", dim);
  MPI_Bcast(&dim, 1, MPI_UNSIGNED, this_process, MPI_COMM_WORLD);

  // Send the coordinates
  const real* coordinates = mesh.coordinates(); 
  //dolfin_debug1("sending geometry %d coordinates", dim*size);
  MPI_Bcast(const_cast<real *>(coordinates), dim*size, MPI_DOUBLE, this_process, MPI_COMM_WORLD);

  // Mesh topology
  uint D = mesh.topology().dim();
  //dolfin_debug1("sending topology D %d", D);
  MPI_Bcast(&D, 1, MPI_UNSIGNED, this_process, MPI_COMM_WORLD);

  // Send num_entities
  uint* num_entities = mesh.topology().num_entities;
  //dolfin_debug1("sending %d num_entities", D+1);
  MPI_Bcast(num_entities, D+1, MPI_UNSIGNED, this_process, MPI_COMM_WORLD);
  
  // Send connectivity
  MeshConnectivity** connectivity = mesh.topology().connectivity;
  if ( D > 0 )
  {
    for (uint d0 = 0; d0 <= D; d0++)
      for (uint d1 = 0; d1 <= D; d1++)
      {
        MeshConnectivity mc = connectivity[d0][d1];
        // size
        MPI_Bcast(&mc._size, 1, MPI_UNSIGNED, this_process, MPI_COMM_WORLD);

        // num_entities
        MPI_Bcast(&mc.num_entities, 1, MPI_UNSIGNED, this_process, MPI_COMM_WORLD);
        
        // offsets
        MPI_Bcast(mc.offsets, mc.num_entities + 1, MPI_UNSIGNED, this_process, MPI_COMM_WORLD);
        
        // connections
        MPI_Bcast(mc.connections, mc._size, MPI_UNSIGNED, this_process, MPI_COMM_WORLD);
      }
  }

  // CellType
  int cell_type = mesh._cell_type->cell_type;
  int facet_type = mesh._cell_type->facet_type;
  //dolfin_debug1("Sending cell_type %d", cell_type);
  MPI_Bcast(&cell_type, 1, MPI_INT, this_process, MPI_COMM_WORLD);
  //dolfin_debug1("Sending facet_type %d", facet_type);
  MPI_Bcast(&facet_type, 1, MPI_INT, this_process, MPI_COMM_WORLD);

  dolfin_debug1("Finished mesh broadcast on process %d", this_process);
}
//-----------------------------------------------------------------------------
void MPIMeshCommunicator::receive(Mesh& mesh)
{
  mesh.clear();
  int process_int;
  MPI_Comm_rank(MPI_COMM_WORLD, &process_int);
  unsigned int this_process = process_int;

  // Define custom MPI datatype?
  //MPI_Datatype mpi_mesh;

  // Receiving number of coordinates 
  uint size = 0;
  MPI_Bcast(&size, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
  //dolfin_debug1("received geometry size %d", size);

  // Receiving dim
  uint dim = 0;
  MPI_Bcast(&dim, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
  //dolfin_debug1("received geometry dim %d", dim);

  // Receiving coordinates
  real* coordinates = new real[dim*size];
  MPI_Bcast(coordinates, dim*size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // Receiving topology
  // Receiving dim
  uint D = 0;
  MPI_Bcast(&D, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
  //dolfin_debug2("process num: %d received topology %d ", this_process, D);

  // Receiving num_entities
  uint* num_entities = new uint[D + 1];
  MPI_Bcast(num_entities, D+1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

  // Receive connectivity
  // Allocate data
  MeshConnectivity** c = new MeshConnectivity*[D + 1];
  for (uint d = 0; d <= D; d++)
    c[d] = new MeshConnectivity[D + 1];

  if ( D > 0 )
  {
    for (uint d0 = 0; d0 <= D; d0++)
      for (uint d1 = 0; d1 <= D; d1++)
      {
        // size
        MPI_Bcast(&c[d0][d1]._size, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

        // num_entities
        MPI_Bcast(&c[d0][d1].num_entities, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
        
        // offsets
        c[d0][d1].offsets = new uint[c[d0][d1].num_entities + 1];
        MPI_Bcast(c[d0][d1].offsets, c[d0][d1].num_entities + 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
        
        // connections
        c[d0][d1].connections = new uint[c[d0][d1]._size];
        MPI_Bcast(c[d0][d1].connections, c[d0][d1]._size, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
      }
  }

  // Receive CellType
  
  int cell_type, facet_type;
  MPI_Bcast(&cell_type, 1, MPI_INT, 0, MPI_COMM_WORLD);
  //dolfin_debug2("process num: %d received cell_type %d ", this_process, cell_type);
  MPI_Bcast(&facet_type, 1, MPI_INT, 0, MPI_COMM_WORLD);
  //dolfin_debug2("process num: %d received facet_type %d ", this_process, facet_type);

  // Updating mesh
  mesh.geometry()._size = size;
  mesh.geometry()._dim = dim;
  mesh.geometry().coordinates = coordinates;

  mesh.topology()._dim = D;
  mesh.topology().num_entities = num_entities;
  mesh.topology().connectivity = c;

  mesh._cell_type = CellType::create(CellType::Type(cell_type));
  mesh._cell_type->facet_type = CellType::Type(facet_type);
  
  dolfin_debug1("Finished mesh receive on process %d", this_process);
}
//-----------------------------------------------------------------------------
void MPIMeshCommunicator::broadcast(const MeshFunction<unsigned int>& mesh_function)
{
  dolfin_debug("MPIMeshCommunicator::broadcast");
  int process_int;
  MPI_Comm_rank(MPI_COMM_WORLD, &process_int);
  //unsigned int this_process = process_int;

  uint size = mesh_function._size;
  uint dim = mesh_function._dim;
  const uint* values = mesh_function.values();

  //dolfin_debug1("sending meshfunction size %d", size);
  MPI_Bcast(&size, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

  //dolfin_debug1("sending meshfunction dim %d", dim);
  MPI_Bcast(&dim, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
  MPI_Bcast(const_cast<uint *>(values), size, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
}
//-----------------------------------------------------------------------------
void MPIMeshCommunicator::receive(MeshFunction<unsigned int>& mesh_function)
{
  dolfin_debug("MPIMeshCommunicator::receive");
  int process_int;
  MPI_Comm_rank(MPI_COMM_WORLD, &process_int);
  //unsigned int this_process = process_int;

  uint size = 0;
  MPI_Bcast(&size, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
  //dolfin_debug1("received meshfunction size %d", size);

  uint dim = 0;
  MPI_Bcast(&dim, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
  //dolfin_debug1("received meshfunction dim %d", dim);

  if (mesh_function._values)
    delete [] mesh_function._values;
  uint* values = new uint[size];
  MPI_Bcast(values, size, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

  // Update mesh_function
  mesh_function._size = size;
  mesh_function._dim = dim;
  mesh_function._values = values;
}
//-----------------------------------------------------------------------------
void MPIMeshCommunicator::distribute(Mesh& mesh,
				     MeshFunction<uint>& distribution)
{
  distributeCommon(mesh, distribution, 0, 0);
}
//-----------------------------------------------------------------------------
  void MPIMeshCommunicator::distribute(Mesh& mesh, 
				       MeshFunction<uint>& distribution, 
				       MeshFunction<bool>& old_cell_marker,
				       MeshFunction<bool>& cell_marker) 
{
  distributeCommon(mesh, distribution, &old_cell_marker, &cell_marker);
}
//-----------------------------------------------------------------------------
  void MPIMeshCommunicator::distributeCommon(Mesh& mesh, 
				       MeshFunction<uint>& distribution, 
				       MeshFunction<bool>* old_cell_marker,
				       MeshFunction<bool>* cell_marker) 
{


  MeshFunction<uint>* global = mesh.data().meshFunction("vertex numbering");
  MeshFunction<uint>* ghost = mesh.data().meshFunction("ghosted vertices");
  
  std::map<uint, uint> tmp_local_map, tmp_global_map;

  uint rank = MPI::process_number();
  uint pe_size = MPI::num_processes();
  uint vi = 0;
  uint src, dest, k;

  Array<real> coords;
  Array<real> *send_buff_coords = new Array<real>[pe_size];
  Array<uint> *send_buff_map = new Array<uint>[pe_size];

  // Process distribution
  if(distribution.dim() == 0) {
    for(VertexIterator v(mesh); !v.end(); ++v) {
      if(!ghost->get(*v)) {
	if(distribution.get(*v) != rank) {
	  dest = distribution.get(*v);
	  send_buff_map[dest].push_back(global->get(*v));
	  send_buff_coords[dest].push_back(v->point().x());
	  send_buff_coords[dest].push_back(v->point().y());
	  if( mesh.geometry().dim() > 2 )
	    send_buff_coords[dest].push_back(v->point().z());	    
	}
	else {
	  coords.push_back(v->point().x());
	  coords.push_back(v->point().y());
	  if( mesh.geometry().dim() > 2 )
	    coords.push_back(v->point().z());
	  tmp_local_map[vi] = global->get(*v);
	  tmp_global_map[global->get(*v)] = vi++;
	}
      }
    }
  }
  else
    error("Not implemented"); //FIXME add code for cell based distribution

  // Exchange mesh entities
  int recv_count, recv_size, send_size;  
  for(uint i = 0; i < pe_size; i++) {
    send_size = send_buff_coords[i].size();
    MPI_Reduce(&send_size, &recv_count, 1, MPI_INT, MPI_MAX, i, MPI_COMM_WORLD);
  }
  
  uint recv_count_map = recv_count / mesh.geometry().dim();
  
  double *recv_buff = new double[recv_count];
  uint* recv_buff_map = new uint[recv_count_map];

  MPI_Status status;
  for(uint i = 1; i < pe_size; i++) {
    
    src = (rank - i + pe_size) % pe_size;
    dest = (rank + i) % pe_size;
    
    MPI_Sendrecv(&send_buff_map[dest][0], send_buff_map[dest].size(),
		 MPI_UNSIGNED, dest, 0, recv_buff_map, recv_count_map, 
		 MPI_UNSIGNED, src, 0, MPI_COMM_WORLD, &status);
    
    MPI_Sendrecv(&send_buff_coords[dest][0], send_buff_coords[dest].size(),
		 MPI_DOUBLE, dest, 1, recv_buff, recv_count, MPI_DOUBLE, src,
		 1, MPI_COMM_WORLD, &status);
    MPI_Get_count(&status,MPI_DOUBLE,&recv_size);
    
    k = 0;
    for(int j = 0; j < recv_size; j += mesh.geometry().dim()) {
      if(tmp_global_map.find(recv_buff_map[k]) == tmp_global_map.end()) {
	tmp_local_map[vi] = recv_buff_map[k]; 
	tmp_global_map[recv_buff_map[k]] = vi++;
	coords.push_back(recv_buff[j]); 
	coords.push_back(recv_buff[j+1]);
	if( mesh.geometry().dim() > 2 )
	   coords.push_back(recv_buff[j+2]);
	k++;
      }
    }
    //FIXME add code for cell based distribution
  }


  for(uint i=0; i<pe_size; i++) {
    send_buff_map[i].clear();
    send_buff_coords[i].clear();
  }

  delete[] send_buff_map;
  delete[] send_buff_coords;

  uint num_vertices = coords.size() / mesh.geometry().dim();
  
  Mesh new_mesh;
  MeshEditor editor;
  editor.open(new_mesh, mesh.type().cellType(),
	      mesh.topology().dim(), mesh.geometry().dim());


  editor.initVertices(num_vertices);
  editor.initCells(1);

  // Add vertices
  vi=0;
  for(uint i=0; i<coords.size(); i += mesh.geometry().dim(), vi++)
    switch( mesh.geometry().dim() ) {
    case 2:
      editor.addVertex(vi, coords[i], coords[i+1]); break;
    case 3:
      editor.addVertex(vi, coords[i], coords[i+1], coords[i+2]); break;
    }
  coords.clear();

  // FIXME add cells

  editor.close();

  // Overwrite old mesh
  mesh = new_mesh;


  // Recreate auxiliary mesh data.
  global = mesh.data().createMeshFunction("vertex numbering");
  std::map<uint, uint>* glb_map = mesh.data().createMap("global to local");
  global->init(mesh, 0);

  std::map<uint, uint>::iterator it;
  for(it = tmp_local_map.begin(); it != tmp_local_map.end(); ++it)
    global->set(it->first, it->second);

  for(it = tmp_global_map.begin(); it != tmp_global_map.end(); ++it)
    (*glb_map)[it->first] = it->second;

}
//-----------------------------------------------------------------------------
#else

//-----------------------------------------------------------------------------
MPIMeshCommunicator::MPIMeshCommunicator()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
MPIMeshCommunicator::~MPIMeshCommunicator()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void MPIMeshCommunicator::broadcast(const Mesh& mesh) 
{ 
  error("Cannot broadcast meshes without MPI.");
}
//-----------------------------------------------------------------------------
void MPIMeshCommunicator::receive(Mesh& mesh) 
{ 
  error("Cannot receive meshes without MPI.");
}
//-----------------------------------------------------------------------------
void MPIMeshCommunicator::broadcast(const MeshFunction<unsigned int>& mesh_function) 
{ 
  error("Cannot broadcast mesh functions without MPI.");
}
//-----------------------------------------------------------------------------
void MPIMeshCommunicator::receive(MeshFunction<unsigned int>& mesh_function) 
{ 
  error("Cannot receive mesh functions without MPI.");
}
//-----------------------------------------------------------------------------
void MPIMeshCommunicator::distribute(Mesh& mesh,
				     MeshFunction<uint>& distribution)
{
  error("Cannot distribute mesh without MPI.");
}
//-----------------------------------------------------------------------------
void MPIMeshCommunicator::distribute(Mesh& mesh, 
				     MeshFunction<uint>& distribution, 
				     MeshFunction<bool>& old_cell_marker,
				     MeshFunction<bool>& cell_marker) 
{
  error("Cannot distribute mesh without MPI.");
}
//-----------------------------------------------------------------------------

#endif
