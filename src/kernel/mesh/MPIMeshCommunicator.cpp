// Copyright (C) 2007 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
//  Modified by Magnus Vikstrøm, 2007.
//
// First added:  2007-05-30
// Last changed: 2007-12-01

#include <dolfin/dolfin_log.h>
#include <dolfin/Mesh.h>
#include <dolfin/MeshFunction.h>
#include <dolfin/MPIMeshCommunicator.h>
#include <dolfin/MPIManager.h>

using namespace dolfin;

#ifdef HAVE_MPI_H
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
  dolfin_debug1("sending geometry size %d", size);
  MPI_Bcast(&size, 1, MPI_UNSIGNED, this_process, MPI_COMM_WORLD);

  // Send dim
  uint dim = mesh.geometry().dim();
  dolfin_debug1("sending geometry dim %d", dim);
  MPI_Bcast(&dim, 1, MPI_UNSIGNED, this_process, MPI_COMM_WORLD);

  // Send the coordinates
  const real* coordinates = mesh.coordinates(); 
  dolfin_debug1("sending geometry %d coordinates", dim*size);
  MPI_Bcast(const_cast<real *>(coordinates), dim*size, MPI_DOUBLE, this_process, MPI_COMM_WORLD);

  // Mesh topology
  uint D = mesh.topology().dim();
  dolfin_debug1("sending topology D %d", D);
  MPI_Bcast(&D, 1, MPI_UNSIGNED, this_process, MPI_COMM_WORLD);

  // Send num_entities
  uint* num_entities = mesh.topology().num_entities;
  dolfin_debug1("sending %d num_entities", D+1);
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
  int cell_type = mesh.data.cell_type->cell_type;
  int facet_type = mesh.data.cell_type->facet_type;
  dolfin_debug1("Sending cell_type %d", cell_type);
  MPI_Bcast(&cell_type, 1, MPI_INT, this_process, MPI_COMM_WORLD);
  dolfin_debug1("Sending facet_type %d", facet_type);
  MPI_Bcast(&facet_type, 1, MPI_INT, this_process, MPI_COMM_WORLD);

  dolfin_debug1("Finished mesh broadcast on process %d", this_process);
}
//-----------------------------------------------------------------------------
void MPIMeshCommunicator::receive(Mesh& mesh)
{
  mesh.data.clear();
  int process_int;
  MPI_Comm_rank(MPI_COMM_WORLD, &process_int);
  unsigned int this_process = process_int;

  // Define custom MPI datatype?
  //MPI_Datatype mpi_mesh;

  // Receiving number of coordinates 
  uint size = 0;
  MPI_Bcast(&size, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
  dolfin_debug1("received geometry size %d", size);

  // Receiving dim
  uint dim = 0;
  MPI_Bcast(&dim, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
  dolfin_debug1("received geometry dim %d", dim);

  // Receiving coordinates
  real* coordinates = new real[dim*size];
  MPI_Bcast(coordinates, dim*size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // Receiving topology
  // Receiving dim
  uint D = 0;
  MPI_Bcast(&D, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
  dolfin_debug2("process num: %d received topology %d ", this_process, D);

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
  dolfin_debug2("process num: %d received cell_type %d ", this_process, cell_type);
  MPI_Bcast(&facet_type, 1, MPI_INT, 0, MPI_COMM_WORLD);
  dolfin_debug2("process num: %d received facet_type %d ", this_process, facet_type);

  // Updating mesh
  mesh.geometry()._size = size;
  mesh.geometry()._dim = dim;
  mesh.geometry().coordinates = coordinates;

  mesh.topology()._dim = D;
  mesh.topology().num_entities = num_entities;
  mesh.topology().connectivity = c;

  mesh.data.cell_type = CellType::create(CellType::Type(cell_type));
  mesh.data.cell_type->facet_type = CellType::Type(facet_type);
  
  dolfin_debug1("Finished mesh receive on process %d", this_process);
}
//-----------------------------------------------------------------------------
void MPIMeshCommunicator::broadcast(const MeshFunction<unsigned int>& mesh_function)
{
  dolfin_debug("MPIMeshCommunicator::broadcast");
  int process_int;
  MPI_Comm_rank(MPI_COMM_WORLD, &process_int);
  unsigned int this_process = process_int;

  uint size = mesh_function._size;
  uint dim = mesh_function._dim;
  const uint* values = mesh_function.values();
  dolfin_debug1("sending meshfunction size %d", size);
  MPI_Bcast(&size, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
  dolfin_debug1("sending meshfunction dim %d", dim);
  MPI_Bcast(&dim, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
  MPI_Bcast(const_cast<uint *>(values), size, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
}
//-----------------------------------------------------------------------------
void MPIMeshCommunicator::receive(MeshFunction<unsigned int>& mesh_function)
{
  dolfin_debug("MPIMeshCommunicator::receive");
  int process_int;
  MPI_Comm_rank(MPI_COMM_WORLD, &process_int);
  unsigned int this_process = process_int;

  uint size = 0;
  MPI_Bcast(&size, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
  dolfin_debug1("received meshfunction size %d", size);

  uint dim = 0;
  MPI_Bcast(&dim, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
  dolfin_debug1("received meshfunction dim %d", dim);

  uint* values = new uint[size];
  MPI_Bcast(values, size, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

  // Update mesh_function
  mesh_function._size = size;
  mesh_function._dim = dim;
  mesh_function._values = values;
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
  // Do nothing
}
//-----------------------------------------------------------------------------
void MPIMeshCommunicator::receive(Mesh& mesh) 
{ 
  // Do nothing
}
//-----------------------------------------------------------------------------
void MPIMeshCommunicator::broadcast(const MeshFunction<unsigned int>& mesh_function) 
{ 
  // Do nothing
}
//-----------------------------------------------------------------------------
void MPIMeshCommunicator::receive(MeshFunction<unsigned int>& mesh_function) 
{ 
  // Do nothing
}
//-----------------------------------------------------------------------------

#endif
