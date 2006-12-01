// Copyright (C) 2006 Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-12-01
// Last changed: 
//
// This file is used for testing parallel assembly

#include <dolfin.h>
#include <dolfin/Poisson2D.h>
#include <parmetis.h>
extern "C"
{
  #include <metis.h>
}

using namespace dolfin;

//-----------------------------------------------------------------------------
MeshFunction<dolfin::uint> testMeshPartition(Mesh& mesh, int num_partitions)
{
  int num_cells     = mesh.numCells() ;
  int num_vertices  = mesh.numVertices();
  
  int index_base = 0;  // zero-based indexing
  int edges_cut  = 0;

  int cell_type = 0;
  idxtype* cell_partition   = new int[num_cells];
  idxtype* vertex_partition = new int[num_vertices];
  idxtype* mesh_data = 0;

  // Set cell type and allocate memory for METIS mesh structure
  if(mesh.type().cellType() == CellType::triangle)
  {
    cell_type = 1;
    mesh_data = new int[3*num_cells];
  }
  else if(mesh.type().cellType() == CellType::tetrahedron) 
  {
    cell_type = 2;
    mesh_data = new int[4*num_cells];
  }
  else
    dolfin_error("Do not know how to partition mesh of this type");
  
  // Create mesh structure for METIS
  dolfin::uint i = 0;
  for (CellIterator cell(mesh); !cell.end(); ++cell)
    for (VertexIterator vertex(cell); !vertex.end(); ++vertex)
      mesh_data[i++] = vertex->index();


  // Use METIS to partition mesh
  METIS_PartMeshNodal(&num_cells, &num_vertices, mesh_data, &cell_type, &index_base, 
                      &num_partitions, &edges_cut, cell_partition, vertex_partition);

  cout << "Output partition data " << endl;
  cout << "  Edges cut " << edges_cut << endl;

  // Create mesh function for partition numbers
  MeshFunction<dolfin::uint> partition(mesh);
  partition.init(mesh.topology().dim());

  // Set partition numbers
  i = 0;
  for (CellIterator cell(mesh); !cell.end(); ++cell)
    partition.set(cell->index(), cell_partition[i++]);

  // Clean up
  delete [] cell_partition;
  delete [] vertex_partition;
  delete [] mesh_data;

  return partition;
}
//-----------------------------------------------------------------------------
int main(int argc, char* argv[])
{
  // Initialise PETSc  
  PETScManager::init();

  // Get number of processes
  int num_processes;
  MPI_Comm_size(PETSC_COMM_WORLD, &num_processes);

  // Get this process number
  int process;
  MPI_Comm_rank(PETSC_COMM_WORLD, &process);

  // Create mesh
  UnitSquare mesh(30,30);

  // Create bilinear form
  Poisson2D::BilinearForm a; 

  if(num_processes < 2 )
    dolfin_error("Cannot create single partition. You need to run woth mpirun -np X . . . ");

  // Partition mesh (number of partitions = number of processes)
  MeshFunction<dolfin::uint> partition = testMeshPartition(mesh, num_processes);

  // Start assembly
  
  // Initialize connectivity
  for(dolfin::uint i = 0; i < mesh.topology().dim(); i++)
    mesh.init(i);

  // Create affine map
  AffineMap map;

  // Global matrix size
  const int N = FEM::size(mesh, a.test());

  // Create PETSc matrix
  Mat A;
  MatCreate(PETSC_COMM_WORLD, &A);
  MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, N, N);   
  MatSetType(A, MATMPIAIJ);

  real* block = 0;
  int*  pos  = 0;
  int vertices_per_cell = 0;
  if(mesh.type().cellType() == CellType::triangle)
    vertices_per_cell = 3;
  else if(mesh.type().cellType() == CellType::tetrahedron) 
    vertices_per_cell = 4;
  else
    dolfin_error("Do not know how to work with meshes of this type");

  block = new real[vertices_per_cell*vertices_per_cell];
  pos   = new int[vertices_per_cell];

  for(int i =0; i<vertices_per_cell*vertices_per_cell; ++i)
      block[i] = 1.0; 

  // Zero matrix
  MatZeroEntries(A);

  // Assemble if cell belongs to this process's partition
  for(CellIterator cell(mesh); !cell.end(); ++cell)
  {
    if(partition.get(*cell) == process )
    {
//      cout << "Assembling cell " << cell->index() << " on process " << process << endl;
      map.update(*cell);

      // Update form
      a.update(map);

      // Create mapping for cell
      int i = 0;
      for(VertexIterator vertex(cell); !vertex.end(); ++vertex)
        pos[i++] = vertex->index();

      // Evaluate element matrix
      a.eval(block, map);

      MatSetValues(A, vertices_per_cell, pos, vertices_per_cell, pos, block, ADD_VALUES);
    }
  }  
  
  // Finalise
  MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);

  delete [] block;
  delete [] pos;

  // Verify assembly.
  real matrix_norm;
  MatNorm(A, NORM_FROBENIUS, &matrix_norm);

  if(process == 0)
  {
    // Compute Frobenius norm of paralle matrix and reference matrix
    std::cout << "Norm (parallel assembly) " << matrix_norm << std::endl;

  // Compute reference norm
    dolfin_log(false);
    PETScMatrix Aref; 
    FEM::assemble(a, Aref, mesh);
    dolfin_log(true);
    std::cout << "Norm (reference matrix) " << Aref.norm(PETScMatrix::frobenius) << std::endl;

  }  

  return 0;
}
