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
void testMeshPartition(Mesh& mesh, MeshFunction<dolfin::uint>& cell_partition_function,
          MeshFunction<dolfin::uint>& vertex_partition_function, int num_partitions)
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
  
  cell_partition_function.init(mesh, mesh.topology().dim());
  vertex_partition_function.init(mesh, 0);

  if(num_partitions > 1)
  {
    // Create mesh structure for METIS
    dolfin::uint i = 0;
    for (CellIterator cell(mesh); !cell.end(); ++cell)
      for (VertexIterator vertex(cell); !vertex.end(); ++vertex)
        mesh_data[i++] = vertex->index();

      // Use METIS to partition mesh
    METIS_PartMeshNodal(&num_cells, &num_vertices, mesh_data, &cell_type, &index_base, 
                        &num_partitions, &edges_cut, cell_partition, vertex_partition);
  
    // Set partition numbers on cells
    i = 0;
    for (CellIterator cell(mesh); !cell.end(); ++cell)
      cell_partition_function.set(cell->index(), cell_partition[i++]);

    // Set partition numbers on vertexes
    i = 0;
    for (VertexIterator vertex(mesh); !vertex.end(); ++vertex)
      vertex_partition_function.set(vertex->index(), vertex_partition[i++]);
  }
  else
  {
    // Set partition numbers on cells
    dolfin::uint i = 0;
    for (CellIterator cell(mesh); !cell.end(); ++cell)
      cell_partition_function.set(cell->index(), 0);

    // Set partition numbers on vertexes
    i = 0;
    for (VertexIterator vertex(mesh); !vertex.end(); ++vertex)
      vertex_partition_function.set(vertex->index(), 0);
  
  }
  // Clean up
  delete [] cell_partition;
  delete [] vertex_partition;
  delete [] mesh_data;
}
//-----------------------------------------------------------------------------
int main(int argc, char* argv[])
{
  cout << "Starting parallel assemble/solve test." << endl;    

  // Initialise PETSc  
  PETScManager::init(argc, argv);

  // Get number of processes
  int num_processes_int;
  MPI_Comm_size(PETSC_COMM_WORLD, &num_processes_int);
  unsigned int num_processes = num_processes_int;

  // Get this process number
  int process_int;
  MPI_Comm_rank(PETSC_COMM_WORLD, &process_int);
  unsigned int process = process_int;

  // Create mesh
  UnitSquare mesh(2000,2000);

  // Create linear and bilinear form
  Function f = 1.0;
  Poisson2D::BilinearForm a; 
  Poisson2D::LinearForm L(f); 

/*  
  if ( num_processes < 2 )
    dolfin_error("Cannot create single partition. You need to run with \"mpirun -np num_proc ./dolfin-parallel-test\"\
 (num_proc > 1)");
*/
  // Partition mesh (number of partitions = number of processes)
  // Create mesh functions for partition numbers
  MeshFunction<dolfin::uint> cell_partition_function;
  MeshFunction<dolfin::uint> vertex_partition_function;
  cout << "Partitioning mesh " << process << endl;
  testMeshPartition(mesh, cell_partition_function, vertex_partition_function,
                    num_processes);
  cout << "Finished partitioning mesh " << process << endl;

  // Need to regenerate degree of freedom mapping here so that matrix/vector entries
  // generated on a given processor also reside on the processor. DOFs for partition 0
  // should run 0 -> m-1, for partition 1 run m -> m+n-1, etc. 
  
  int vertices_per_cell = 0;
  if(mesh.type().cellType() == CellType::triangle)
    vertices_per_cell = 3;
  else if(mesh.type().cellType() == CellType::tetrahedron) 
    vertices_per_cell = 4;
  else
    dolfin_error("Do not know how to work with meshes of this type");

  // Renumber degrees of freedom. Starting at process 0, go through all cells
  // on given process, then the vertices of the cell and number sequentially 
  // if they have not already been renumbered.
  const int M = mesh.numVertices(); 
  int*  new_map = new int[M];


  cout << "Renumbering dofs " << process << endl;
/*
  int i = 0;
  std::set<int> remapped_dof;
  std::pair<std::set<int>::const_iterator, bool> set_return;
  for(dolfin::uint process = 0; process < num_processes; ++process)
    for(CellIterator cell(mesh); !cell.end(); ++cell)
      if(cell_partition_function.get(*cell) == static_cast<unsigned int>(process) )
        for(VertexIterator vertex(cell); !vertex.end(); ++vertex)
        {  
          set_return = remapped_dof.insert(vertex->index());
          if(set_return.second) // If insertion is successful, renumber. Otherwise dof has already been renumbered
            new_map[ vertex->index() ] = i++;
        }
  remapped_dof.clear();
*/
  // No renumbering
  for(VertexIterator vertex(mesh); !vertex.end(); ++vertex)
    new_map[ vertex->index() ] = vertex->index();
  
  cout << "Finished renumbering dofs " << process << endl;

  // Global matrix size
  cout << "Computing global matrix size " << process << endl;
//  const int N = FEM::size(mesh, a.test());
  const int N = mesh.numVertices();
  cout << "Finsihed computing global matrix size " << N << "  " << process << endl;

  cout << "Computing local matrix size " << process << endl;
  // Compute number of vertices belonging to this processor 
  int num_local_vertices = 0;
  for (VertexIterator vertex(mesh); !vertex.end(); ++vertex)
    if ( vertex_partition_function.get(*vertex) == process )
      ++num_local_vertices;
  cout << "Finished computing local matrix size " << process << endl;

  cout << "Creating and intialising parallel vectors/matrices " << process << endl;
  // Create PETSc parallel vectors
  Vec b, x;
  VecCreateMPI(PETSC_COMM_WORLD, num_local_vertices, N, &b);
  VecCreateMPI(PETSC_COMM_WORLD, num_local_vertices, N, &x);

  // Create PETSc parallel  matrix (with guess at number of non-zeroes)
  Mat A;
  MatCreateMPIAIJ(PETSC_COMM_WORLD, num_local_vertices, num_local_vertices, N, N, 
                    10, PETSC_NULL, 10, PETSC_NULL, &A); 
  cout << "Finished creating and intialising parallel vectors/matrices " << process << endl;
 
  // Zero matrix
  cout << "Zeroing parallel vectors/matrices " << process << endl;
  MatZeroEntries(A);
  VecZeroEntries(b);
  cout << "Finished zeroing parallel vectors/matrices " << process << endl;


  /// Start assembly
  real* A_block = new real[vertices_per_cell*vertices_per_cell];
  real* b_block = new real[vertices_per_cell];
  int*  pos     = new int[vertices_per_cell];

  // Create affine map
  AffineMap map;

  cout << "Starting assembly " << process << endl;
  tic();
  // Assemble if cell belongs to this process's partition
  for(CellIterator cell(mesh); !cell.end(); ++cell)
  {
    if(cell_partition_function.get(*cell) == static_cast<unsigned int>(process) )
    {
      map.update(*cell);
      a.update(map);
//      L.update(map);

      // Create mapping for cell
      int i = 0;
      for(VertexIterator vertex(cell); !vertex.end(); ++vertex)
        pos[i++] = new_map[ vertex->index() ];

      // Evaluate element matrix and vector
      a.eval(A_block, map);
//      L.eval(b_block, map);

      // Add values
      MatSetValues(A, vertices_per_cell, pos, vertices_per_cell, pos, A_block, ADD_VALUES);
//      VecSetValues(b, vertices_per_cell, pos, b_block, ADD_VALUES);
    }
  }  
  cout << "Finished assembly " << process << "  " << toc() << endl;
  
  // Finalise assembly
  cout << "Starting finalise assmebly " << process << endl;
  tic();
//  VecAssemblyBegin(b);
//  VecAssemblyEnd(b);
  MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
  cout << "Finished finalise assembly " << process << "  " << toc() << endl;

  // Apply some boundary conditions so that the system can be solved
  // Just apply homogeneous Dirichlet bc to the first three vertices
/*
  IS is = 0;
  int nrows = 3;
//  int rows[3] = {0, 1, 2};
  int rows[3] = {new_map[0], new_map[1], new_map[2]};
  ISCreateGeneral(PETSC_COMM_WORLD, nrows, rows, &is);
  PetscScalar one = 1.0;
  MatZeroRowsIS(A, is, one);
  ISDestroy(is);


  real bc_values[3] = {0, 0, 0};
  VecSetValues(b, nrows, rows, bc_values, INSERT_VALUES);
  VecAssemblyBegin(b);
  VecAssemblyEnd(b);

  // Solve linear system system
  KSP ksp;
  KSPCreate(PETSC_COMM_WORLD, &ksp);
  KSPSetFromOptions(ksp);
  KSPSetOperators(ksp, A, A, SAME_NONZERO_PATTERN);

  cout << "Starting solve " << process << endl;
  KSPSolve(ksp, b, x);
  cout << "Finished solve " << process << endl;
*/
  // Print parallel  results
//  cout << "Parallel RHS vector " << endl;
//  VecView(b, PETSC_VIEWER_STDOUT_WORLD);
//  cout << "Parallel matrix " << endl;
//  MatView(A, PETSC_VIEWER_STDOUT_WORLD);

/*
  if(process == 0)
    cout << "*** Parallel solution vector " << endl;
  VecView(x, PETSC_VIEWER_STDOUT_WORLD);
*/

  delete [] A_block;
  delete [] b_block;
  delete [] pos;
  delete [] new_map;

/*
  if(process == 0)
  {
    dolfin_log(false);
    PETScMatrix Aref;
    PETScVector bref, xref;
    FEM::assemble(a, L, Aref, bref, mesh); 
    Aref.ident(rows, nrows);
    bref(0) = 0.0; bref(1) = 0.0; bref(2) = 0.0;

    KrylovSolver solver;
    solver.solve(Aref, xref, bref);
    dolfin_log(true);

  // Print reference results
//    cout << "Single process reference vector " << endl;
//    VecView(bref.vec(), PETSC_VIEWER_STDOUT_SELF);
//    cout << "Single process reference matrix " << endl;
//    MatView(Aref.mat(), PETSC_VIEWER_STDOUT_SELF);
    cout << "*** Single process solution vector " << endl;
    VecView(xref.vec(), PETSC_VIEWER_STDOUT_SELF);

    cout << "Finsished parallel assemble/solve test." << endl;    
  }
*/
  return 0;
}
