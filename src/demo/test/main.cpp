// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <iostream>
#include <petsc/petsc.h>
#include <dolfin.h>

using namespace dolfin;

real testDOLFIN1(Mesh& mesh)
{
  // Simplest version of assembly in DOLFIN
  std::cout << "Testing DOLFIN 1" << std::endl;
  tic();
  
  Matrix A(mesh.noNodes(), mesh.noNodes());
  
  for (CellIterator cell(mesh); !cell.end(); ++cell)
    for (NodeIterator n0(cell); !n0.end(); ++n0)
      for (NodeIterator n1(cell); !n1.end(); ++n1)
	A(n0->id(), n1->id()) += 1.0;

  return toc();
}

real testDOLFIN2(Mesh& mesh)
{
  // This version does things the same way as we do for PETSc
  // so the comparison is fair
  std::cout << "Testing DOLFIN 2" << std::endl;
  tic();
  
  Matrix A(mesh.noNodes(), mesh.noNodes());
  int dofs[4];
  double values[16];

  for (int i = 0; i < 16; i++)
    values[i] = 1.0;

  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    int pos = 0;
    for (NodeIterator n0(cell); !n0.end(); ++n0)
      dofs[pos++] = n0->id();

    pos = 0;
    for (int i = 0; i < 4; i++)
      for (int j = 0; j < 4; j++)
	A(dofs[i], dofs[j]) += values[pos++];
  }

  return toc();
}

real testPETSc1(Mesh& mesh)
{
  // Simplest version of assembly in PETSc
  std::cout << "Testing PETSc 1" << std::endl;
  tic();

  Mat A;
  int dofs[4];
  double values[16];

  MatCreate(PETSC_COMM_SELF, PETSC_DECIDE, PETSC_DECIDE,
	    mesh.noNodes(), mesh.noNodes(), &A);
  
  MatSetFromOptions(A);

  for (int i = 0; i < 16; i++)
    values[i] = 1.0;

  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    int pos = 0;
    for (NodeIterator n0(cell); !n0.end(); ++n0)
      dofs[pos++] = n0->id();
    
    MatSetValues(A, 4, dofs, 4, dofs, values, ADD_VALUES);
  }

  MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);
  MatDestroy(A);

  return toc();
}

real testPETSc2(Mesh& mesh)
{
  // Tell PETSc how many nonzeros we have
  std::cout << "Testing PETSc 2" << std::endl;
  tic();

  Mat A;
  int dofs[4];
  double values[16];

  MatCreateSeqAIJ(PETSC_COMM_SELF, mesh.noNodes(), mesh.noNodes(),
		  16, PETSC_NULL, &A);
  MatSetFromOptions(A);

  for (int i = 0; i < 16; i++)
    values[i] = 1.0;

  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    int pos = 0;
    for (NodeIterator n0(cell); !n0.end(); ++n0)
      dofs[pos++] = n0->id();
    
    MatSetValues(A, 4, dofs, 4, dofs, values, ADD_VALUES);
  }

  MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);
  MatDestroy(A);

  return toc();
}

int main(int argc, char** argv)
{
  dolfin_set("output", "plain text");
  PetscInitialize(&argc, &argv, 0, 0);

  Mesh mesh("mesh.xml.gz");
  mesh.refineUniformly();
  mesh.refineUniformly();

  real t1 = testDOLFIN1(mesh);
  real t2 = testDOLFIN2(mesh);
  real t3 = testPETSc1(mesh);
  real t4 = testPETSc2(mesh);
 
  std::cout << "DOLFIN 1: " << t1 << " s" << std::endl;
  std::cout << "DOLFIN 2: " << t2 << " s" << std::endl;
  std::cout << "PETSc  1: " << t3 << " s" << std::endl;
  std::cout << "PETSc  2: " << t4 << " s" << std::endl;

  PetscFinalize();
  return 0;
}
