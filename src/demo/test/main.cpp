// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <iostream>
#include <petsc/petsc.h>
#include <dolfin.h>

using namespace dolfin;

#define SIZE 50

int N;
int N2;
int N3;
double values[64];
int dofs[8];

void setdofs(int i, int j, int k, int dofs[])
{
  // Compute node numbers for all 8 vertices of the cube
  int pos = 0;
  for (int ii = 0; ii < 2; ii++)
    for (int jj = 0; jj < 2; jj++)
      for (int kk = 0; kk < 2; kk++)
	dofs[pos++] = N2*(k + kk) + N*(j + jj) + i + ii;
}

void testDOLFIN(double* t1, double* t2)
{
  std::cout << "Testing DOLFIN" << std::endl;
  tic();
  
  Matrix A(N3, N3, 27);

  for (int i = 0; i < (N-1); i++)
  {
    for (int j = 0; j < (N-1); j++)
    {
      for (int k = 0; k < (N-1); k++)
      {
	setdofs(i, j, k, dofs);
	
	int pos = 0;
	for (int ii = 0; ii < 8; ii++)
	  for (int jj = 0; jj < 8; jj++)
	    A(dofs[ii], dofs[jj]) += values[pos++];
      }
    }
  }

  *t1 = toc();
  std::cout << "Testing DOLFIN again" << std::endl;
  tic();
  
  A = 0.0;
  for (int i = 0; i < (N-1); i++)
  {
    for (int j = 0; j < (N-1); j++)
    {
      for (int k = 0; k < (N-1); k++)
      {
	setdofs(i, j, k, dofs);
	
	int pos = 0;
	for (int ii = 0; ii < 8; ii++)
	  for (int jj = 0; jj < 8; jj++)
	    A(dofs[ii], dofs[jj]) += values[pos++];
      }
    }
  }  

  *t2 = toc();
}

void testPETSc(double* t1, double* t2)
{
  std::cout << "Testing PETSc" << std::endl;
  tic();

  Mat A;
  MatCreateSeqAIJ(PETSC_COMM_SELF, N3, N3, 27, PETSC_NULL, &A);
  MatSetFromOptions(A);
  //MatSetOption(A, MAT_ROWS_SORTED);
  //MatSetOption(A, MAT_COLUMNS_SORTED);
  MatSetOption(A, MAT_USE_HASH_TABLE);
  
  for (int i = 0; i < (N-1); i++)
  {
    for (int j = 0; j < (N-1); j++)
    {
      for (int k = 0; k < (N-1); k++)
      {
	setdofs(i, j, k, dofs);
	
	MatSetValues(A, 8, dofs, 8, dofs, values, ADD_VALUES);
      }
    }
  }

  MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);

  *t1 = toc();
  std::cout << "Testing PETSc again" << std::endl;
  MatSetOption(A, MAT_NEW_NONZERO_LOCATION_ERR);
  tic();

  MatZeroEntries(A);
  for (int i = 0; i < (N-1); i++)
  {
    for (int j = 0; j < (N-1); j++)
    {
      for (int k = 0; k < (N-1); k++)
      {
	setdofs(i, j, k, dofs);
	
	MatSetValues(A, 8, dofs, 8, dofs, values, ADD_VALUES);
      }
    }
  }

  MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);

  *t2 = toc();
  
  MatDestroy(A);
}

int main(int argc, char** argv)
{
  dolfin_set("output", "plain text");
  PetscInitialize(&argc, &argv, 0, 0);

  // Set all values
  for (int i = 0; i < 64; i++)
    values[i] = 1.0;

  // Set problem size
  N = SIZE;
  N2 = N*N;
  N3 = N*N*N;
  
  std::cout << "System size: " << N3 << " x " << N3 << std::endl;

  double t1 = 0.0;
  double t2 = 0.0;
  double t3 = 0.0;
  double t4 = 0.0;

  testDOLFIN(&t1, &t2);
  testPETSc(&t3, &t4);
 
  std::cout << "DOLFIN assembly:    " << t1 << " s" << std::endl;
  std::cout << "DOLFIN re-assembly: " << t2 << " s" << std::endl;
  std::cout << "PETSc  assembly:    " << t3 << " s" << std::endl;
  std::cout << "PETSc  re-assembly: " << t4 << " s" << std::endl;

  PetscFinalize();
  return 0;
}
