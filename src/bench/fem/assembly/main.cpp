// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-11-19
// Last changed: 2007-01-15

#include <dolfin.h>

#include "AdvectionOperator_3D_3_FFC.h"
#include "AdvectionOperator_3D_3_FER.h"

using namespace dolfin;

// Compare result from two different bilinear forms
void verifyElementTensor(double block0[], double block1[], unsigned int n)
{
  double tol = 1e-10;

  for (unsigned int i = 0; i < n; i++)
  {
    for (unsigned int j = 0; j < n; j++)
    {
      if ( fabs(block0[i*n + j] - block1[i*n + j]) > tol )
      {
        dolfin_info("Block 0: %.15e Block 1: %.15e", block0[i*n + j], block1[i*n + j]);
        dolfin_error2("Verification of data failed. Results differ in entry (%d, %d).", i, j);
      }
    }
  }

  cout << "Data verified ok" << endl;
}

// Initialize connectivity (don't include in benchmark)
void initConnectivity(Mesh& mesh)
{
  dolfin_log(false);

  // This is a temporary fix. We need to get information from FFC about
  // which connectivity is needed for the mapping of nodes.

  // This is needed to for the mapping of nodes
  for (unsigned int i = 0; i < mesh.topology().dim(); i++)
    mesh.init(i);

  dolfin_log(true);
}

// Time evaluation of element tensor
double timeElementTensor(BilinearForm& a, double* block, AffineMap& map, std::string s)
{
  cout << "Timing evaluation of element tensor for " << s << endl;
  unsigned int M = 10000000;

  tic();
  for (unsigned int i = 0; i < M; i++)
  {
    a.eval(block, map, 1.0);
  }
  double t = toc();

  return t / static_cast<double>(M);
}

// Adaptive timing of assembly (run at least for TMIN)
double timeAssembly(BilinearForm& a, double* block, AffineMap& map, std::string s)
{
  cout << "Timing global assembly for " << s << endl;
  unsigned int M = 100;

  UnitCube mesh(8, 8, 8);
  initConnectivity(mesh);
  cout << mesh << endl;
  
  Matrix A;
  
  dolfin_log(false);
  tic();
  for (unsigned int i = 0; i < M; i++)
  {
    FEM::assemble(a, A, mesh);
  }
  double t = toc();
  dolfin_log(true);
  
  return t / static_cast<double>(M);
}

// Run benchmark for two different bilinear forms
void bench(BilinearForm& a0, BilinearForm& a1, std::string s0, std::string s1)
{
  // Prepare data structures
  unsigned int m0 = a0.test().spacedim();
  unsigned int n0 = a0.trial().spacedim();
  unsigned int m1 = a1.test().spacedim();
  unsigned int n1 = a1.trial().spacedim();
  if ( m0 != n0 || n0 != m1 || m1 != n1 )
    dolfin_error("Space dimensions don't match.");
  unsigned int n = m0;
  double* block0 = new double[n*n];
  double* block1 = new double[n*n];
  AffineMap map;
  map.det = 1.0;
  map.g00 = 1.0; map.g01 = 4.0; map.g02 = 7.0;
  map.g10 = 2.0; map.g11 = 5.0; map.g12 = 8.0;
  map.g20 = 3.0; map.g21 = 6.0; map.g22 = 9.0;
  double tl_0(0), tl_1(0), ta_0(0), ta_1(0);

  // Run test cases for local element tensor
  tl_0 = timeElementTensor(a0, block0, map, s0);
  tl_1 = timeElementTensor(a1, block1, map, s1);

  // Verify that we get the same results for both methods
  verifyElementTensor(block0, block1, n);

  // Run test cases for global assembly
  ta_0 = timeAssembly(a0, block0, map, s0);
  ta_1 = timeAssembly(a1, block1, map, s1);

  // Report results
  dolfin_info("");
  dolfin_info("%s, local tensor: %.3e s", s0.c_str(), tl_0);
  dolfin_info("%s, local tensor: %.3e s", s1.c_str(), tl_1);
  dolfin_info("%s, global assembly: %.3e s", s0.c_str(), ta_0);
  dolfin_info("%s, global assembly: %.3e s", s1.c_str(), ta_1);
  dolfin_info("");
  dolfin_info("Speedup, local tensor:    %.2f", tl_0/tl_1);
  dolfin_info("Speedup, global assembly: %.2f", ta_0/ta_1);
  dolfin_info("");

  delete [] block0;
  delete [] block1;
}

int main()
{
  Zero w;

  AdvectionOperator_3D_3_FFC::BilinearForm a_AdvectionOperator_3D_3_FFC(w);
  AdvectionOperator_3D_3_FER::BilinearForm a_AdvectionOperator_3D_3_FER(w);
  
  bench(a_AdvectionOperator_3D_3_FFC, a_AdvectionOperator_3D_3_FER,
        "AdvectionOperator_3D_3_FFC", "AdvectionOperator_3D_3_FER");

  return 0;
}
