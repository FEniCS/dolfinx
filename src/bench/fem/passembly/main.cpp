// Copyright (C) 2007 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Magnus Vikstrøm 
// First added:  2007-10-30
// Last changed: 2008-01-09
//
// This file is used for testing distribution of the mesh using MPI

#include <dolfin.h>
#include "pPoisson.h"
#include "Poisson.h"
#include "pNonlinear2D.h"
#include "Nonlinear2D.h"
#include <iostream>
#include <fstream>
#include "getopts.h"
#include "assemble.h"

using namespace dolfin;

real timer(Mesh& mesh, int num_iterations, Form& a)
{
  dolfin::cout << "Assembling with sequential assembler." << dolfin::endl;
  Matrix A;
  Assembler assembler(mesh);

  assembler.assemble(A, a, true);
  tic();
  for(int i=0; i<num_iterations; ++i)
    assembler.assemble(A, a, false);
  return toc()/static_cast<real>(num_iterations);
}

real p_timer(Mesh& mesh, MeshFunction<dolfin::uint>& partitions, int num_iterations, pForm& a)
{
  dolfin::cout << "Assembling with parallel assembler." << dolfin::endl;
  Matrix B;
  pAssembler passembler(mesh, partitions);

  passembler.assemble(B, a, true);
  tic();
  for(int i=0; i<num_iterations; ++i)
    passembler.assemble(B, a, false);
  return toc()/static_cast<real>(num_iterations);
}

int main(int argc, char* argv[])
{
  std::string meshfile, partitionsfile, resultfile;
  std::string assembler = "parallel";
  int cells = 400;
  int cells_3D = 40;
  int num_iterations = 1;
  int num_part = dolfin::MPI::numProcesses();
  int debug = -1;
  bool check = false;
  bool sequential = false;
  std::string testtype = "Poisson3D";
  bool petsc_info = false;

  Options listOpts;
  int switchInt;
  listOpts.addOption("", "mesh", "Mesh File", true);
  listOpts.addOption("", "partitions", "Mesh partitions file", true);
  listOpts.addOption("", "sequential", "Run with sequential assembler", false);
  listOpts.addOption("", "cells", "Number of cells", true);
  listOpts.addOption("", "num_part", "Number of partitions", true);
  listOpts.addOption("", "resultfile", "File to save results in", true);
  listOpts.addOption("", "num_iterations", "Number of times to assemble", true);
  listOpts.addOption("", "check", "Verify assembly result", false);
  listOpts.addOption("", "debug", "Prints debugging info", false);
  listOpts.addOption("", "testtype", "Type of test: Poisson2D, Poisson3D, Nonlinear2D", true);
  listOpts.addOption("", "petsc_info", "Print info from PETSc", false);

  if (listOpts.parse(argc, argv))
    while ((switchInt = listOpts.cycle()) >= 0)
    {
      switch(switchInt)
      {
        case 0:
          meshfile = listOpts.getArgs(switchInt);
          break;
        case 1:
          partitionsfile = listOpts.getArgs(switchInt);
          break;
        case 2:
          sequential = true;
          break;
        case 3:
          cells = atoi(listOpts.getArgs(switchInt).c_str());
          cells_3D = atoi(listOpts.getArgs(switchInt).c_str());
          break;
        case 4:
          num_part = atoi(listOpts.getArgs(switchInt).c_str());
          break;
        case 5:
          resultfile = listOpts.getArgs(switchInt);
          break;
        case 6:
          num_iterations = atoi(listOpts.getArgs(switchInt).c_str());
          break;
        case 7:
          check = true;
          break;
        case 8:
          debug = 1;
          break;
        case 9:
          testtype = listOpts.getArgs(switchInt);
          break;
        case 10:
          petsc_info = true;
          break;
        default:
          break;
      }
    }

  Mesh mesh;
  MeshFunction<dolfin::uint>* partitions;
  set("debug level", debug);
  if(petsc_info)
  {
    char** init_argv = new char*[2];
    init_argv[0] = argv[0];
    sprintf(init_argv[1], "%s", "-info");
    dolfin::cout << "dolfin_init" << dolfin::endl;
    dolfin_init(2, init_argv);
  }
  if(meshfile != "")
  {
    dolfin::cout << "Reading mesh from file: " << meshfile << dolfin::endl;
    mesh = Mesh(meshfile);
  }
  else
  {
    if(testtype == "Nonlinear2D" || testtype == "Poisson2D")
    {
      printf("Creating UnitSquare(%d, %d)\n",  cells, cells);
      mesh = UnitSquare(cells, cells);
    }
    else
    {
      printf("Creating UnitCube(%d, %d, %d)\n", cells_3D, cells_3D, cells_3D);
      mesh = UnitCube(cells_3D, cells_3D, cells_3D);
    }
  }
  real time = 0;
  if(sequential)
  {
    Form* a;
    if(testtype == "Poisson3D")
      a = new PoissonBilinearForm();
    else if(testtype == "Nonlinear2D")
    {
      Function w(mesh, 1.0);
      a = new Nonlinear2DBilinearForm(w);
    }
    else
      a = new PoissonBilinearForm();

    dolfin::cout << "Running test " << testtype << dolfin::endl;
    dolfin::cout << "Assembling with sequential assembler." << dolfin::endl;
    printf("Number of iteration(s): %d\n", num_iterations);
    time = timer(mesh, num_iterations, *a);
  }
  else
  {
    if(partitionsfile != "")
    {
      dolfin::cout << "Reading partitions from file: " << partitionsfile << dolfin::endl;
      partitions = new MeshFunction<dolfin::uint>(mesh, partitionsfile);
    }
    else
    {
      dolfin::cout << "Partitioning mesh into: " << num_part << " partitions" << dolfin::endl;
      partitions = new MeshFunction<dolfin::uint>(mesh);
      mesh.partition(num_part, *partitions);
    }

    pForm* a;
    if(testtype == "Poisson3D")
      a = new pPoissonBilinearForm();
    else if(testtype == "Nonlinear2D")
    {
      Function w(mesh, 1.0);
      a = new pNonlinear2DBilinearForm(w);
    }
    else
      a = new pPoissonBilinearForm();

    dolfin::cout << "Running test " << testtype << dolfin::endl;
    dolfin::cout << "Assembling with parallel assembler." << dolfin::endl;
    printf("Number of iteration(s): %d Number of partitions: %d\n", num_iterations, num_part);
    time = p_timer(mesh, *partitions, num_iterations, *a);
  }
  if(resultfile != "")
  {
    dolfin::cout << "Appending results to " << resultfile << dolfin::endl;
    std::ofstream outfile(resultfile.c_str(), std::ofstream::app);
    outfile << dolfin::MPI::numProcesses() << " " << time << std::endl;
    outfile.close();
  }
  else
  {
    printf("Average assemble time: %.3e\n", time);
  }
  if(check)
  {
    dolfin::cout << "Not implemented" << dolfin::endl;
    //check_assembly(mesh, *partitions);
  }
  return 0;
}
