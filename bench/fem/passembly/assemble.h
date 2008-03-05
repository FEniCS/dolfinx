#include <dolfin.h>
#include "Poisson.h"

using namespace dolfin;
//-----------------------------------------------------------------------------
std::string appendRank(std::string base, std::string ext)
{
  std::stringstream stream;
  stream << base << dolfin::MPI::processNumber() << "." << ext;
  return stream.str();
}

void assemble(Mesh& mesh, MeshFunction<dolfin::uint>& partitions, char* filename,
              char* plotname)
{
}

void check_assembly(Mesh& mesh, MeshFunction<dolfin::uint>& partitions)
{
  // Do normal assembly on process 1
  PoissonBilinearForm a;
  PoissonBilinearForm b;
  Matrix A, B;

  if(dolfin::MPI::numProcesses() == 1)
  {
    Assembler assembler(mesh);
    assembler.assemble(A, a, true);
  }
  // Parallel assembly on all procs
  {
    pAssembler passembler(mesh, partitions);
    passembler.assemble(B, b, true);
  }
  DofMapSet& dof_map_set = b.dofMaps();

  // Would be nice to have automatic testing of B = A * modified dofs
  // Currently just printing so that matrices can be manually inspected
  PetscViewer viewer_a(PETSC_VIEWER_STDOUT_SELF);
  PetscViewerASCIIOpen(PETSC_COMM_WORLD, "A.m", &viewer_a);
  //PetscViewerSetFormat(viewer_a, PETSC_VIEWER_ASCII_MATLAB);
  MatView(A.mat().mat(), viewer_a);

  dolfin::cout << "Mapping: " << dolfin::endl;
  std::map<dolfin::uint, dolfin::uint> map = dof_map_set[0].getMap();
  for(dolfin::uint i=0; i<map.size(); ++i)
  {
    dolfin::cout << i << " => " << map[i] << dolfin::endl;
  }
  PetscViewer viewer(PETSC_VIEWER_STDOUT_WORLD);
  PetscViewerASCIIOpen(PETSC_COMM_WORLD, "B.m", &viewer);
  //PetscViewerSetFormat(viewer, PETSC_VIEWER_ASCII_MATLAB);
  MatView(B.mat().mat(), viewer);
}

