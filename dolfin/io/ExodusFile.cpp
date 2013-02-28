// Copyright (C) 2013 Nico Schlömer
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2013-02-27

#ifdef HAS_VTK && HAS_VTK_EXODUS

#include <dolfin/fem/GenericDofMap.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/la/GenericVector.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshFunction.h>
#include "ExodusFile.h"

#include <vtkUnsignedIntArray.h>
#include <vtkIntArray.h>
#include <vtkDoubleArray.h>
#include <vtkCellType.h>
#include <vtkPointData.h>
#include <vtkCellData.h>
#include <vtkCellArray.h>
#include <vtkIdTypeArray.h>
#include <vtkUnstructuredGrid.h>
#include <vtkExodusIIWriter.h>

using namespace dolfin;

//----------------------------------------------------------------------------
ExodusFile::ExodusFile(const std::string filename)
  : GenericFile(filename, "Exodus")
{
}
//----------------------------------------------------------------------------
ExodusFile::~ExodusFile()
{
  // Do nothing
}
//----------------------------------------------------------------------------
void ExodusFile::operator<<(const Mesh& mesh)
{
  perform_write(create_vtk_mesh(mesh));
  log(TRACE, "Saved mesh %s (%s) to file %s in Exodus format.",
      mesh.name().c_str(), mesh.label().c_str(), filename.c_str());
  return;
}
//----------------------------------------------------------------------------
void ExodusFile::operator<<(const MeshFunction<unsigned int>& meshfunction)
{
  const Mesh& mesh = meshfunction.mesh();
  const uint cell_dim = meshfunction.dim();

  // Throw error for MeshFunctions on vertices for interval elements
  if (mesh.topology().dim() == 1 && cell_dim == 0)
  {
    dolfin_error("ExodusFile.cpp",
                 "write mesh function to Exodus file",
                 "Exodus output of mesh functions on interval facets is not supported");
  }

  if (cell_dim != mesh.topology().dim() && cell_dim != mesh.topology().dim() - 1)
  {
    dolfin_error("ExodusFile.cpp",
                 "write mesh function to Exodus file",
                 "Exodus output of mesh functions is implemented for cell- and facet-based functions only");
  }

  // Create Exodus mesh.
  vtkSmartPointer<vtkUnstructuredGrid> vtk_mesh = create_vtk_mesh(mesh);

  // Add cell data.
  const int dim = meshfunction.dim();
  const int numCells = mesh.num_cells();
  vtkSmartPointer<vtkUnsignedIntArray> cellData =
    vtkSmartPointer<vtkUnsignedIntArray>::New();
  cellData->SetNumberOfComponents(dim);
  cellData->SetArray(const_cast<unsigned int*>(meshfunction.values()), dim*numCells, 1);
  cellData->SetName(meshfunction.name().c_str());
  vtk_mesh->GetCellData()->AddArray(cellData);

  // Write out.
  perform_write(vtk_mesh);

  log(TRACE, "Saved mesh function %s (%s) to file %s in Exodus format.",
      mesh.name().c_str(), mesh.label().c_str(), filename.c_str());
}
//----------------------------------------------------------------------------
void ExodusFile::operator<<(const MeshFunction<int>& meshfunction)
{
  const Mesh& mesh = meshfunction.mesh();
  const uint cell_dim = meshfunction.dim();

  // Throw error for MeshFunctions on vertices for interval elements
  if (mesh.topology().dim() == 1 && cell_dim == 0)
  {
    dolfin_error("ExodusFile.cpp",
                 "write mesh function to Exodus file",
                 "Exodus output of mesh functions on interval facets is not supported");
  }

  if (cell_dim != mesh.topology().dim() && cell_dim != mesh.topology().dim() - 1)
  {
    dolfin_error("ExodusFile.cpp",
                 "write mesh function to Exodus file",
                 "Exodus output of mesh functions is implemented for cell- and facet-based functions only");
  }

  // Create Exodus mesh.
  vtkSmartPointer<vtkUnstructuredGrid> vtk_mesh = create_vtk_mesh(mesh);

  // Add cell data.
  const int dim = meshfunction.dim();
  const int numCells = mesh.num_cells();
  vtkSmartPointer<vtkIntArray> cellData =
    vtkSmartPointer<vtkIntArray>::New();
  cellData->SetNumberOfComponents(dim);
  cellData->SetArray(const_cast<int*>(meshfunction.values()), dim*numCells, 1);
  cellData->SetName(meshfunction.name().c_str());
  vtk_mesh->GetCellData()->AddArray(cellData);

  // Write out.
  perform_write(vtk_mesh);

  log(TRACE, "Saved mesh function %s (%s) to file %s in Exodus format.",
      mesh.name().c_str(), mesh.label().c_str(), filename.c_str());
}
//----------------------------------------------------------------------------
void ExodusFile::operator<<(const MeshFunction<double>& meshfunction)
{
  const Mesh& mesh = meshfunction.mesh();
  const uint cell_dim = meshfunction.dim();

  // Throw error for MeshFunctions on vertices for interval elements
  if (mesh.topology().dim() == 1 && cell_dim == 0)
  {
    dolfin_error("ExodusFile.cpp",
                 "write mesh function to Exodus file",
                 "Exodus output of mesh functions on interval facets is not supported");
  }

  if (cell_dim != mesh.topology().dim() && cell_dim != mesh.topology().dim() - 1)
  {
    dolfin_error("ExodusFile.cpp",
                 "write mesh function to Exodus file",
                 "Exodus output of mesh functions is implemented for cell- and facet-based functions only");
  }

  // Create Exodus mesh.
  vtkSmartPointer<vtkUnstructuredGrid> vtk_mesh = create_vtk_mesh(mesh);

  // Add cell data.
  const int dim = meshfunction.dim();
  const int numCells = mesh.num_cells();
  vtkSmartPointer<vtkDoubleArray> cellData =
    vtkSmartPointer<vtkDoubleArray>::New();
  cellData->SetNumberOfComponents(dim);
  cellData->SetArray(const_cast<double*>(meshfunction.values()), dim*numCells, 1);
  cellData->SetName(meshfunction.name().c_str());
  vtk_mesh->GetCellData()->AddArray(cellData);

  // Write out.
  perform_write(vtk_mesh);

  log(TRACE, "Saved mesh function %s (%s) to file %s in Exodus format.",
      mesh.name().c_str(), mesh.label().c_str(), filename.c_str());
}
//----------------------------------------------------------------------------
void ExodusFile::operator<<(const Function& u)
{
  u.update();
  write_function(u, counter);
}
//----------------------------------------------------------------------------
void ExodusFile::operator<<(const std::pair<const Function*, double> u)
{
  dolfin_assert(u.first);
  u.first->update();
  write_function(*(u.first), u.second);
}
//----------------------------------------------------------------------------
void ExodusFile::write_function(const Function& u, double time) const
{
  // Write results
  // Get rank of Function
  const uint rank = u.value_rank();
  if (rank > 2)
  {
    dolfin_error("ExodusFile.cpp",
                 "write data to Exodus file",
                 "Only scalar, vector and tensor functions can be saved in Exodus format");
  }

  // Get number of components
  const uint dim = u.value_size();

  // Check that function type can be handled, cf.
  // http://www.vtk.org/Bug/view.php?id=13508.
  if (dim > 6)
  {
    dolfin_error("ExodusFile.cpp",
                 "write data to Exodus file",
                 "Can't handle more than 6 components");
  }

  // Test for cell-based element type
  dolfin_assert(u.function_space()->mesh());
  const Mesh& mesh = *u.function_space()->mesh();

  vtkSmartPointer<vtkUnstructuredGrid> vtk_mesh =
    create_vtk_mesh(mesh);

  uint cell_based_dim = 1;
  for (uint i = 0; i < rank; i++)
    cell_based_dim *= mesh.topology().dim();

  dolfin_assert(u.function_space()->dofmap());
  const GenericDofMap& dofmap = *u.function_space()->dofmap();
  // Define the vector that holds the values outside the if
  // to make sure it doesn't get destroyed before the data
  // is written out to a file
  std::vector<double> values;
  if (dofmap.max_cell_dimension() == cell_based_dim)
  {
    // Extract DOFs from u.
    const uint num_cells = mesh.num_cells();
    const uint size = num_cells*dim;
    std::vector<int> dof_set;
    for (CellIterator cell(mesh); !cell.end(); ++cell)
    {
      const std::vector<int>& dofs = dofmap.cell_dofs(cell->index());
      for(uint i = 0; i < dofmap.cell_dimension(cell->index()); ++i)
        dof_set.push_back(dofs[i]);
    }
    // Get  values
    values.resize(dof_set.size());
    dolfin_assert(u.vector());
    u.vector()->get_local(&values[0], dof_set.size(), &dof_set[0]);

    // Set the cell array.
    vtkSmartPointer<vtkDoubleArray> cellData =
      vtkSmartPointer<vtkDoubleArray>::New();
    cellData->SetNumberOfComponents(dim);
    cellData->SetArray(&values[0], dof_set.size(), 1);
    cellData->SetName(u.name().c_str());
    vtk_mesh->GetCellData()->AddArray(cellData);
  }
  else
  {
    // Extract point values.
    const uint num_vertices = mesh.num_vertices();
    const uint size = num_vertices*dim;
    values.resize(size);
    u.compute_vertex_values(values, mesh);
    // Set the point array.
    vtkSmartPointer<vtkDoubleArray> pointData =
      vtkSmartPointer<vtkDoubleArray>::New();
    pointData->SetNumberOfComponents(dim);
    pointData->SetArray(&values[0], size, 1);
    pointData->SetName(u.name().c_str());
    vtk_mesh->GetPointData()->AddArray(pointData);
  }

  // Actually write out the data.
  perform_write(vtk_mesh);

  log(TRACE, "Saved function %s (%s) to file %s in Exodus format.",
      u.name().c_str(), u.label().c_str(), filename.c_str());
}
//----------------------------------------------------------------------------
vtkSmartPointer<vtkUnstructuredGrid> ExodusFile::create_vtk_mesh(const Mesh& mesh) const
{
  // Build Exodus unstructured grid object.
  vtkSmartPointer<vtkUnstructuredGrid> unstructuredGrid =
    vtkSmartPointer<vtkUnstructuredGrid>::New();
  // Set the points.
  const int numPoints = mesh.num_vertices();
  vtkSmartPointer<vtkDoubleArray> pointData =
    vtkSmartPointer<vtkDoubleArray>::New();
  pointData->SetNumberOfComponents(3);
  pointData->SetArray(const_cast<double*>(&mesh.coordinates()[0]), 3*numPoints, 1);
  vtkSmartPointer<vtkPoints> points =
    vtkSmartPointer<vtkPoints>::New();
  points->SetData(pointData);
  unstructuredGrid->SetPoints(points);
  // Set cells. Those need to be copied over since the
  // default Dolfin node ID data type is dolfin::uint
  // (typically unsigned int), and the node ID of Exodus
  // is vtkIdType (typically long long int).
  const int numCells = mesh.num_cells();
  const std::vector<unsigned int> cells = mesh.cells();
  vtkSmartPointer<vtkCellArray> cellData =
    vtkSmartPointer<vtkCellArray>::New();
  vtkIdType tmp[4];
  for (int k=0; k<numCells; k++)
  {
    for (int i=0; i<4; i++)
      tmp[i] = cells[4*k+i];
    cellData->InsertNextCell(4, tmp);
  }
  unstructuredGrid->SetCells(VTK_TETRA, cellData);

  return unstructuredGrid;
}
//----------------------------------------------------------------------------
void ExodusFile::perform_write(const vtkSmartPointer<vtkUnstructuredGrid> & vtk_mesh) const
{
  // Instantiate writer.
  vtkSmartPointer<vtkExodusIIWriter> writer =
    vtkSmartPointer<vtkExodusIIWriter>::New();

  // Write out to file.
  writer->SetFileName(filename.c_str());
  writer->SetInput(vtk_mesh);
  writer->Write();

  return;
}
//----------------------------------------------------------------------------
#endif // HAS_VTK && HAS_VTK_EXODUS
