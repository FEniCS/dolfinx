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

#ifdef HAS_VTK
#ifdef HAS_VTK_EXODUS

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
  : GenericFile(filename, "Exodus"),
    _writer(vtkSmartPointer<vtkExodusIIWriter>::New())
{
  // Set filename.
  _writer->SetFileName(filename.c_str());
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
      mesh.name().c_str(), mesh.label().c_str(), _filename.c_str());
  return;
}
//----------------------------------------------------------------------------
void ExodusFile::operator<<(const MeshFunction<unsigned int>& meshfunction)
{
  dolfin_assert(meshfunction.mesh());
  const Mesh& mesh = *meshfunction.mesh();
  const std::size_t cell_dim = meshfunction.dim();

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

  // Create Exodus mesh
  vtkSmartPointer<vtkUnstructuredGrid> vtk_mesh = create_vtk_mesh(mesh);

  // Add cell data
  const std::size_t dim = meshfunction.dim();
  const std::size_t numCells = mesh.num_cells();
  vtkSmartPointer<vtkUnsignedIntArray> cellData =
    vtkSmartPointer<vtkUnsignedIntArray>::New();
  cellData->SetNumberOfComponents(dim);
  cellData->SetArray(const_cast<unsigned int*>(meshfunction.values()), dim*numCells, 1);
  cellData->SetName(meshfunction.name().c_str());
  vtk_mesh->GetCellData()->AddArray(cellData);

  // Write out.
  perform_write(vtk_mesh);

  log(TRACE, "Saved mesh function %s (%s) to file %s in Exodus format.",
      mesh.name().c_str(), mesh.label().c_str(), _filename.c_str());
}
//----------------------------------------------------------------------------
void ExodusFile::operator<<(const MeshFunction<int>& meshfunction)
{
  dolfin_assert(meshfunction.mesh());
  const Mesh& mesh = *meshfunction.mesh();
  const std::size_t cell_dim = meshfunction.dim();

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

  // Create Exodus mesh
  vtkSmartPointer<vtkUnstructuredGrid> vtk_mesh = create_vtk_mesh(mesh);

  // Add cell data
  const std::size_t dim = meshfunction.dim();
  const std::size_t numCells = mesh.num_cells();
  vtkSmartPointer<vtkIntArray> cellData =
    vtkSmartPointer<vtkIntArray>::New();
  cellData->SetNumberOfComponents(dim);
  cellData->SetArray(const_cast<int*>(meshfunction.values()), dim*numCells, 1);
  cellData->SetName(meshfunction.name().c_str());
  vtk_mesh->GetCellData()->AddArray(cellData);

  // Write out
  perform_write(vtk_mesh);

  log(TRACE, "Saved mesh function %s (%s) to file %s in Exodus format.",
      mesh.name().c_str(), mesh.label().c_str(), _filename.c_str());
}
//----------------------------------------------------------------------------
void ExodusFile::operator<<(const MeshFunction<double>& meshfunction)
{
  dolfin_assert(meshfunction.mesh());
  const Mesh& mesh = *meshfunction.mesh();
  const std::size_t cell_dim = meshfunction.dim();

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

  // Create Exodus mesh
  vtkSmartPointer<vtkUnstructuredGrid> vtk_mesh = create_vtk_mesh(mesh);

  // Add cell data
  const std::size_t dim = meshfunction.dim();
  const std::size_t numCells = mesh.num_cells();
  vtkSmartPointer<vtkDoubleArray> cellData =
    vtkSmartPointer<vtkDoubleArray>::New();
  cellData->SetNumberOfComponents(dim);
  cellData->SetArray(const_cast<double*>(meshfunction.values()), dim*numCells, 1);
  cellData->SetName(meshfunction.name().c_str());
  vtk_mesh->GetCellData()->AddArray(cellData);

  // Write out
  perform_write(vtk_mesh);

  log(TRACE, "Saved mesh function %s (%s) to file %s in Exodus format.",
      mesh.name().c_str(), mesh.label().c_str(), _filename.c_str());
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
  const std::size_t rank = u.value_rank();
  if (rank > 2)
  {
    dolfin_error("ExodusFile.cpp",
                 "write data to Exodus file",
                 "Only scalar, vector and tensor functions can be saved in Exodus format");
  }

  // Get number of components
  const std::size_t dim = u.value_size();

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

  std::size_t cell_based_dim = 1;
  for (std::size_t i = 0; i < rank; i++)
    cell_based_dim *= mesh.topology().dim();

  dolfin_assert(u.function_space()->dofmap());
  const GenericDofMap& dofmap = *u.function_space()->dofmap();

  // Define the vector that holds the values outside the if
  // to make sure it doesn't get destroyed before the data
  // is written out to a file
  std::vector<double> values;
  if (dofmap.max_cell_dimension() == cell_based_dim)
  {
    // Extract DOFs from u
    std::vector<int> dof_set;
    for (CellIterator cell(mesh); !cell.end(); ++cell)
    {
      const std::vector<int>& dofs = dofmap.cell_dofs(cell->index());
      for(std::size_t i = 0; i < dofmap.cell_dimension(cell->index()); ++i)
        dof_set.push_back(dofs[i]);
    }
    // Get values
    values.resize(dof_set.size());
    dolfin_assert(u.vector());
    u.vector()->get_local(&values[0], dof_set.size(), &dof_set[0]);

    // Set the cell array
    vtkSmartPointer<vtkDoubleArray> cellData =
      vtkSmartPointer<vtkDoubleArray>::New();
    cellData->SetNumberOfComponents(dim);
    cellData->SetArray(&values[0], dof_set.size(), 1);
    cellData->SetName(u.name().c_str());
    vtk_mesh->GetCellData()->AddArray(cellData);
  }
  else
  {
    // Extract point values
    const std::size_t num_vertices = mesh.num_vertices();
    const std::size_t size = num_vertices*dim;
    values.resize(size);
    u.compute_vertex_values(values, mesh);

    // Set the point array
    vtkSmartPointer<vtkDoubleArray> pointData =
      vtkSmartPointer<vtkDoubleArray>::New();
    pointData->SetNumberOfComponents(dim);
    pointData->SetArray(&values[0], size, 1);
    pointData->SetName(u.name().c_str());
    vtk_mesh->GetPointData()->AddArray(pointData);
  }

  // Actually write out the data
  perform_write(vtk_mesh);

  log(TRACE, "Saved function %s (%s) to file %s in Exodus format.",
      u.name().c_str(), u.label().c_str(), _filename.c_str());
}
//----------------------------------------------------------------------------
vtkSmartPointer<vtkUnstructuredGrid> ExodusFile::create_vtk_mesh(const Mesh& mesh) const
{
  // Build Exodus unstructured grid object
  vtkSmartPointer<vtkUnstructuredGrid> unstructuredGrid =
    vtkSmartPointer<vtkUnstructuredGrid>::New();

  const std::size_t d = mesh.topology().dim();

  // Set the points
  const std::size_t numPoints = mesh.num_vertices();
  vtkSmartPointer<vtkPoints> points =
    vtkSmartPointer<vtkPoints>::New();
  // In VTK, all mesh topologies have nodes with coordinates X, Y, Z.
  // Hence, for 1D and 2D, fill the remaining coordinates with 0
  if (d==1)
  {
    points->SetNumberOfPoints(numPoints);
    const std::vector<double> & coords = mesh.coordinates();
    for (std::size_t k=0; k<numPoints; k++)
      points->SetPoint(k, coords[k], 0.0, 0.0);
  }
  else if (d==2)
  {
    points->SetNumberOfPoints(numPoints);
    const std::vector<double> & coords = mesh.coordinates();
    for (std::size_t k=0; k<numPoints; k++)
      points->SetPoint(k, coords[2*k], coords[2*k+1], 0.0);
  }
  else if (d==3)
  {
    // For 3D topologies, we can just move some pointers around to give VTK
    // access to the node coordinates
    vtkSmartPointer<vtkDoubleArray> pointData =
      vtkSmartPointer<vtkDoubleArray>::New();
    pointData->SetNumberOfComponents(3);
    pointData->SetArray(const_cast<double*>(&mesh.coordinates()[0]), 3*numPoints, 1);
    points->SetData(pointData);
  }
  else
    dolfin_error("ExodusFile.cpp",
                 "extract node coordinates",
                 "Illegal topological dimension");
  unstructuredGrid->SetPoints(points);

  // Set cells. Those need to be copied over since the default Dolfin
  // node ID data type is std::size_t and the node ID of Exodus is
  // vtkIdType (typically long long int).
  const std::size_t n = d+1;
  const std::size_t numCells = mesh.num_cells();
  const std::vector<unsigned int> cells = mesh.cells();
  vtkSmartPointer<vtkCellArray> cellData =
    vtkSmartPointer<vtkCellArray>::New();
  // Allocate 4 entries, we may use less though
  vtkIdType tmp[4];
  for (std::size_t k=0; k<numCells; k++)
  {
    for (std::size_t i=0; i<n; i++)
      tmp[i] = cells[n*k+i];
    cellData->InsertNextCell(n, tmp);
  }

  if (n == 2)
    unstructuredGrid->SetCells(VTK_LINE, cellData);
  else if (n == 3)
    unstructuredGrid->SetCells(VTK_TRIANGLE, cellData);
  else if (n == 4)
    unstructuredGrid->SetCells(VTK_TETRA, cellData);
  else
    dolfin_error("ExodusFile.cpp",
                 "construct VTK mesh",
                 "Illegal element node number");


  return unstructuredGrid;
}
//----------------------------------------------------------------------------
void ExodusFile::perform_write(const vtkSmartPointer<vtkUnstructuredGrid> & vtk_mesh) const
{
  // Write out to file.
  _writer->SetInput(vtk_mesh);
  _writer->Write();

  return;
}
//----------------------------------------------------------------------------

#endif // HAS_VTK_EXODUS
#endif // HAS_VTK
