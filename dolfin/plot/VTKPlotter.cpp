// Copyright (C) 2012 Fredrik Valdmanis
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
// First added:  2012-05-23
// Last changed: 2012-05-24 

#ifdef HAS_VTK

#include <vtkPoints.h>
#include <vtkIdList.h>
#include <vtkCellType.h>
#include <vtkGeometryFilter.h>
#include <vtkPolyDataMapper.h>
#include <vtkActor.h>
#include <vtkProperty.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkInteractorStyleTrackballCamera.h>

#include <dolfin/function/FunctionSpace.h> 
#include <dolfin/mesh/Vertex.h>

#include "VTKPlotter.h"

using namespace dolfin;
//----------------------------------------------------------------------------
VTKPlotter::VTKPlotter(const Mesh& mesh) :
  _mesh(reference_to_no_delete_pointer(mesh)),
 _grid(vtkSmartPointer<vtkUnstructuredGrid>::New())
{
  // Do nothing
}
//----------------------------------------------------------------------------
VTKPlotter::VTKPlotter(const Function& function) :
  _mesh(reference_to_no_delete_pointer(*function.function_space()->mesh())),
  _function(reference_to_no_delete_pointer(function)),
  _grid(vtkSmartPointer<vtkUnstructuredGrid>::New())
{
  // Do nothing
}
//----------------------------------------------------------------------------
VTKPlotter::VTKPlotter(const Expression& expression, const Mesh& mesh) :
  _mesh(reference_to_no_delete_pointer(mesh)),
  _function(reference_to_no_delete_pointer(expression)),
  _grid(vtkSmartPointer<vtkUnstructuredGrid>::New()) 
{
  // Do nothing
}
//----------------------------------------------------------------------------
VTKPlotter::~VTKPlotter()
{
  // Do nothing
}
//----------------------------------------------------------------------------
void VTKPlotter::init()
{
  // FIXME: Is this assert redundant because of the constructors' initialization lists?
  dolfin_assert(_mesh);

  construct_vtk_grid();

  // Extract function values into VTK array
  // FIXME: Must be generalized to vector valued functions
  /*if (_function) {
    vtkSmartPointer<vtkFloatArray> scalars = vtkSmartPointer<vtkFloatArray>::New();
    uint num_vertices = _mesh->num_vertices();
    std::vector<double> vertex_values(num_vertices);
    _function->compute_vertex_values(vertex_values, *_mesh);
    for(int i = 0; i < num_vertices; ++i) {
      scalars->InsertValue(i, vertex_values[i]);
    }
    _grid->GetPointData()->SetScalars(scalars);
  */
}
//----------------------------------------------------------------------------
void VTKPlotter::construct_vtk_grid()
{
  // Construct vtkUnstructuredGrid from DOLFIN mesh
  // FIXME: Is this assert redundant because of the constructors' initialization lists?
  dolfin_assert(_grid);

  // Construct VTK point array from DOLFIN mesh vertices
  vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New(); 
  points->Allocate(_mesh->num_vertices());
  dolfin::Point p;
 
  for (VertexIterator vertex(*_mesh); !vertex.end(); ++vertex) {
    p = vertex->point();
    points->InsertNextPoint(p.x(), p.y(), p.z());
  }

  // Allocate storage in VTK grid for number of cells times spatial dim + 1, 
  // since triangles (2D) have 3 vertices, tetrahedrons (3D) have 4 vertices, etc.
  uint spatial_dim = _mesh->topology().dim();
  _grid->Allocate(_mesh->num_cells()*(spatial_dim+1));
  
  // Get mesh connectivity (i.e. cells), iterate and add cells to VTK grid
  const uint *connectivity = _mesh->cells();

  // vtkIdList to hold point IDs for a new cell in each iteration
  vtkSmartPointer<vtkIdList> ids = vtkSmartPointer<vtkIdList>::New();
  for (uint i = 0; i < _mesh->num_cells(); ++i) {
    ids->Initialize();
    
    // Insert all vertex ids for a given cell. For a simplex cell in nD, n+1 ids are inserted.
    // The connectivity array must be indexed at ((n+1) x cell_number + id_offset)
    for(uint j = 0; j <= spatial_dim; ++j) {
      ids->InsertNextId((vtkIdType) connectivity[(spatial_dim+1)*i + j]);
    }
   
    // Insert cell into VTK grid
    switch (spatial_dim) {
      case 1:
        _grid->InsertNextCell(VTK_LINE, ids);
        break;
      case 2:
        _grid->InsertNextCell(VTK_TRIANGLE, ids);
        break;
      case 3:
        _grid->InsertNextCell(VTK_TETRA, ids);
        break;
      default:
        // Throw dolfin_error?
        break;
    }
  }

  // Set points in grid and free unused allocated memory
  _grid->SetPoints(points);
  _grid->Squeeze();


}
//----------------------------------------------------------------------------
void VTKPlotter::plot(std::string title)
{
  // dolfin_assert(_grid) should be performed but doesn't make sense here
  // should the contents of init be moved to the constructor? Or would that
  // be a problem with regards to parameter handling?

  // Create VTK geometry filter and attach grid to it
  vtkSmartPointer<vtkGeometryFilter> geometryFilter = 
    vtkSmartPointer<vtkGeometryFilter>::New();
  geometryFilter->SetInput(_grid);
  geometryFilter->Update();

  // Create VTK mapper and attach geometry filter to it
  vtkSmartPointer<vtkPolyDataMapper> mapper = 
    vtkSmartPointer<vtkPolyDataMapper>::New();
  mapper->SetInputConnection(geometryFilter->GetOutputPort());

  // Create VTK actor and attach the mapper to it 
  vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
  actor->SetMapper(mapper);

  // FIXME: These properties should be gotten from parameters
  // Idea: Wireframe is a parameter. plot(mesh) sets it to true, plot(function) to false
  // default is wireframe on?
  actor->GetProperty()->SetRepresentationToWireframe();
  actor->GetProperty()->SetColor(0,0,1);

  vtkSmartPointer<vtkRenderer> renderer = vtkSmartPointer<vtkRenderer>::New();
  renderer->AddActor(actor);
  // FIXME: Get background color from parameters?
  renderer->SetBackground(1,1,1);

  vtkSmartPointer<vtkRenderWindow> window = 
    vtkSmartPointer<vtkRenderWindow>::New();
  window->AddRenderer(renderer);
  window->SetSize(600,600);

  // Make window title. Should probably be fetched from parameters?
  std::stringstream full_title;
  full_title << title << " - DOLFIN VTK Plotter";
  window->SetWindowName(full_title.str().c_str());

  vtkSmartPointer<vtkRenderWindowInteractor> interactor = 
    vtkSmartPointer<vtkRenderWindowInteractor>::New();
  interactor->SetRenderWindow(window);
 
  // FIXME: Get interactorstyle from parameters? 
  vtkSmartPointer<vtkInteractorStyleTrackballCamera> style = 
    vtkSmartPointer<vtkInteractorStyleTrackballCamera>::New();
  interactor->SetInteractorStyle(style);
  interactor->Initialize();
  interactor->Start();
}
//----------------------------------------------------------------------------

#endif
