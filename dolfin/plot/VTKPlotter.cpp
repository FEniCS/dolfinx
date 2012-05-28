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
// Modified by Benjamin Kehlet, 2012
//
// First added:  2012-05-23
// Last changed: 2012-05-28 

#ifdef HAS_VTK

#include <vtkPoints.h>
#include <vtkIdList.h>
#include <vtkCellType.h>
#include <vtkFloatArray.h>
#include <vtkPointData.h>
#include <vtkWarpScalar.h>
#include <vtkWarpVector.h>
#include <vtkArrowSource.h>
#include <vtkGlyph3D.h>
#include <vtkGeometryFilter.h>
#include <vtkPolyDataMapper.h>
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
  parameters = default_parameters();
}
//----------------------------------------------------------------------------
VTKPlotter::VTKPlotter(const Function& function) :
  _mesh(reference_to_no_delete_pointer(*function.function_space()->mesh())),
  _function(reference_to_no_delete_pointer(function)),
  _grid(vtkSmartPointer<vtkUnstructuredGrid>::New())
{
  parameters = default_parameters();
}
//----------------------------------------------------------------------------
VTKPlotter::VTKPlotter(const Expression& expression, const Mesh& mesh) :
  _mesh(reference_to_no_delete_pointer(mesh)),
  _function(reference_to_no_delete_pointer(expression)),
  _grid(vtkSmartPointer<vtkUnstructuredGrid>::New()) 
{
  parameters = default_parameters();
}
//----------------------------------------------------------------------------
VTKPlotter::VTKPlotter(const MeshFunction<uint>& mesh_function) :
  _mesh(reference_to_no_delete_pointer(mesh_function.mesh())),
  _grid(vtkSmartPointer<vtkUnstructuredGrid>::New()) 
{
  parameters = default_parameters();
}
//----------------------------------------------------------------------------
VTKPlotter::VTKPlotter(const MeshFunction<double>& mesh_function) :
  _mesh(reference_to_no_delete_pointer(mesh_function.mesh())),
  _grid(vtkSmartPointer<vtkUnstructuredGrid>::New()) 
{
  parameters = default_parameters();
}
//----------------------------------------------------------------------------
VTKPlotter::VTKPlotter(const MeshFunction<bool>& mesh_function) :
  _mesh(reference_to_no_delete_pointer(mesh_function.mesh())),
  _grid(vtkSmartPointer<vtkUnstructuredGrid>::New()) 
{
  parameters = default_parameters();
}
//----------------------------------------------------------------------------
VTKPlotter::~VTKPlotter()
{
  // Do nothing
}
//----------------------------------------------------------------------------
void VTKPlotter::plot()
{
  // FIXME: Is this assert redundant because of the constructors' initialization lists?
  dolfin_assert(_mesh);

  construct_vtk_grid();

  if (_function) {
    // Are we plotting a function?
 
      // TODO: Check if the function is vector valued or scalar valued
    //
    // Make value arrays of vectors/scalars
    //
    // Call corresponding functions for computing the visualization of the values over the grid
    //
    // There can be different visualization functions for the different cases. They end up with calling 
    // VTKPlotter::filter_and_map with their computed vtkPointSet
    //
    // Except those that visualize glyphs, they must create the Glyphs, actors etc and directly call 
    // VTKPlotter::render with the computed actor
    
    switch (_function->value_rank()) {
      case 0:
        plot_scalar_function();
        break;
      case 1:
        plot_vector_function();
        break;
      default:
        dolfin_error("VTKPlotter.cpp",
                     "plot function of rank > 2.",
                     "Plotting of higher order functions is not supported.");
    }
  }
  /*else if (_mesh_function) {
   // Or are we plotting a mesh function?

  }*/
  else {
    // Or just a mesh?
    filter_and_map(_grid);
  }
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
void VTKPlotter::plot_scalar_function()
{
  dolfin_assert(_function->value_rank() == 0);

  // Make VTK float array and allocate storage for function values
  uint num_vertices = _mesh->num_vertices();
  vtkSmartPointer<vtkFloatArray> scalars = 
    vtkSmartPointer<vtkFloatArray>::New();
    scalars->SetNumberOfValues(num_vertices);

  // Evaluate DOLFIN function and copy values to the VTK array
  std::vector<double> vertex_values(num_vertices); 
  _function->compute_vertex_values(vertex_values, *_mesh);
  for(uint i = 0; i < num_vertices; ++i) {
    scalars->SetValue(i, vertex_values[i]);
  }

  // Attach scalar values as point data in the VTK grid
  _grid->GetPointData()->SetScalars(scalars);

  // Depending on the geometrical dimension, we use different algorithms
  // to visualize the scalar data.
  if (_mesh->topology().dim() < 3) {
    // In 1D and 2D, we warp the mesh according to the scalar values
    vtkSmartPointer<vtkWarpScalar> warp = 
      vtkSmartPointer<vtkWarpScalar>::New();
      warp->SetInput(_grid);
      warp->SetScaleFactor(1.0); // FIXME: Get from parameters

    // Pass VTK point set to be filtered, mapped and rendered 
    filter_and_map(warp->GetOutput());
  }
  else {
    // In 3D, we just show the scalar values as colors on the mesh by 
    // passing the grid the grid with scalar values attached (i.e. a
    // VTK point set) to be filtered, mapped and rendered 
    filter_and_map(_grid);
  }
}
//----------------------------------------------------------------------------
void VTKPlotter::plot_vector_function()
{
  dolfin_assert(_function->value_rank() == 1);

  // Make VTK float array and allocate storage for function vector values
  uint num_vertices = _mesh->num_vertices();
  uint num_components = _function->value_dimension(0);
  std::cout << "numcomp: " << num_components << std::endl;
  vtkSmartPointer<vtkFloatArray> vectors = 
    vtkSmartPointer<vtkFloatArray>::New();
    // NOTE: Allocation must be done in this order!
    // Note also that the number of VTK vector components must always be 3 
    // regardless of the function vector value dimension
    vectors->SetNumberOfComponents(3);
    vectors->SetNumberOfTuples(num_vertices); 

  // Evaluate DOLFIN function and copy values to the VTK array
  // The entries in "vertex_values" must be copied to "vectors". Viewing
  // these arrays as matrices, the transpose of vertex values should be copied,
  // since DOLFIN and VTK store vector function values differently
  std::vector<double> vertex_values(num_vertices*num_components);
  _function->compute_vertex_values(vertex_values, *_mesh);
  for(uint i = 0; i < num_vertices; ++i) {
    vectors->SetValue(3*i,     vertex_values[i]);
    vectors->SetValue(3*i + 1, vertex_values[i + num_vertices]);
    // If the DOLFIN function vector value dimension is 2, pad with a 0
    if(num_components == 2) {
      vectors->SetValue(3*i + 2, 0.0);
    } else {
      vectors->SetValue(3*i + 2, vertex_values[i + num_vertices*2]);
    }
  }

  // Attach vector values as point data in the VTK grid
  _grid->GetPointData()->SetVectors(vectors);

  // FIXME: The below block calls plot_warp or plot_glyphs assuming
  // that function data has been attached to the grid. They depend on a 
  // certain state of the object and it will make no sense to call them
  // if the object is not in this state (i.e. if function values has not been
  // added). How to make this more secure?

  const std::string mode = this->parameters["vector_mode"];
  if(mode == "warp") {
    info("Using warp (displacement) vector plotting mode.");
    plot_warp();
    return;
  } else if (mode == "glyphs") {
    info("Using glyphs (displacement) for plotting vectors.");
  } else {
    warning("Unrecognized option " + mode + ", using default (glyphs).");
  }
  plot_glyphs();
}
//----------------------------------------------------------------------------
void VTKPlotter::plot_warp()
{
  // Use warp vector visualization 
  // FIXME: Read "mode" parameter and branch for glyph visualization
  vtkSmartPointer<vtkWarpVector> warp = vtkSmartPointer<vtkWarpVector>::New();
  warp->SetInput(_grid);
  warp->SetScaleFactor(1.0); // FIXME: Get from parameters

  filter_and_map(warp->GetOutput());
}
//----------------------------------------------------------------------------
void VTKPlotter::plot_glyphs()
{
  // Create the glyph, a VTK arrow
  vtkSmartPointer<vtkArrowSource> arrow = 
    vtkSmartPointer<vtkArrowSource>::New();
    arrow->SetTipRadius(0.08);
    arrow->SetTipResolution(16);
    arrow->SetTipLength(0.25);
    arrow->SetShaftRadius(0.05);
    arrow->SetShaftResolution(16);

  vtkSmartPointer<vtkGlyph3D> glyphs = vtkSmartPointer<vtkGlyph3D>::New();
    glyphs->SetSourceConnection(arrow->GetOutputPort());
    glyphs->SetInput(_grid);
    glyphs->SetVectorModeToUseVector();
    glyphs->SetScaleModeToScaleByVector();
    glyphs->SetColorModeToColorByVector();
    //glyphs->SetScaleFactor(0.1);

  vtkSmartPointer<vtkPolyDataMapper> mapper = 
    vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper->SetInputConnection(glyphs->GetOutputPort());

  vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);

  render(actor);

}
//----------------------------------------------------------------------------
void VTKPlotter::filter_and_map(vtkSmartPointer<vtkPointSet> point_set)
{
  // Create VTK geometry filter and attach grid to it
  vtkSmartPointer<vtkGeometryFilter> geometryFilter = 
    vtkSmartPointer<vtkGeometryFilter>::New();
    geometryFilter->SetInput(point_set);
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

  render(actor);
}
//----------------------------------------------------------------------------
void VTKPlotter::render(vtkSmartPointer<vtkActor> actor)
{
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
    full_title << std::string(parameters["title"]) << " - DOLFIN VTK Plotter";
    window->SetWindowName(full_title.str().c_str());

  // FIXME: Get interactorstyle from parameters? 
  vtkSmartPointer<vtkInteractorStyleTrackballCamera> style = 
    vtkSmartPointer<vtkInteractorStyleTrackballCamera>::New();

  vtkSmartPointer<vtkRenderWindowInteractor> interactor = 
    vtkSmartPointer<vtkRenderWindowInteractor>::New();
    interactor->SetRenderWindow(window);
    interactor->SetInteractorStyle(style);
    interactor->Initialize();
    interactor->Start();
}

#endif
