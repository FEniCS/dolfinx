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
// Last changed: 2012-05-30

#ifdef HAS_VTK

#include <vtkPoints.h>
#include <vtkCellArray.h>
#include <vtkCellType.h>
#include <vtkFloatArray.h>
#include <vtkPointData.h>
#include <vtkVectorNorm.h>
#include <vtkWarpScalar.h>
#include <vtkWarpVector.h>
#include <vtkArrowSource.h>
#include <vtkGlyph3D.h>
#include <vtkGeometryFilter.h>
#include <vtkLookupTable.h>
#include <vtkTextProperty.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkRenderer.h>
#include <vtkTextActor.h>
#include <vtkBalloonRepresentation.h>
#include <vtkBalloonWidget.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkInteractorStyleTrackballCamera.h>

#include <dolfin/function/FunctionSpace.h> 
#include <dolfin/mesh/Vertex.h>
#include <dolfin/common/Timer.h>

#include "VTKPlotter.h"

using namespace dolfin;
//----------------------------------------------------------------------------
VTKPlotter::VTKPlotter(const Mesh& mesh) :
  _mesh(reference_to_no_delete_pointer(mesh)),
  _grid(vtkSmartPointer<vtkUnstructuredGrid>::New())
{
  parameters = default_mesh_parameters();
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
  dolfin_assert(_mesh);

  construct_vtk_grid();

  if (_function) {
    // Are we plotting a function?
    switch (_function->value_rank()) {
      // Is it scalar valued?
      case 0:
        plot_scalar_function();
        break;
        // Or vector valued?
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
  dolfin_assert(_grid);

  Timer t("Construct VTK grid");

  // Construct VTK point array from DOLFIN mesh vertices
  vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New(); 
  points->SetNumberOfPoints(_mesh->num_vertices());
  Point p;

  for (VertexIterator vertex(*_mesh); !vertex.end(); ++vertex) {
    p = vertex->point();
    points->SetPoint(vertex->index(), p.x(), p.y(), p.z());
  }

  // Add mesh cells to VTK cell array. Note: Preallocation of storage 
  // in cell array did not give speedups when testing during development 
  vtkSmartPointer<vtkCellArray> cells = vtkSmartPointer<vtkCellArray>::New();
  const uint *connectivity = _mesh->cells();
  uint spatial_dim = _mesh->topology().dim();

  for (uint i = 0; i < _mesh->num_cells(); ++i) {

    // Insert all vertex indices for a given cell. For a simplex cell in nD, 
    // n+1 indices are inserted. The connectivity array must be indexed at 
    // ((n+1) x cell_number + idx_offset)
    cells->InsertNextCell(spatial_dim+1);
    for(uint j = 0; j <= spatial_dim; ++j) {
      cells->InsertCellPoint(connectivity[(spatial_dim+1)*i + j]);
    }
  }
  // Free unused memory in cell array 
  // (automatically allocated during cell insertion)
  cells->Squeeze();

  // Insert points and cells in VTK unstructured grid
  _grid->SetPoints(points);
  switch (spatial_dim) {
    case 1:
      _grid->SetCells(VTK_LINE, cells);
      break;
    case 2:
      _grid->SetCells(VTK_TRIANGLE, cells);
      break;
    case 3:
      _grid->SetCells(VTK_TETRA, cells);
      break;
    default:
      // Should never be reached
      break;
  }
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
    warp->SetScaleFactor(parameters["warp_scalefactor"]);

    // Pass VTK point set to be filtered, mapped and rendered 
    filter_and_map(warp->GetOutput());
  }
  else {
    // In 3D, we just show the scalar values as colors on the mesh by 
    // passing the grid with scalar values attached (such a grid is a
    // VTK point set and hence accepted by filter_and_map) to be filtered, 
    // mapped and rendered 
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
      // else, add the last entry in the value vector
    } else {
      vectors->SetValue(3*i + 2, vertex_values[i + num_vertices*2]);
    }
  }
  // Attach vectors as vector point data in the VTK grid
  _grid->GetPointData()->SetVectors(vectors);

  // Compute norms of vector data
  vtkSmartPointer<vtkVectorNorm> norms = 
    vtkSmartPointer<vtkVectorNorm>::New();
  norms->SetInput(_grid);
  norms->SetAttributeModeToUsePointData();
  //NOTE: This update is necessary to actually compute the norms
  norms->Update();

  // Attach vector norms as scalar point data in the VTK grid
  _grid->GetPointData()->SetScalars(
      norms->GetOutput()->GetPointData()->GetScalars());

  // FIXME: The below block calls plot_warp or plot_glyphs assuming
  // that function data has been attached to the grid. They depend on a 
  // certain state of the object and it will make no sense to call them
  // if the object is not in this state (i.e. if function values has not been
  // attached). How to make this more secure?
  const std::string mode = this->parameters["vector_mode"];
  if(mode == "warp") {
    info("Using warp (displacement) vector plotting mode.");
    plot_warp();
    return;
  } else if (mode == "glyphs") {
    info("Using glyphs (displacement) for plotting vectors.");
  } else {
    warning("Unrecognized option \"" + mode + "\", using default (glyphs).");
  }
  plot_glyphs();
}
//----------------------------------------------------------------------------
void VTKPlotter::plot_warp()
{
  vtkSmartPointer<vtkWarpVector> warp = vtkSmartPointer<vtkWarpVector>::New();
  warp->SetInput(_grid);
  warp->SetScaleFactor(parameters["warp_scalefactor"]);

  // The warp must be filtered and mapped before rendering
  filter_and_map(warp->GetOutput());
}
//----------------------------------------------------------------------------
void VTKPlotter::plot_glyphs()
{
  // Create the glyph symbol to use, a VTK arrow
  vtkSmartPointer<vtkArrowSource> arrow = 
    vtkSmartPointer<vtkArrowSource>::New();
  arrow->SetTipRadius(0.08);
  arrow->SetTipResolution(16);
  arrow->SetTipLength(0.25);
  arrow->SetShaftRadius(0.05);
  arrow->SetShaftResolution(16);

  // Create the glyph object, set source (the arrow) and input (the grid) and
  // adjust various parameters
  vtkSmartPointer<vtkGlyph3D> glyphs = vtkSmartPointer<vtkGlyph3D>::New();
  glyphs->SetSourceConnection(arrow->GetOutputPort());
  glyphs->SetInput(_grid);
  glyphs->SetVectorModeToUseVector();
  glyphs->SetScaleModeToScaleByVector();
  glyphs->SetColorModeToColorByVector();
  glyphs->SetScaleFactor(parameters["glyph_scalefactor"]);

  // The glyphs need not be filtered and can be mapped directly
  map(glyphs);
}
//----------------------------------------------------------------------------
void VTKPlotter::filter_and_map(vtkSmartPointer<vtkPointSet> point_set)
{
  // Create VTK geometry filter and attach VTK point set to it
  vtkSmartPointer<vtkGeometryFilter> geometryFilter = 
    vtkSmartPointer<vtkGeometryFilter>::New();
  geometryFilter->SetInput(point_set);
  geometryFilter->Update();

  map(geometryFilter);
}
//----------------------------------------------------------------------------
void VTKPlotter::map(vtkSmartPointer<vtkPolyDataAlgorithm> polyData)
{
  // Get range of scalar values
  double range[2];
  _grid->GetScalarRange(range);

  // Make lookup table
  vtkSmartPointer<vtkLookupTable> lut = vtkSmartPointer<vtkLookupTable>::New();
  lut->SetRange(range);
  lut->Build();

  // Create VTK mapper and attach VTK poly data to it
  vtkSmartPointer<vtkPolyDataMapper> mapper = 
    vtkSmartPointer<vtkPolyDataMapper>::New();
  mapper->SetInputConnection(polyData->GetOutputPort());
  mapper->SetLookupTable(lut);
  // This call is what actually changes the surface color
  mapper->SetScalarRange(range);

  // Create scalar bar if the parameter tells us to
  if (parameters["scalarbar"]) {
    // FIXME: _scalarbar is a class member now. There must be cleaner way
    // to do this
    // FIXME: The title "Vector magnitude" should be added to the scalarbar
    // when plotting glyphs
    _scalarbar = vtkSmartPointer<vtkScalarBarActor>::New();
    _scalarbar->SetLookupTable(lut);
    vtkSmartPointer<vtkTextProperty> labelprop = 
      _scalarbar->GetLabelTextProperty();
    labelprop->SetFontFamilyToTimes();
    labelprop->SetColor(0, 0, 0);
    labelprop->SetFontSize(14);
    // TODO: Add similar appearance changes to TitleTextProperty
  }

  // Create VTK actor and attach the mapper to it 
  vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
  actor->SetMapper(mapper);
  if (parameters["wireframe"]) {
    actor->GetProperty()->SetRepresentationToWireframe();
  }
  // FIXME: Get color from parameters?
  // This color property is only used for plotting of meshes. 
  actor->GetProperty()->SetColor(0,0,1);

  render(actor);
}
//----------------------------------------------------------------------------
void VTKPlotter::render(vtkSmartPointer<vtkActor> actor)
{
  // Set up renderer and add the actor to it
  vtkSmartPointer<vtkRenderer> renderer = vtkSmartPointer<vtkRenderer>::New();
  renderer->AddActor(actor);
  // FIXME: Get background color from parameters?
  renderer->SetBackground(1,1,1);

  // If present, add the scalar bar actor to the renderer
  if (_scalarbar) {
    renderer->AddActor(_scalarbar);
  }

  // Add helptextactor text actor
  vtkSmartPointer<vtkTextActor> helptextactor =
    vtkSmartPointer<vtkTextActor>::New();
  helptextactor->SetPosition(10,10);
  helptextactor->SetInput("Help ");
  helptextactor->GetTextProperty()->SetColor(0.0, 0.0, 0.0);
  helptextactor->GetTextProperty()->SetFontSize(24);
  helptextactor->GetTextProperty()->SetFontFamilyToTimes();
  renderer->AddActor2D(helptextactor);

  // Set up renderwindow, add renderer, set size and make window title
  vtkSmartPointer<vtkRenderWindow> window = 
    vtkSmartPointer<vtkRenderWindow>::New();
  window->AddRenderer(renderer);
  window->SetSize(600,600);
  std::stringstream full_title;
  full_title << "DOLFIN: " << std::string(parameters["title"]);
  window->SetWindowName(full_title.str().c_str());

  // Set interactor style to trackball camera
  vtkSmartPointer<vtkInteractorStyleTrackballCamera> style = 
    vtkSmartPointer<vtkInteractorStyleTrackballCamera>::New();

  // Set up interactor and start rendering loop
  vtkSmartPointer<vtkRenderWindowInteractor> interactor = 
    vtkSmartPointer<vtkRenderWindowInteractor>::New();
  interactor->SetRenderWindow(window);
  interactor->SetInteractorStyle(style);

  // Set up the representation for the hover-over help text box
  vtkSmartPointer<vtkBalloonRepresentation> balloonrep =
    vtkSmartPointer<vtkBalloonRepresentation>::New();
  balloonrep->SetOffset(5,5);
  balloonrep->GetTextProperty()->SetFontSize(18);
  balloonrep->GetTextProperty()->SetFontFamilyToTimes();
  
  // Set up the actual widget that makes the help text pop up
  vtkSmartPointer<vtkBalloonWidget> balloonwidget =
    vtkSmartPointer<vtkBalloonWidget>::New();
  balloonwidget->SetInteractor(interactor);
  balloonwidget->SetRepresentation(balloonrep);
  balloonwidget->AddBalloon(helptextactor,
      get_helptext().c_str(),NULL);

  // Render window and enable widget. The Render() call is necessary
  window->Render(); 
  balloonwidget->EnabledOn();

  // Initialize and start the mouse interaction
  interactor->Initialize();
  interactor->Start();
}
//----------------------------------------------------------------------------
std::string VTKPlotter::get_helptext()
{
  std::stringstream text;

  text << "Mouse control:\n";
  text << "\t Left mouse button: Rotate figure\n";
  text << "\t Right mouse button (or scroolwheel): Zoom \n";
  text << "\t Middle mouse button (or left+right): Translate figure\n\n";
  text << "Keyboard control:\n";
  text << "\t R: Reset zoom\n";
  text << "\t W: View figure as wireframe\n";
  text << "\t S: View figure with solid surface\n";
  text << "\t F: Fly to the point currently under the mouse pointer\n";
  text << "\t P: Add bounding box\n";
  text << "\t E/Q: Exit\n";
  return text.str();
}
#endif
