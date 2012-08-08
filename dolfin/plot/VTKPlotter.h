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
// Modified by Benjamin Kehlet 2012
//
// First added:  2012-05-23
// Last changed: 2012-08-11

#ifndef __VTK_PLOTTER_H
#define __VTK_PLOTTER_H

#include <list>
#include <string>
#include <boost/scoped_ptr.hpp>
#include <boost/shared_ptr.hpp>

#include <dolfin/common/types.h>
#include <dolfin/common/Variable.h>
#include <dolfin/parameter/Parameters.h>

class vtkObject;

namespace dolfin
{

  // Forward declarations
  class DirichletBC;
  class Expression;
  class ExpressionWrapper;
  class Function;
  class GenericVTKPlottable;
  class Mesh;
  class PrivateVTKPipeline;
  template<typename T> class Array;
  template<typename T> class MeshFunction;

  /// This class enables visualization of various DOLFIN entities.
  /// It supports visualization of meshes, functions, expressions, boundary
  /// conditions and mesh functions. It can plot data wrapped in classes
  /// conforming to the GenericVTKPlottable interface.
  /// The plotter has several parameters that the user can set and adjust to
  /// affect the appearance and behavior of the plot.
  ///
  /// A plotter can be created and used in the following way:
  ///
  ///   Mesh mesh = ...;
  ///   VTKPlotter plotter(mesh);
  ///   plotter.plot();
  ///
  /// Parameters can be adjusted at any time and will take effect on the next
  /// call to the plot() method. The following parameters exist:
  ///
  /// ============= ============ =============== =================================
  ///  Name          Value type   Default value              Description
  /// ============= ============ =============== =================================
  ///  mode           String        "auto"        For vector valued functions,
  ///                                             this parameter may be set to
  ///                                             "warp" to enable vector warping
  ///                                             visualization
  ///  interactive    Boolean     False           Enable/disable interactive mode
  ///                                             for the rendering window.
  ///                                             For repeated plots of the same
  ///                                             object (animated plots), this
  ///                                             parameter must be set to false
  ///  wireframe      Boolean     True for        Enable/disable wireframe
  ///                             meshes, else    rendering of the object
  ///                             false
  ///  title          String      Inherited       The title of the rendering
  ///                             from the        window
  ///                             name/label of
  ///                             the object
  ///  scale          Double      1.0             Adjusts the scaling of the
  ///                                             warping and glyphs
  ///  scalarbar      Boolean     False for       Hide/show the colormapping bar
  ///                             meshes, else
  ///                             true
  ///  rescale        Boolean     False           Enable/disable recomputation
  ///                                             of the scalar to color mapping
  ///                                             on every iteration when performing
  ///                                             repeated/animated plots of the same
  ///                                             data
  ///  prefix         String      "dolfin_plot_"  Filename prefix used when
  ///                                             saving plots to file in
  ///                                             interactive mode. An integer
  ///                                             counter is appended after the
  ///                                             prefix.
  ///  helptext       Boolean     True            Enable/disable the hover-over
  ///                                             help-text in interactive
  ///                                             mode
  ///  window_width   Integer     600             The width of the plotting window
  ///                                             in pixels
  ///  window_height  Integer     400             The height of the plotting window
  ///                                             in pixels
  /// ============= ============ =============== =================================
  ///
  /// The default visualization mode for the different plot types are as follows:
  ///
  /// =========================  ============================ ===================
  ///  Plot type                  Default visualization mode   Alternatives
  /// =========================  ============================ ===================
  ///  Meshes                     Wireframe rendering           None
  ///  2D scalar functions        Scalar warping                None
  ///  3D scalar functions        Color mapping                 None
  ///  2D/3D vector functions     Glyphs (vector arrows)        Vector warping
  /// =========================  ============================ ===================
  ///
  /// Expressions and boundary conditions are also visualized according to the
  /// above table.

  class VTKPlotter : public Variable
  {
  public:

    /// Create plotter for a mesh
    explicit VTKPlotter(boost::shared_ptr<const Mesh> mesh);

    /// Create plotter for a function
    explicit VTKPlotter(boost::shared_ptr<const Function> function);

    /// Create plotter for an expression
    explicit VTKPlotter(boost::shared_ptr<const ExpressionWrapper> expression);

    /// Create plotter for an expression
    explicit VTKPlotter(boost::shared_ptr<const Expression> expression,
                        boost::shared_ptr<const Mesh> mesh);

    /// Create plotter for Dirichlet B.C.
    explicit VTKPlotter(boost::shared_ptr<const DirichletBC> bc);

    /// Create plotter for an uint valued mesh function
    explicit VTKPlotter(boost::shared_ptr<const MeshFunction<unsigned int> > mesh_function);

    /// Create plotter for an intr valued mesh function
    explicit VTKPlotter(boost::shared_ptr<const MeshFunction<int> > mesh_function);

    /// Create plotter for a double valued mesh function
    explicit VTKPlotter(boost::shared_ptr<const MeshFunction<double> > mesh_function);

    /// Create plotter for a boolean valued mesh function
    explicit VTKPlotter(boost::shared_ptr<const MeshFunction<bool> > mesh_function);

    /// Destructor
    ~VTKPlotter();

    /// Default parameter values
    static Parameters default_parameters()
    {
      Parameters p("vtk_plotter");
      p.add("mode", "auto");
      p.add("interactive", false);
      p.add("wireframe", false);
      p.add("title", "Plot");
      p.add("scale", 1.0);
      p.add("scalarbar", true);
      p.add("autorange", true);
      p.add("range_min", 0.0);
      p.add("range_max", 1.0);
      p.add("rescale", false);
      p.add("prefix", "dolfin_plot_");
      p.add("helptext", true);
      p.add("window_width", 600);
      p.add("window_height", 400);
      return p;
    }

    /// Default parameter values for mesh plotting
    static Parameters default_mesh_parameters()
    {
      Parameters p = default_parameters();
      p["wireframe"] = true;
      p["scalarbar"] = false;
      return p;
    }

    // This function should be private, but is available
    // to keep backward compatibilty with Viper
    // Update all VTK structures
    void update();

    // These functions are kept for backward compatibility with Viper
    // TODO: Clean up this and deprecate these functions.
    void update(boost::shared_ptr<const Mesh> mesh);
    void update(boost::shared_ptr<const Function> function);
    void update(boost::shared_ptr<const ExpressionWrapper> expression);
    void update(boost::shared_ptr<const Expression> expression, boost::shared_ptr<const Mesh> mesh);
    void update(boost::shared_ptr<const DirichletBC> bc);
    void update(boost::shared_ptr<const MeshFunction<unsigned int> > mesh_function);
    void update(boost::shared_ptr<const MeshFunction<int> > mesh_function);
    void update(boost::shared_ptr<const MeshFunction<double> > mesh_function);
    void update(boost::shared_ptr<const MeshFunction<bool> > mesh_function);

    /// Plot the object
    void plot();

    /// Make the current plot interactive
    void interactive(bool enter_eventloop = true);

    void start_eventloop();

    /// Save plot to PNG file (file suffix appended automatically)
    void write_png(std::string filename);

    /// Get size of the plot window
    void get_window_size(int& width, int& height);

    /// Set the position of the plot window on the screen
    void set_window_position(int x, int y);

    /// Return unique ID of the object to plot
    uint id() const
    { return _id; }

    /// Camera control
    void azimuth(double angle);
    void elevate(double angle);
    void dolly(double value);
    void set_viewangle(double angle);

    // Set the range of the color table
    void set_min_max(double min, double max);

    void add_polygon(const Array<double>& points);

    // Make all plot windows interactive
    static void all_interactive();

  private:

    // The pool of plotter objects. Objects register
    // themselves in the list when created and remove themselves when
    // destroyed. 
    // Used when calling interactive() (which should have effect on
    // all plot windows)
    static boost::shared_ptr<std::list<VTKPlotter*> > all_plotters;


    // Initialization common to all constructors.
    // Setup all pipeline objects and connect them.
    void init();

    // Set the title parameter from the name and label of the Variable to plot
    void set_title(const std::string& name, const std::string& label);

    // Return the hover-over help text
    std::string get_helptext();

    // Keypress callback
    void keypressCallback(vtkObject* caller,
                          long unsigned int eventId,
                          void* callData);

    // The plottable object (plot data wrapper)
    boost::shared_ptr<GenericVTKPlottable> _plottable;

    boost::scoped_ptr<PrivateVTKPipeline> vtk_pipeline;

    // The number of plotted frames
    uint _frame_counter;

    // The unique ID (inherited from Variable) for the object to plot
    uint _id;

    // Flag to set the state of vertex labels
    bool _toggle_vertex_labels;

    // Counter for the automatically named hardcopies
    static int hardcopy_counter;

    bool no_plot;

    // Keep a shared_ptr to the list of plotter to ensure that the
    // list is not destroyed before the last VTKPlotter object is
    // destroyed.
    boost::shared_ptr<std::list<VTKPlotter*> > all_plotters_local_copy;
  };

}

#endif
