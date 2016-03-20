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
// Modified by Joachim B Haga 2012
//
// First added:  2012-05-23
// Last changed: 2012-11-13

#ifndef __VTK_PLOTTER_H
#define __VTK_PLOTTER_H

#include <list>
#include <memory>
#include <string>

#include <dolfin/common/Variable.h>
#include <dolfin/parameter/Parameters.h>

class vtkObject;

namespace dolfin
{

  // Forward declarations
  class Expression;
  class Mesh;
  class GenericVTKPlottable;
  class VTKWindowOutputStage;
  template<typename T> class Array;

  /// This class enables visualization of various DOLFIN entities.  It
  /// supports visualization of meshes, functions, expressions,
  /// boundary conditions and mesh functions. It can plot data wrapped
  /// in classes conforming to the GenericVTKPlottable interface.  The
  /// plotter has several parameters that the user can set and adjust
  /// to affect the appearance and behavior of the plot.
  ///
  /// A plotter can be created and used in the following way:
  ///
  ///   Mesh mesh = ...;
  ///   VTKPlotter plotter(mesh);
  ///   plotter.plot();
  ///
  /// Parameters can be adjusted at any time and will take effect on
  /// the next call to the plot() method. The following parameters
  /// exist:
  ///
  /// ============== ============ ================ ====================================
  ///  Name           Value type   Default value              Description
  /// ============== ============ ================ ====================================
  ///  mode            String        "auto"         For vector valued functions,
  ///                                               this parameter may be set to
  ///                                               "glyphs" or "displacement".
  ///                                               Scalars may be set to "warp" in
  ///                                               2D only. A value of "color" is
  ///                                               valid in all cases; for vectors,
  ///                                               the norms are used. See below for
  ///                                               a summary of default modes,
  ///                                               used when set to "auto".
  ///  interactive     Boolean     False            Enable/disable interactive mode
  ///                                               for the rendering window.
  ///                                               For repeated plots of the same
  ///                                               object (animated plots), this
  ///                                               parameter should be set to false.
  ///  wireframe       Boolean     True for         Enable/disable wireframe
  ///                              meshes, else     rendering of the object.
  ///                              false
  ///  title           String      Inherited        The title of the rendering
  ///                              from the         window
  ///                              name/label of
  ///                              the object
  ///  scale           Double      1.0              Adjusts the scaling of the
  ///                                               warping and glyphs
  ///  scalarbar       Boolean     False for        Hide/show the colormapping bar
  ///                              meshes, else
  ///                              true
  ///  axes            Boolean     False            Show X-Y-Z axes.
  ///
  ///  rescale         Boolean     True             Enable/disable recomputation
  ///                                               of the scalar to color mapping
  ///                                               on every iteration when performing
  ///                                               repeated/animated plots of the same
  ///                                               data. If both range_min and
  ///                                               range_max are set, this parameter
  ///                                               is ignored.
  ///  range_min       Double                       Set lower range of data values.
  ///                                               Disables automatic (re-)computation
  ///                                               of the lower range.
  ///  range_max       Double                       Set upper range of data values.
  ///                                               Disables automatic (re-)computation
  ///                                               of the upper range.
  ///  elevate         Double      -65.0 for 2D     Set camera elevation.
  ///                              warped scalars,
  ///                              0.0 otherwise
  ///  prefix          String      "dolfin_plot_"   Filename prefix used when
  ///                                               saving plots to file in
  ///                                               interactive mode. An integer
  ///                                               counter is appended after the
  ///                                               prefix.
  ///  helptext        Boolean     True             Enable/disable the hover-over
  ///                                               help-text in interactive
  ///                                               mode
  ///  window_width    Integer     600              The width of the plotting window
  ///                                               in pixels
  ///  window_height   Integer     400              The height of the plotting window
  ///                                               in pixels
  ///  tile_windows    Boolean     True             Automatically tile plot windows.
  ///
  ///  key             String                       Key (id) of the plot window, used to
  ///                                               decide if a new plotter should be
  ///                                               created or a current one updated
  ///                                               when called through the static
  ///                                               plot() interface (in plot.h).
  ///                                               If not set, the object's unique
  ///                                               id (Variable::id) is used.
  ///  input_keys      String      ""               Synthesize key presses, as if these
  ///                                               keys are pressed by the user in
  ///                                               the plot window.
  ///                                               For example: "ww++m" shows the data
  ///                                               as large points on a wireframe
  ///                                               mesh.
  ///  hide_above      Double                       If either of these are set, scalar
  ///  hide_below      Double                       values above or below will not be
  ///                                               shown in the plot.
  /// ============== ============ ================ ====================================
  ///
  /// The default visualization mode for the different plot types are as follows:
  ///
  /// =========================  ============================ =====================
  ///  Plot type                  Default visualization mode   Alternatives
  /// =========================  ============================ =====================
  ///  Meshes                     Wireframe rendering          None
  ///  2D scalar functions        Scalar warping               Color mapping
  ///  3D scalar functions        Color mapping                None
  ///  2D/3D vector functions     Glyphs (vector arrows)       Displacements,
  ///                                                          Color mapping (norm)
  /// =========================  ============================ =====================
  ///
  /// Expressions and boundary conditions are also visualized
  /// according to the above table.

  class VTKPlotter : public Variable
  {
  public:

    /// Create plotter for a variable. If a widget is supplied, this
    /// widget will be used for drawing, instead of a new top-level
    /// widget. Ownership is transferred.
    VTKPlotter(std::shared_ptr<const Variable>);

    /// Create plotter for an Expression with associated Mesh. If a
    /// widget is supplied, this widget will be used for drawing,
    /// instead of a new top-level widget. Ownership is transferred.
    VTKPlotter(std::shared_ptr<const Expression> expression,
               std::shared_ptr<const Mesh> mesh);

    /// Destructor
    virtual ~VTKPlotter();

    /// Default parameter values
    static Parameters default_parameters()
    {
      std::set<std::string> allowed_modes;
      allowed_modes.insert("auto");
      allowed_modes.insert("displacement");
      allowed_modes.insert("warp");
      allowed_modes.insert("glyphs");
      allowed_modes.insert("color");

      Parameters p("vtk_plotter");
      p.add("mode", "auto", allowed_modes);
      p.add("interactive", false);
      p.add("wireframe", false);
      p.add("title", "Plot");
      p.add("scale", 1.0);
      p.add("scalarbar", true);
      p.add("axes", false);
      p.add<double>("elevate");
      p.add<double>("range_min");
      p.add<double>("range_max");
      p.add("rescale", true);
      p.add("prefix", "dolfin_plot_");
      p.add("helptext", true);
      p.add("window_width",  600, /*min*/ 50, /*max*/ 5000);
      p.add("window_height", 400, /*min*/ 50, /*max*/ 5000);
      p.add("tile_windows", true);

      p.add<std::string>("key");
      p.add<double>("hide_below"); // Undocumented on purpose, may be removed
      p.add<double>("hide_above"); // Undocumented on purpose, may be removed
      p.add<std::string>("input_keys");
      return p;
    }

    bool is_compatible(std::shared_ptr<const Variable> variable) const;

    /// Plot the object
    void plot(std::shared_ptr<const Variable>
              variable=std::shared_ptr<const Variable>());

    // FIXME: Deprecated? What should it do?
    void update(std::shared_ptr<const Variable>
                variable=std::shared_ptr<const Variable>())
    {
      warning("VTKPlotter::update is deprecated, use ::plot instead");
      plot(variable);
    }

    /// Make the current plot interactive
    void interactive(bool enter_eventloop = true);

    /// Save plot to PNG file (file suffix appended automatically, filename
    /// optionally built from prefix)
    void write_png(std::string filename="");

    /// Save plot to PDF file (file suffix appended automatically, filename
    /// optionally built from prefix)
    void write_pdf(std::string filename="");

    /// Return key (i.e., plotter id) of the object to plot
    const std::string& key() const;

    /// Set the key (plotter id)
    void set_key(std::string key);

    /// Return default key (plotter id) of a Variable (object to plot).
    static std::string to_key(const Variable &var);

    /// Camera control
    void azimuth(double angle);
    void elevate(double angle);
    void dolly(double value);
    void zoom(double zoomfactor);
    void set_viewangle(double angle);

    // Set the range of the color table
    void set_min_max(double min, double max);

    void add_polygon(const Array<double>& points);

    /// Make all plot windows interactive. If really is set, the
    /// interactive mode is entered even if 'Q' has been pressed.
    static void all_interactive(bool really=false);

    enum class Modifiers : int
    {
      // Zero low byte, so that a char can be added
      SHIFT    = 0x100,
      ALT      = 0x200,
      CONTROL  = 0x400
    };

    // Called (from within VTKWindowOutputStage) when a key is
    // pressed. Public, but intended for internal (and subclass)
    // use. Returns true if the keypress is handled.
    virtual bool key_pressed(int modifiers, char key, std::string keysym);

  protected:

    void update_pipeline(std::shared_ptr<const Variable>
                         variable=std::shared_ptr<const Variable>());

    // The pool of plotter objects. Objects register themselves in the
    // list when created and remove themselves when destroyed.  Used
    // when calling interactive() (which should have effect on all
    // plot windows)
    static std::shared_ptr<std::list<VTKPlotter*> > active_plotters;

    // Initialization common to all constructors.  Setup all pipeline
    // objects and connect them.
    void init();

    // Has init been called
    bool _initialized;

    // Rescales ranges and glyphs
    void rescale();

    // Set the title parameter from the name and label of the Variable
    // to plot
    void set_title_from(const Variable &variable);

    // Return the hover-over help text
    std::string get_helptext();

    // The plottable object (plot data wrapper)
    std::shared_ptr<GenericVTKPlottable> _plottable;

    // The output stage
    std::unique_ptr<VTKWindowOutputStage> vtk_pipeline;

    // The number of plotted frames
    std::size_t _frame_counter;

    // The window id (derived from Variable::id unless overridden by
    // user)
    std::string _key;

    // Counter for the automatically named hardcopies
    static int hardcopy_counter;

    bool no_plot;

    // Keep a shared_ptr to the list of plotter to ensure that the
    // list is not destroyed before the last VTKPlotter object is
    // destroyed.
    std::shared_ptr<std::list<VTKPlotter*> > active_plotters_local_copy;

    // Usually false, but if true ('Q' keyboard binding) then all
    // event loops are skipped.
    static bool run_to_end;
  };

}

#endif
