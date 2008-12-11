// Copyright (C) 2007-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Joachim Berdal Haga, 2008.
// Modified by Garth N. Wells, 2008.
//
// First added:  2007-05-02
// Last changed: 2008-12-06

#include <stdio.h>
#include <stdlib.h>

#include <dolfin/log/log.h>
#include <dolfin/parameter/parameters.h>
#include <dolfin/function/Function.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/io/File.h>
#include "plot.h"

#ifdef __WIN32__
#include <windows.h>  // neccessary to create temporary files on Windows
#endif

using namespace dolfin;

namespace dolfin
{
  std::string secure_tmp_filename(std::string ident)
  {
    // Return a temporary file name, pointing to an empty file owned by user
#ifdef __WIN32__
    char buffer[MAX_PATH];
    std::string tmppath;
    if (GetTempPath(512, buffer) == 0) 
      tmppath = ".";  // use current directory instead
    else
      tmppath = std::string(buffer);
    std::string script_name;
    // From doc: "If a unique file name is generated, an empty file is created
    // and the handle to it is released; otherwise, only a file name is
    // generated." As I understand it, the name is guaranteed to be unique as
    // long as the third param is 0. Thus, this should be secure.
    // Only the first three characters of prefix are used.
    if (GetTempFileName(tmppath.c_str(), ident.c_str(), 0, buffer) == 0) 
      error("Unable to create temporary plotting script in %s.", tmppath.c_str());
    return std::string(buffer);
#else
    char buffer[512];
    const char *tmppath;
    if (getenv("TMPDIR"))
      tmppath = getenv("TMPDIR");
    else if (getenv("TEMP"))
      tmppath = getenv("TEMP");
    else if (getenv("TMP"))
      tmppath = getenv("TMP");
    else
      tmppath = "/tmp";

    // "XXXXXX" must be the last 6 chars of template
    snprintf(buffer, sizeof(buffer), "%s/dolfin-%s-XXXXXX", tmppath, ident.c_str());
    // Create file and associated name securely, and close it. Re-open is
    // secure since the file now exists and is owned by user.  If the directory
    // is world-writable and non-sticky, this is not secure, but then the user
    // is to blame (/tmp is normally sticky).
    int fd = mkstemp(buffer);
    if (fd == -1)
      error("Unable to create temporary plotting script in %s.", tmppath);
    close(fd);
    return std::string(buffer);
#endif
  }

  void plot_write_and_run_script(std::string type, std::string data_name, std::string mode, bool mesh)
  {
    std::string script_name = secure_tmp_filename("plot.py");
    FILE* script_file = fopen(script_name.c_str(), "w");

    fprintf(script_file, "is_mesh     = %d\n",   !!mesh);
    fprintf(script_file, "plot_mode   = '%s'\n", mode.c_str());
    fprintf(script_file, "plot_type   = '%s'\n", type.c_str());
    fprintf(script_file, "data_name   = '%s'\n", data_name.c_str());
    fprintf(script_file, "script_name = '%s'\n", script_name.c_str());

    // We really really want to remove the temporary data files, since they can
    // be large. To be extra sure, we do it in two places:
    // - in python, as soon as it is read, in case of crash (this happens a lot,
    // because of dodgy opengl stacks)
    // - in c++ if the command fails (maybe because python is not installed)
    // Deleting the script itself is less vital since it's small. Doing it in
    // python works, but it seems iffy to delete it while it's running. So we do
    // that in c++ only.

#define _ "\n"
    const char *script_body =
      "import os, sys"_
      ""_
      "try:"_
      "    what = 'import DOLFIN python module'"_
      "    from dolfin import *"_
      ""_
      "    what = 'read DOLFIN mesh or function XML file for plotting'"_
      "    if is_mesh:"_
      "        mesh = Mesh(data_name)"_
      "        f = MeshFunction(plot_type, mesh, data_name)"_
      "    else:"_
      "        plot_class = eval(plot_type)  # convert class name -> class"_
      "        f = plot_class(data_name)"_
      "    os.remove(data_name)"_
      ""_
      "    what = 'plot DOLFIN mesh or function'"_
      "    if plot_mode:"_
      "        plot(f, mode=plot_mode)"_
      "    else:"_
      "        plot(f)"_
      "    interactive()"_
      ""_
      "except Exception, e:"_
      "    print 'Failed to %s:' % what"_
      "    print '  %s'          % str(e)"_
      "    sys.exit(1)"_
      ;
#undef _
    fprintf(script_file, "%s\n", script_body);
    fclose(script_file);
    
    message("Plotting %s, press q to continue...", type.c_str());

    // Run script
    std::string command = "python " + script_name;
    if (system(command.c_str()) != 0) {
      message("Unable to plot (DOLFIN Python module or Viper plotter not available).");
      remove(data_name.c_str()); // ignore errors, file may be deleted already
    }
    remove(script_name.c_str());
  }

  template<class T> void plot(T& t, std::string class_name, std::string mode)
  {
    std::string data_name = secure_tmp_filename("plot.xml");
    
    dolfin_set("output destination", "silent");
    File file(data_name);
    file << t;
    dolfin_set("output destination", "terminal");

    plot_write_and_run_script(class_name, data_name, mode, false);
  }

  // Plotting of mesh functions
  template<class T> void plot(const MeshFunction<T>& f, std::string type, std::string mode)
  {
    // Save data to temporary file
    std::string data_name = secure_tmp_filename("plot.xml");

    dolfin_set("output destination", "silent");
    File file(data_name);
    file << f.mesh();
    file << f;
    dolfin_set("output destination", "terminal");

    plot_write_and_run_script(type, data_name, mode, true);
  }

}

//-----------------------------------------------------------------------------
void dolfin::plot(const Function& f, std::string mode)
{
  plot(f, "Function", mode);
}
//-----------------------------------------------------------------------------
void dolfin::plot(const Mesh& mesh, std::string mode)
{
  plot(mesh, "Mesh", mode);
}
//-----------------------------------------------------------------------------
void dolfin::plot(const MeshFunction<uint>& f, std::string mode)
{
  plot(f, "uint", mode);
}
//-----------------------------------------------------------------------------
void dolfin::plot(const MeshFunction<double>& f, std::string mode)
{
  plot(f, "real", mode);
}
//-----------------------------------------------------------------------------
void dolfin::plot(const MeshFunction<bool>& f, std::string mode)
{
  plot(f, "bool", mode);
}
//-----------------------------------------------------------------------------
