// Copyright (C) 2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-05-02
// Last changed: 2007-06-13

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

    // Write script file
    fprintf(script_file, "import os, sys\n");
    fprintf(script_file, "try:\n");
    fprintf(script_file, "    from dolfin import *\n\n");
    if (mesh) {
      fprintf(script_file, "    mesh = Mesh('%s')\n", data_name.c_str());
      fprintf(script_file, "    f = MeshFunction('%s', mesh, '%s')\n\n", type.c_str(), data_name.c_str());
    }
    else
      fprintf(script_file, "    f = %s(r'%s')\n\n", type.c_str(), data_name.c_str());
    fprintf(script_file, "    os.remove('%s')\n", data_name.c_str());
    fprintf(script_file, "    os.remove('%s')\n\n", script_name.c_str());
    if (mode == "")
      fprintf(script_file, "    plot(f)\n");
    else
      fprintf(script_file, "    plot(f, mode='%s')\n", mode.c_str());
    fprintf(script_file, "    interactive()\n");
    fprintf(script_file, "except:\n");
    fprintf(script_file, "    sys.exit(1)\n");
    fclose(script_file);
    
    message("Plotting %s, press q to continue...", type.c_str());

    // Run script
#ifdef __WIN32__
    std::string command = "python " + script_name + " > nul";
#else
    std::string command = "python " + script_name + " > /dev/null";
#endif
    if ( system(command.c_str()) != 0 )
      message("Unable to plot (PyDOLFIN or Viper plotter not available).");
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
  template<class T> void plot(MeshFunction<T>& f, std::string type, std::string mode)
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
void dolfin::plot(Function& f, std::string mode)
{
  plot(f, "Function", mode);
}
//-----------------------------------------------------------------------------
void dolfin::plot(Mesh& mesh, std::string mode)
{
  plot(mesh, "Mesh", mode);
}
//-----------------------------------------------------------------------------
void dolfin::plot(MeshFunction<uint>& f, std::string mode)
{
  plot(f, "uint", mode);
}
//-----------------------------------------------------------------------------
void dolfin::plot(MeshFunction<double>& f, std::string mode)
{
  plot(f, "real", mode);
}
//-----------------------------------------------------------------------------
void dolfin::plot(MeshFunction<bool>& f, std::string mode)
{
  plot(f, "bool", mode);
}
//-----------------------------------------------------------------------------
