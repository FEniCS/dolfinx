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

  // Plotting of objects (functions and meshes)
  template<class T> void plot(T& t, std::string class_name, std::string mode)
  {
    message("Plotting %s, press q to continue...", class_name.c_str());
    
    // Open temporary script file
#ifdef __WIN32__
    char buffer[MAX_PATH];
    std::string tmppath;
    if (GetTempPath(512, buffer) == 0) 
      tmppath = ".";  // use current directory instead
    else
      tmppath = std::string(buffer);
    std::string script_name;
    if (GetTempFileName(tmppath.c_str(), "", 0, buffer) == 0) 
      error("Unable to create temporary plotting script in %s.", tmppath.c_str());
    else
      script_name = std::string(buffer) + ".py";
#else
    std::string script_name = std::string(tmpnam(0)) + ".py";
#endif
    FILE* script_file = fopen(script_name.c_str(), "w");
    
    // Save data to temporary file
#ifdef __WIN32__
    std::string data_name;
    if (GetTempFileName(tmppath.c_str(), "", 0, buffer) == 0)
      error("Unable to create temporary xml file in %s.", tmppath.c_str());
    else
      data_name = std::string(buffer) + ".xml";
#else
    std::string data_name = std::string(tmpnam(0)) + ".xml";
#endif
    dolfin_set("output destination", "silent");
    File file(data_name);
    file << t;
    dolfin_set("output destination", "terminal");
    
    // Write script file
    fprintf(script_file, "exitcode = 0\n");
    fprintf(script_file, "try:\n");
    fprintf(script_file, "    from dolfin import *\n\n");
    fprintf(script_file, "    object = %s(r\"%s\")\n", class_name.c_str(), data_name.c_str());
    if (mode == "")
      fprintf(script_file, "    plot(object)\n");
    else
      fprintf(script_file, "    plot(object, mode=\"%s\")\n", mode.c_str());
    fprintf(script_file, "    interactive()\n");
    fprintf(script_file, "except:\n");
    fprintf(script_file, "    exitcode = 1\n");
    fprintf(script_file, "import os, sys\n");
    fprintf(script_file, "os.remove('%s')\n", data_name.c_str());
    fprintf(script_file, "os.remove('%s')\n", script_name.c_str());
    fprintf(script_file, "sys.exit(exitcode)\n");
    fclose(script_file);
    
    // Run script
#ifdef __WIN32__
    std::string command = "python " + script_name + " > nul";
#else
    std::string command = "python " + script_name + " > /dev/null";
#endif
    if ( system(command.c_str()) != 0 )
      message("Unable to plot (PyDOLFIN or Viper plotter not available).");
  }

  // Plotting of mesh functions
  template<class T> void plot(MeshFunction<T>& f, std::string type, std::string mode)
  {
    message("Plotting MeshFunction, press q to continue...");
    
    // Open temporary script file
    std::string script_name = std::string(tmpnam(0)) + ".py";
    FILE* script_file = fopen(script_name.c_str(), "w");
    
    // Save data to temporary file
    std::string data_name = std::string(tmpnam(0)) + ".xml";
    dolfin_set("output destination", "silent");
    File file(data_name);
    file << f.mesh();
    file << f;
    dolfin_set("output destination", "terminal");
    
    // Write script file
    fprintf(script_file, "try:\n");
    fprintf(script_file, "    from dolfin import *\n\n");
    fprintf(script_file, "    mesh = Mesh(\"%s\")\n", data_name.c_str());
    fprintf(script_file, "    f = MeshFunction(\"%s\", mesh, \"%s\")\n", type.c_str(), data_name.c_str());
    if (mode == "")
      fprintf(script_file, "    plot(f)\n");
    else
      fprintf(script_file, "    plot(f, mode=\"%s\")\n", mode.c_str());
    fprintf(script_file, "    interactive()\n");
    fprintf(script_file, "except:\n");
    fprintf(script_file, "    import sys\n");
    fprintf(script_file, "    sys.exit(1)\n");
    fclose(script_file);
    
    // Run script
    std::string command = "python " + script_name; // + " > /dev/null";
    if ( system(command.c_str()) != 0 )
      message("Unable to plot (PyDOLFIN or Viper plotter not available).");
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
