// Copyright (C) 2007 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2007-05-02
// Last changed: 2007-05-08

#include <stdio.h>
#include <stdlib.h>

#include <dolfin/dolfin_log.h>
#include <dolfin/Function.h>
#include <dolfin/Mesh.h>
#include <dolfin/File.h>
#include <dolfin/plot.h>

using namespace dolfin;

namespace dolfin
{

  // Plotting of objects (functions and meshes)
  template<class T> void plot(T& t, std::string class_name, std::string mode)
  {
    dolfin_info("Plotting %s, press q to continue...", class_name.c_str());
    
    // Open temporary script file
    std::string script_name = std::string(tmpnam(0)) + ".py";
    FILE* script_file = fopen(script_name.c_str(), "w");
    
    // Save data to temporary file
    std::string data_name = std::string(tmpnam(0)) + ".xml";
    dolfin_log(false);
    File file(data_name);
    file << t;
    dolfin_log(true);
    
    // Write script file
    fprintf(script_file, "try:\n");
    fprintf(script_file, "    from dolfin import *\n\n");
    fprintf(script_file, "    object = %s(\"%s\")\n", class_name.c_str(), data_name.c_str());
    if (mode == "")
      fprintf(script_file, "    plot(object)\n");
    else
      fprintf(script_file, "    plot(object, mode=\"%s\")\n", mode.c_str());
    fprintf(script_file, "except:\n");
    fprintf(script_file, "    exit(1)\n");
    fclose(script_file);
    
    // Run script
    std::string command = "python " + script_name + " > /dev/null";
    if ( system(command.c_str()) != 0 )
      dolfin_info("Unable to plot (PyDOLFIN or Viper plotter not available).");
  }

  // Plotting of mesh functions
  template<class T> void plot(MeshFunction<T>& f, std::string type, std::string mode)
  {
    dolfin_info("Plotting MeshFunction, press q to continue...");
    
    // Open temporary script file
    std::string script_name = std::string(tmpnam(0)) + ".py";
    FILE* script_file = fopen(script_name.c_str(), "w");
    
    // Save data to temporary file
    std::string data_name = std::string(tmpnam(0)) + ".xml";
    dolfin_log(false);
    File file(data_name);
    file << f.mesh();
    file << f;
    dolfin_log(true);
    
    // Write script file
    fprintf(script_file, "try:\n");
    fprintf(script_file, "    from dolfin import *\n\n");
    fprintf(script_file, "    mesh = Mesh(\"%s\")\n", data_name.c_str());
    fprintf(script_file, "    f = MeshFunction(\"%s\", mesh, \"%s\")\n", type.c_str(), data_name.c_str());
    if (mode == "")
      fprintf(script_file, "    plot(f)\n");
    else
      fprintf(script_file, "    plot(f, mode=\"%s\")\n", mode.c_str());
    fprintf(script_file, "except:\n");
    fprintf(script_file, "    exit(1)\n");
    fclose(script_file);
    
    // Run script
    std::string command = "python " + script_name; // + " > /dev/null";
    if ( system(command.c_str()) != 0 )
      dolfin_info("Unable to plot (PyDOLFIN or Viper plotter not available).");
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
void dolfin::plot(MeshFunction<real>& f, std::string mode)
{
  plot(f, "real", mode);
}
//-----------------------------------------------------------------------------
void dolfin::plot(MeshFunction<bool>& f, std::string mode)
{
  plot(f, "bool", mode);
}
//-----------------------------------------------------------------------------
