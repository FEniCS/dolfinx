// Copyright (C) 2007 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2007-05-02
// Last changed: 2007-05-02

#include <stdio.h>
#include <stdlib.h>

#include <dolfin/dolfin_log.h>
#include <dolfin/Function.h>
#include <dolfin/Mesh.h>
#include <dolfin/File.h>
#include <dolfin/plot.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
void dolfin::plot(Function& f, std::string mode)
{
  dolfin_info("Plotting function, press q to continue...");

  // Open temporary script file
  std::string script_name = std::string(tmpnam(0)) + ".py";
  FILE* script_file = fopen(script_name.c_str(), "w");

  // Save data to temporary file
  std::string data_name = std::string(tmpnam(0)) + ".xml";
  dolfin_log(false);
  File file(data_name);
  file << f;
  dolfin_log(true);

  // Write script file
  fprintf(script_file, "try:\n");
  fprintf(script_file, "    from dolfin import *\n\n");
  fprintf(script_file, "    u = Function(\"%s\")\n", data_name.c_str());
  fprintf(script_file, "    plot(u)\n");
  fprintf(script_file, "except:\n");
  fprintf(script_file, "    exit(1)\n");
  fclose(script_file);
  
  // Run script
  std::string command = "python " + script_name + " > /dev/null";
  if ( system(command.c_str()) != 0 )
    dolfin_info("Unable to plot function (PyDOLFIN or Viper plotter not available).");
}
//-----------------------------------------------------------------------------
void dolfin::plot(Mesh& mesh)
{
  dolfin_info("Plotting mesh, press q to continue...");

  // Open temporary script file
  std::string script_name = std::string(tmpnam(0)) + ".py";
  FILE* script_file = fopen(script_name.c_str(), "w");

  // Save data to temporary file
  std::string data_name = std::string(tmpnam(0)) + ".xml";
  dolfin_log(false);
  File file(data_name);
  file << mesh;
  dolfin_log(true);

  // Write script file
  fprintf(script_file, "try:\n");
  fprintf(script_file, "    from dolfin import *\n\n");
  fprintf(script_file, "    mesh = Function(\"%s\")\n", data_name.c_str());
  fprintf(script_file, "    plot(mesh)\n");
  fprintf(script_file, "except:\n");
  fprintf(script_file, "    exit(1)\n");
  fclose(script_file);
  
  // Run script
  std::string command = "python " + script_name + " > /dev/null";
  if ( system(command.c_str()) != 0 )
    dolfin_info("Unable to plot mesh (PyDOLFIN or Viper plotter not available).");
}
//-----------------------------------------------------------------------------
