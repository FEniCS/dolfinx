// Copyright (C) 2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-07-02
// Last changed: 2009-09-06

#include <fstream>
#include <cstdlib>
#include <dolfin/log/dolfin_log.h>
#include <dolfin/io/File.h>
#include "GlobalParameters.h"

using namespace dolfin;

/// The global parameter database
GlobalParameters dolfin::parameters;

//-----------------------------------------------------------------------------
GlobalParameters::GlobalParameters() : Parameters("dolfin")
{
  // Set default parameter values
  *static_cast<Parameters*>(this) = default_parameters();

  // Search paths to parameter files in order of increasing priority
  std::vector<std::string> parameter_files;
  std::string home_directory(std::getenv("HOME"));
  parameter_files.push_back(home_directory + "/.dolfin/parameters.xml.gz");
  parameter_files.push_back(home_directory + "/.dolfin/parameters.xml");
  parameter_files.push_back("parameters.xml.gz");
  parameter_files.push_back("parameters.xml");

  // Try reading parameters from files
  for (uint i = 0; i < parameter_files.size(); ++i)
  {
    // Check if file exists
    std::ifstream f;
    f.open(parameter_files[i].c_str());
    if (!f.is_open())
      continue;
    f.close();

    // Note: Cannot use DOLFIN log system here since it's not initialized
    std::cout << "Reading DOLFIN parameters from file \"" << parameter_files[i] << "\"." << std::endl;

    // Read parameters from file and update global parameters
    File file(parameter_files[i]);
    Parameters p;
    file >> p;
    this->update(p);
  }
}
//-----------------------------------------------------------------------------
GlobalParameters::~GlobalParameters()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
