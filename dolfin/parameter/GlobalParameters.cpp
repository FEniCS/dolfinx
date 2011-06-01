// Copyright (C) 2009 Anders Logg
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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2009-07-02
// Last changed: 2011-06-01

#include <fstream>
#include <cstdlib>
#include <dolfin/log/dolfin_log.h>
#include <dolfin/io/File.h>
#include <dolfin/la/KrylovSolver.h>
#include "GlobalParameters.h"

using namespace dolfin;

/// The global parameter database
GlobalParameters dolfin::parameters;

//-----------------------------------------------------------------------------
GlobalParameters::GlobalParameters() : Parameters("dolfin")
{
  // Set default parameter values
  *static_cast<Parameters*>(this) = default_parameters();

  // FIXME: Consider adding the default parameter sets for all
  // FIXME: classes as nested parameter sets here.

  // Add nested parameter sets
  this->add(KrylovSolver::default_parameters());

  // Search paths to parameter files in order of increasing priority
  std::vector<std::string> parameter_files;
#ifdef _WIN32
  std::string home_directory(std::getenv("USERPROFILE"));
  parameter_files.push_back(home_directory + "\\.dolfin\\parameters.xml.gz");
  parameter_files.push_back(home_directory + "\\.dolfin\\parameters.xml");
#else
  std::string home_directory(std::getenv("HOME"));
  parameter_files.push_back(home_directory + "/.dolfin/parameters.xml.gz");
  parameter_files.push_back(home_directory + "/.dolfin/parameters.xml");
#endif
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
void GlobalParameters::parse(int argc, char* argv[])
{
  log(TRACE, "Parsing command-line arguments.");

  // Extract DOLFIN and PETSc arguments
  std::vector<std::string> args_dolfin;
  std::vector<std::string> args_petsc;
  std::vector<std::string>* current = 0;
  args_dolfin.push_back(argv[0]);
  args_petsc.push_back(argv[0]);
  for (int i = 1; i < argc; ++i)
  {
    std::string arg(argv[i]);

    if (arg.size() > 2 && arg.substr(0, 2) == "--")
    {
      if (arg.size() > 8 && arg.substr(0, 8) == "--petsc.")
      {
        current = &args_petsc;
        current->push_back("-" + arg.substr(8));
      }
      else
      {
        current = &args_dolfin;
        current->push_back(arg);
      }
    }
    else
    {
      if (current)
        current->push_back(arg);
      else
        error("Illegal command-line options.");
    }
  }

  // Copy to argv lists
  char** argv_dolfin = new char*[args_dolfin.size()];
  for (uint i = 0; i < args_dolfin.size(); ++i)
  {
    argv_dolfin[i] = new char[args_dolfin[i].size() + 1];
    sprintf(argv_dolfin[i], "%s", args_dolfin[i].c_str());
  }
  char** argv_petsc = new char*[args_petsc.size()];
  for (uint i = 0; i < args_petsc.size(); ++i)
  {
    argv_petsc[i] = new char[args_petsc[i].size() + 1];
    sprintf(argv_petsc[i], "%s", args_petsc[i].c_str());
  }

  // Debugging
  const bool debug = false;
  if (debug)
  {
    cout << "DOLFIN args:";
    for (uint i = 0; i < args_dolfin.size(); i++)
      cout << " " << args_dolfin[i];
    cout << endl;
    cout << "PETSc args: ";
    for (uint i = 0; i < args_petsc.size(); i++)
      cout << " " << args_petsc[i];
    cout << endl;
  }

  // Parse DOLFIN and PETSc options
  parse_common(args_dolfin.size(), argv_dolfin);
  parse_petsc(args_petsc.size(), argv_petsc);

  // Cleanup
  for (uint i = 0; i < args_dolfin.size(); ++i)
    delete [] argv_dolfin[i];
  for (uint i = 0; i < args_petsc.size(); ++i)
    delete [] argv_petsc[i];
  delete [] argv_dolfin;
  delete [] argv_petsc;
}
//-----------------------------------------------------------------------------
