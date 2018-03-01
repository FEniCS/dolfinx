// Copyright (C) 2009 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "GlobalParameters.h"
#include <cstdlib>
#include <dolfin/log/LogStream.h>
#include <dolfin/log/log.h>
#include <fstream>
#include <iostream>

using namespace dolfin;
using namespace dolfin::parameter;

/// The global parameter database
GlobalParameters dolfin::parameter::parameters;

//-----------------------------------------------------------------------------
GlobalParameters::GlobalParameters() : Parameters("dolfin")
{
  // Set default parameter values
  *static_cast<Parameters*>(this) = default_parameters();
}
//-----------------------------------------------------------------------------
GlobalParameters::~GlobalParameters()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void GlobalParameters::parse(int argc, char* argv[])
{
  log::log(TRACE, "Parsing command-line arguments.");

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
      {
        log::dolfin_error("GlobalParameters.cpp", "parse command-line options",
                     "Illegal command-line options");
      }
    }
  }

  // Copy to argv lists
  char** argv_dolfin = new char*[args_dolfin.size()];
  for (std::size_t i = 0; i < args_dolfin.size(); ++i)
  {
    argv_dolfin[i] = new char[args_dolfin[i].size() + 1];
    sprintf(argv_dolfin[i], "%s", args_dolfin[i].c_str());
  }
  char** argv_petsc = new char*[args_petsc.size()];
  for (std::size_t i = 0; i < args_petsc.size(); ++i)
  {
    argv_petsc[i] = new char[args_petsc[i].size() + 1];
    sprintf(argv_petsc[i], "%s", args_petsc[i].c_str());
  }

  // Debugging
  const bool debug = false;
  if (debug)
  {
    cout << "DOLFIN args:";
    for (std::size_t i = 0; i < args_dolfin.size(); i++)
      cout << " " << args_dolfin[i];
    cout << endl;
    cout << "PETSc args: ";
    for (std::size_t i = 0; i < args_petsc.size(); i++)
      cout << " " << args_petsc[i];
    cout << endl;
  }

  // Parse DOLFIN and PETSc options
  parse_common(args_dolfin.size(), argv_dolfin);
  parse_petsc(args_petsc.size(), argv_petsc);

  // Cleanup
  for (std::size_t i = 0; i < args_dolfin.size(); ++i)
    delete[] argv_dolfin[i];
  for (std::size_t i = 0; i < args_petsc.size(); ++i)
    delete[] argv_petsc[i];
  delete[] argv_dolfin;
  delete[] argv_petsc;
}
//-----------------------------------------------------------------------------
