// Copyright (C) 2019 Chris Richardson
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "log.h"

#include <gflags/gflags.h>
#include <vector>

namespace dolfinx
{
// Global name
std::string logname;
} // namespace dolfinx

//-----------------------------------------------------------------------------
void dolfinx::init_logging(int argc, char* argv[])
{
  // Copy program name to static memory
  logname = std::string(argv[0]);
  google::InitGoogleLogging(logname.c_str());

  // Set LogLevel to ERROR, only fatal errors logged by default
  FLAGS_minloglevel = google::GLOG_ERROR;

  // Filter command line arguments, so only those relevant to glog get passed
  // through
  std::vector<char*> argv_filtered;
  std::vector<std::string> allowed_flags
      = {"--logtostderr",      "--stderrthreshold", "--colorlogtostderr",
         "--colorlogtostdout", "--logtostdout",     "--logbuflevel",
         "--vmodule",          "--minloglevel",     "--log_dir",
         "--max_log_size"};
  argv_filtered.push_back(argv[0]);
  for (int i = 1; i < argc; ++i)
  {
    std::string arg_i(argv[i]);
    for (std::string& c : allowed_flags)
    {
      if (c.size() <= arg_i.size() and c.compare(0, c.size(), arg_i))
        argv_filtered.push_back(argv[i]);
    }
  }
  int argc_filtered = argv_filtered.size();
  char** argv_ptr = argv_filtered.data();
  gflags::ParseCommandLineFlags(&argc_filtered, &argv_ptr, false);
}
//-----------------------------------------------------------------------------
