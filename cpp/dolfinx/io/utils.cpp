// Copyright (C) 2020 Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "utils.h"
#include <boost/filesystem.hpp>

using namespace dolfinx;

//-----------------------------------------------------------------------------
std::string io::get_filename(const std::string& fullname)
{
  const boost::filesystem::path p(fullname);
  return p.filename().string();
}
//-----------------------------------------------------------------------------
