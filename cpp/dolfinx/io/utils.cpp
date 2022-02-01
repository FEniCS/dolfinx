// Copyright (C) 2020 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "utils.h"
#include <filesystem>

using namespace dolfinx;

//-----------------------------------------------------------------------------
std::string io::get_filename(const std::string& fullname)
{
  const std::filesystem::path p(fullname);
  return std::string(p.filename().c_str());
}
//-----------------------------------------------------------------------------
