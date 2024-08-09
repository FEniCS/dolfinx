// Copyright (C) 2009-2011 Johan Hake
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "defines.h"

//-------------------------------------------------------------------------
std::string dolfinx::version() { return std::string(DOLFINX_VERSION); }
//-------------------------------------------------------------------------
std::string dolfinx::ufcx_signature() { return std::string(UFCX_SIGNATURE); }
//-------------------------------------------------------------------------
std::string dolfinx::git_commit_hash()
{
  return std::string(DOLFINX_GIT_COMMIT_HASH);
}
//-------------------------------------------------------------------------