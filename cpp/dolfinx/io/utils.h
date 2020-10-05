// Copyright (C) 2020 Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <string>

/// Tools for supporting IO
namespace dolfinx::io
{

/// Get filename from a fully qualified path and filename
/// @param[in] fullname Full file path and name
/// @return The filename (without path)
std::string get_filename(const std::string& fullname);

} // namespace dolfinx::io
