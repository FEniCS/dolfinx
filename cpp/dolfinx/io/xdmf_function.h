// Copyright (C) 2012-2018 Chris N. Richardson and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "HDF5File.h"

namespace pugi
{
class xml_document;
} // namespace pugi

namespace dolfinx
{
namespace function
{
class Function;
}

namespace io
{
/// Low-level methods for reading XDMF files
namespace xdmf_function
{

/// TODO
void write(const function::Function& u, double t, int counter,
           pugi::xml_document& xml_doc, hid_t h5_id);

} // namespace xdmf_function
} // namespace io
} // namespace dolfinx
