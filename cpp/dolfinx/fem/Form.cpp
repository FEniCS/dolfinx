// Copyright (C) 2025 Jørgen S. Dokken
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "Form.h"

bool dolfinx::fem::operator<(const dolfinx::fem::IntegralType& self,
                             const dolfinx::fem::IntegralType& other)
{
  // First, compare codim
  if (self.codim < other.codim)
  {
    return true;
  }
  // If codims are equal, compare num_cells
  if (self.codim == other.codim)
  {
    return self.num_cells < other.num_cells;
  }
  // Otherwise, this object is not less than 'other'
  return false;
}
