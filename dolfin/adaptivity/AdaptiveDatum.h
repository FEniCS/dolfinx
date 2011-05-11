// Copyright (C) 2010 Marie E. Rognes
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
// Modified by Anders Logg, 2010-2011.
//
// First added:  2010-11-01
// Last changed: 2011-01-28

#ifndef __ADAPTIVE_DATUM_H
#define __ADAPTIVE_DATUM_H

#include <string>
#include <dolfin/common/Variable.h>

namespace dolfin
{

  class Table;

  /// An _AdaptiveDatum_ is a storage unit for data created in an
  /// adaptive process.

  class AdaptiveDatum : public Variable
  {
  public:

    /// Create adaptive datum
    ///
    /// *Arguments*
    ///
    ///     refinement_level (an (unsigned int)
    ///         the number of refinements relative to coarset mesh
    ///
    ///     num_dofs (an (unsigned) int)
    ///         dimension of discrete solution space
    ///
    ///     num_cells (an (unsigned) int)
    ///         number of cells in mesh
    ///
    ///     error_estimate (double)
    ///         error estimate
    ///
    ///     tolerance (double)
    ///         error (or num_dofs) tolerance
    AdaptiveDatum(const uint refinement_level,
                  const uint num_dofs,
                  const uint num_cells,
                  const double error_estimate,
                  const double tolerance,
                  const double functional_value);

    /// Destructor.
    ~AdaptiveDatum() { /* Do nothing */};

    /// Store adaptive datum to file
    ///
    /// *Arguments*
    ///     filename (string)
    ///         Name of file to store in
    void store(std::string filename) const;

    /// Store adaptive datum to _Table_.
    ///
    /// *Arguments*
    ///     table (_Table_)
    ///         Table to store in
    void store(Table& table) const;

    void set_reference_value(const double reference);

    // AdaptiveVariationalSolver is AdaptiveDatum's BFF.
    friend class AdaptiveVariationalSolver;

  private:

    uint refinement_level;
    uint num_dofs;
    uint num_cells;
    double error_estimate;
    double tolerance;

    double functional_value;
    double reference;

    bool reference_value_given;

  };

}
#endif
