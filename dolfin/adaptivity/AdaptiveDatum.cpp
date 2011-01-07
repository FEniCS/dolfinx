// Copyright (C) 2010 Marie E. Rognes
// Licensed under the GNU LGPL Version 3.0 or any later version
//
// First added:  2010-11-01
// Last changed: 2011-01-07

#include <sstream>
#include <dolfin/common/utils.h>
#include <dolfin/log/Table.h>
#include "AdaptiveDatum.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
AdaptiveDatum::AdaptiveDatum(const uint refinement_level,
                             const uint num_dofs,
                             const uint num_cells,
                             const double error_estimate,
                             const double tolerance,
                             const double functional_value)
  : refinement_level(refinement_level),
    num_dofs(num_dofs),
    num_cells(num_cells),
    error_estimate(error_estimate),
    tolerance(tolerance),
    functional_value(functional_value),
    reference(0)
{
  reference_value_given = false;
}
//-----------------------------------------------------------------------------
void AdaptiveDatum::set_reference_value(const double reference)
{
  this->reference = reference;
  reference_value_given = true;
}
//-----------------------------------------------------------------------------
void AdaptiveDatum::store(std::string filename) const
{
  dolfin_not_implemented();
}
//-----------------------------------------------------------------------------
void AdaptiveDatum::store(Table& table) const
{
  std::stringstream s;
  s << refinement_level;

  table(s.str(), "M(u_h)")         = functional_value;
  if (reference_value_given)
    table(s.str(), "M(u)")         = reference;
  else
    table(s.str(), "M(u)")         = "N/A";
  table(s.str(), "TOL")            = tolerance;
  table(s.str(), "Error estimate") = error_estimate;
  table(s.str(), "#cells")         = num_cells;
  table(s.str(), "#dofs")          = num_dofs;

  if (reference_value_given) {
    const double error = reference - functional_value;
    double efficiency_index = 0.0;
    if (error)
      efficiency_index = error_estimate/error;

    table(s.str(), "eta") = error;
    table(s.str(), "eta_h/eta") = efficiency_index;
  }
}
//-----------------------------------------------------------------------------
