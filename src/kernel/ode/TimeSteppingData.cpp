// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_settings.h>
#include <dolfin/Element.h>
#include <dolfin/TimeSteppingData.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
TimeSteppingData::TimeSteppingData(unsigned int N) : regulators(N)
{
  // Get parameters
  TOL                = dolfin_get("tolerance");
  kmax               = dolfin_get("maximum time step");
  interval_threshold = dolfin_get("interval threshold");
  _debug             = dolfin_get("debug time steps");
  real k0            = dolfin_get("initial time step");

  // Scale tolerance with the number of components
  TOL /= static_cast<real>(N);

  // Specify initial time steps
  for (unsigned int i = 0; i < regulators.size(); i++)
    regulators[i].init(k0);

  // Open debug file
  if ( _debug )
    file.open("timesteps.debug", std::ios::out);
}
//-----------------------------------------------------------------------------
TimeSteppingData::~TimeSteppingData()
{
  // Close debug file
  if ( _debug )
    file.close();
}
//-----------------------------------------------------------------------------
unsigned int TimeSteppingData::size() const
{
  return regulators.size();
}
//-----------------------------------------------------------------------------
Regulator& TimeSteppingData::regulator(unsigned int i)
{
  dolfin_assert(i < regulators.size());
  return regulators[i];
}
//-----------------------------------------------------------------------------
const Regulator& TimeSteppingData::regulator(unsigned int i) const
{
  dolfin_assert(i < regulators.size());
  return regulators[i];
}
//-----------------------------------------------------------------------------
real TimeSteppingData::tolerance() const
{
  return TOL;
}
//-----------------------------------------------------------------------------
real TimeSteppingData::maxstep() const
{
  // FIXME: Should we have an individual kmax for each component?
  // FIXME: In that case we should put kmax into the Regulator class.

  return kmax;
}
//-----------------------------------------------------------------------------
real TimeSteppingData::threshold() const
{
  return interval_threshold;
}
//-----------------------------------------------------------------------------
void TimeSteppingData::debug(Element& element, Action action)
{
  if ( !_debug )
    return;

  // Write debug info to file
  file << action << " "
       << element.index() << " " 
       << element.starttime() << " " 
       << element.endtime() << "\n";
}
//-----------------------------------------------------------------------------
