// Copyright (C) 2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-11-11
// Last changed: 2009-11-11

#include <algorithm>
#include <sstream>
#include <dolfin/io/File.h>
#include "TimeSeries.h"

using namespace dolfin;

// Template function for storing objects
template <class T> void store_object(const T& object, double t,
                                     std::vector<double>& times,
                                     std::string series_name, std::string type_name)
{
  // Write object
  File file(TimeSeries::filename(series_name, type_name, times.size()));
  file << object;

  // Store time
  times.push_back(t);
}

// Template function for retrieving objects
template <class T> void retrieve_object(T& object, double t,
                                        const std::vector<double>& times,
                                        std::string series_name, std::string type_name)
{
  // Must have at least one value stored
  if (times.size() == 0)
    error("Unable to retrieve %s, no %s stored in time series.",
          type_name.c_str(), type_name.c_str());

  // Find lower and upper bound for given time. Note that lower_bound()
  // returns the first item that is larger than the given time, or end
  // of vector if no such item exists.
  std::vector<double>::const_iterator lower, upper;
  lower = std::lower_bound(times.begin(), times.end(), t);
  if (lower == times.begin())
    upper = lower;
  else if (lower == times.end())
    upper = lower = lower - 1;
  else {
    lower = lower - 1;
    upper = lower + 1;
  }

  // Check which is closer
  unsigned int index = 0;
  if (std::abs(t - *lower) < std::abs(t - *upper))
    index = lower - times.begin();
  else
    index = upper - times.begin();

  dolfin_debug1("Looking for value at time t = %g", t);
  dolfin_debug2("Neighboring values are %g and %g", *lower, *upper);
  dolfin_debug2("Using closest value %g (index = %d)", times[index], index);

  // Read object
  File file(TimeSeries::filename(series_name, type_name, index));
  file >> object;
}

//-----------------------------------------------------------------------------
TimeSeries::TimeSeries(std::string name) : _series_name(name)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
TimeSeries::~TimeSeries()
{
  // Do nothing (keep files)
}
//-----------------------------------------------------------------------------
void TimeSeries::store(const GenericVector& vector, double t)
{
  store_object(vector, t, _vector_times, _series_name, "vector");
}
//-----------------------------------------------------------------------------
void TimeSeries::store(const Mesh& mesh, double t)
{
  store_object(mesh, t, _mesh_times, _series_name, "mesh");
}
//-----------------------------------------------------------------------------
void TimeSeries::retrieve(GenericVector& vector, double t) const
{
  retrieve_object(vector, t, _vector_times, _series_name, "vector");
}
//-----------------------------------------------------------------------------
void TimeSeries::retrieve(Mesh& mesh, double t) const
{
  retrieve_object(mesh, t, _vector_times, _series_name, "mesh");
}
//-----------------------------------------------------------------------------
std::string TimeSeries::filename(std::string series_name,
                                std::string type_name,
                                uint index)
{
  std::stringstream s;
  s << series_name << "_" << type_name << "_" << index << ".bin";
  return s.str();
}
//-----------------------------------------------------------------------------
std::string TimeSeries::str(bool verbose) const
{
  std::stringstream s;

  if (verbose)
  {
    s << str(false) << std::endl << std::endl;

    s << "Vectors:";
    for (uint i = 0; i < _vector_times.size(); ++i)
      s << "  " << i << ": " << _vector_times[i];
    s << std::endl;

    s << "Meshes:";
    for (uint i = 0; i < _mesh_times.size(); ++i)
      s << "  " << i << ": " << _mesh_times[i];
    s << std::endl;
  }
  else
  {
    s << "<Time series with "
      << _vector_times.size()
      << " vector(s) and "
      << _mesh_times.size()
      << " mesh(es)>";
  }

  return s.str();
}
//-----------------------------------------------------------------------------
