// Copyright (C) 2009-20111 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-11-11
// Last changed: 2011-02-24

#include <algorithm>
#include <sstream>

#include <dolfin/io/File.h>
#include "TimeSeries.h"

using namespace dolfin;

// Template function for storing objects
template <class T>
void store_object(const T& object, double t,
                  std::vector<double>& times,
                  std::string series_name,
                  std::string type_name)
{
  // Write object
  File file_data(TimeSeries::filename_data(series_name, type_name, times.size()));
  file_data << object;

  // Add time
  times.push_back(t);

  // Store times
  File file_times(TimeSeries::filename_times(series_name, type_name));
  file_times << times;
}

//-----------------------------------------------------------------------------
TimeSeries::TimeSeries(std::string name)
  : _name(name), _cleared(false), _reversed(false)
{
  not_working_in_parallel("Storing of data to time series");

  std::string filename;

  // Read vector times
  filename = TimeSeries::filename_times(_name, "vector");
  if (File::exists(filename))
  {
    // Read from file
    File file(filename);
    file >> _vector_times;
    info("Found %d vector sample(s) in time series.", _vector_times.size());

    // Reverse values if necessary
    if (!increasing(_vector_times))
    {
      std::reverse(_vector_times.begin(), _vector_times.end());
      _reversed = true;
    }
    if (!increasing(_vector_times))
      error("Sample times for vector time series are not strictly increasing or decreasing.");
  }
  else
    info("No vector samples found in time series.");

  // Read mesh times
  filename = TimeSeries::filename_times(_name, "mesh");
  if (File::exists(filename))
  {
    // Read from file
    File file(filename);
    file >> _mesh_times;
    info("Found %d mesh sample(s) in time series.", _vector_times.size());

    // Reverse values if necessary
    if (!increasing(_mesh_times))
    {
      std::reverse(_mesh_times.begin(), _mesh_times.end());
      _reversed = true;
    }
    if (!increasing(_mesh_times))
      error("Sample times for mesh time series are not strictly increasing or decreasing.");
  }
  else
    info("No mesh samples found in time series.");
}
//-----------------------------------------------------------------------------
TimeSeries::~TimeSeries()
{
  // Do nothing (keep files)
}
//-----------------------------------------------------------------------------
void TimeSeries::store(const GenericVector& vector, double t)
{
  // Clear earlier history first time we store a value
  if (!_cleared)
    clear();

  // Store object
  store_object(vector, t, _vector_times, _name, "vector");
}
//-----------------------------------------------------------------------------
void TimeSeries::store(const Mesh& mesh, double t)
{
  // Clear earlier history first time we store a value
  if (!_cleared)
    clear();

  // Store object
  store_object(mesh, t, _mesh_times, _name, "mesh");
}
//-----------------------------------------------------------------------------
void TimeSeries::retrieve(GenericVector& vector, double t) const
{
  // Get index closest to given time
  const uint index = find_closest_index(t, _vector_times, _name,
                                        "vector", _reversed);

  // Read vector
  File file(filename_data(_name, "vector", index));
  file >> vector;
}
//-----------------------------------------------------------------------------
void TimeSeries::retrieve(Mesh& mesh, double t) const
{
  // Get index closest to given time
  const uint index = find_closest_index(t, _vector_times, _name,
                                        "vector", _reversed);

  // Read vector
  File file(filename_data(_name, "mesh", index));
  file >> mesh;
}
//-----------------------------------------------------------------------------
Array<double> TimeSeries::vector_times() const
{
  Array<double> times(_vector_times.size());
  for (uint i = 0; i < _vector_times.size(); i++)
    times[i] = _vector_times[i];
  return times;
}
//-----------------------------------------------------------------------------
Array<double> TimeSeries::mesh_times() const
{
  Array<double> times(_mesh_times.size());
  for (uint i = 0; i < _mesh_times.size(); i++)
    times[i] = _mesh_times[i];
  return times;
}
//-----------------------------------------------------------------------------
void TimeSeries::clear()
{
  _vector_times.clear();
  _mesh_times.clear();
  _cleared = true;
}
//-----------------------------------------------------------------------------
std::string TimeSeries::filename_data(std::string series_name,
                                      std::string type_name,
                                      uint index)
{
  std::stringstream s;
  s << series_name << "_" << type_name << "_" << index << ".bin";
  return s.str();
}
//-----------------------------------------------------------------------------
std::string TimeSeries::filename_times(std::string series_name,
                                       std::string type_name)
{
  std::stringstream s;
  s << series_name << "_" << type_name << "_times" << ".bin";
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
bool TimeSeries::increasing(const std::vector<double>& times)
{
  for (uint i = 0; i < times.size() - 1; i++)
    if (!(times[i + 1] > times[i]))
      return false;
  return true;
}
//-----------------------------------------------------------------------------
dolfin::uint TimeSeries::find_closest_index(double t,
                                            const std::vector<double>& times,
                                            std::string series_name,
                                            std::string type_name,
                                            bool reversed)
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

  // Check if series is reversed
  if (reversed)
  {
    index = times.size() - 1 - index;
    dolfin_debug1("Series is reversed so I'm using index = %d instead.", index);
  }

  return index;
}
//-----------------------------------------------------------------------------
