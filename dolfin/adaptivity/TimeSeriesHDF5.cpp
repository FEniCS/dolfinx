// Copyright (C) 2009-2013 Chris Richardson and Anders Logg
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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2013-04-10
// Last changed: 2013-04-19

#ifdef HAS_HDF5

#include <algorithm>
#include <iostream>
#include <sstream>
#include <boost/lexical_cast.hpp>
#include <boost/scoped_ptr.hpp>

#include <dolfin/log/LogStream.h>
#include <dolfin/common/constants.h>
#include <dolfin/io/File.h>
#include <dolfin/io/HDF5File.h>
#include <dolfin/io/HDF5Interface.h>
#include <dolfin/la/GenericVector.h>
#include <dolfin/la/GenericLinearAlgebraFactory.h>

#include "TimeSeriesHDF5.h"

using namespace dolfin;

// Template function for storing objects
template <typename T>
void TimeSeriesHDF5::store_object(const T& object, double t,
                                  std::vector<double>& times,
                                  std::string series_name,
                                  std::string group_name)
{
  // Write object

  // Check for pre-existing file to append to
  std::string mode = "w";
  if(File::exists(series_name) && 
     (_vector_times.size() > 0 || _mesh_times.size() > 0))
    mode = "a";
    
  // Get file handle for low level operations
  HDF5File hdf5_file(series_name, mode);
  const hid_t fid = hdf5_file.hdf5_file_id;

  // Find existing datasets (should be equal to number of times)
  std::size_t nobjs = 0;
  if(HDF5Interface::has_group(fid, group_name))
    nobjs = HDF5Interface::num_datasets_in_group(fid, group_name);

  dolfin_assert(nobjs == times.size());

  // Write new dataset (mesh or vector)
  std::string dataset_name = group_name + "/" + boost::lexical_cast<std::string>(nobjs);
  hdf5_file.write(object, dataset_name);

  // Check that time values are strictly increasing
  const std::size_t n = times.size();
  if (n >= 2 and (times[n - 1] - times[n - 2])*(t - times[n - 1]) < 0.0)
  {
    dolfin_error("TimeSeries.cpp",
                 "store object to time series",
                 "Sample points must be strictly monotone (t_0 = %g, t_1 = %g, t_2 = %g)",
                 times[n - 2], times[n - 1], t);
  }

  // Add time
  times.push_back(t);

  // Store times
  HDF5Interface::add_attribute(fid, group_name, "times", times);
  
}

//-----------------------------------------------------------------------------
TimeSeriesHDF5::TimeSeriesHDF5(std::string name) : _name(name + ".h5"), _cleared(false)
{
  // Set default parameters
  parameters = default_parameters();

  if (File::exists(_name))
  {
    // Read from file
    const hid_t hdf5_file_id = HDF5Interface::open_file(_name, "r", true);

    if(HDF5Interface::has_group(hdf5_file_id, "/Vector") && 
       HDF5Interface::has_attribute(hdf5_file_id, "/Vector", "times"))
      {
        HDF5Interface::get_attribute(hdf5_file_id, "/Vector", "times", _vector_times);
        log(PROGRESS, "Found %d vector sample(s) in time series.", _vector_times.size());    
        if (!monotone(_vector_times))
        {
          dolfin_error("TimeSeriesHDF5.cpp",
                       "read time series from file",
                       "Sample points for vector data are not strictly monotone in series \"%s\"",
                       name.c_str());
        }
      }
    
    if(HDF5Interface::has_group(hdf5_file_id, "/Mesh") &&
       HDF5Interface::has_attribute(hdf5_file_id, "/Mesh", "times"))
    {
      HDF5Interface::get_attribute(hdf5_file_id, "/Mesh", "times", _mesh_times);
      log(PROGRESS, "Found %d mesh sample(s) in time series.", _mesh_times.size());
      if (!monotone(_mesh_times))
      {
        dolfin_error("TimeSeries.cpp",
                     "read time series from file",
                     "Sample points for mesh data are not strictly monotone in series \"%s\"",
                     name.c_str());
      }
    }
    
    HDF5Interface::close_file(hdf5_file_id);
  }
  else
    log(PROGRESS, "No samples found in time series.");

}
//-----------------------------------------------------------------------------
TimeSeriesHDF5::~TimeSeriesHDF5()
{
  // Do nothing (keep files)
}
//-----------------------------------------------------------------------------
void TimeSeriesHDF5::store(const GenericVector& vector, double t)
{
  // Clear earlier history first time we store a value
  const bool clear_on_write = this->parameters["clear_on_write"];
  if (!_cleared && clear_on_write)
    clear();

  // Store object
  store_object(vector, t, _vector_times, _name, "/Vector");

}
//-----------------------------------------------------------------------------
void TimeSeriesHDF5::store(const Mesh& mesh, double t)
{
  // Clear earlier history first time we store a value
  const bool clear_on_write = this->parameters["clear_on_write"];
  if (!_cleared && clear_on_write)
    clear();

  // Store object
  store_object(mesh, t, _mesh_times, _name, "/Mesh");
  
}
//-----------------------------------------------------------------------------
void TimeSeriesHDF5::retrieve(GenericVector& vector, double t, bool interpolate) const
{
  HDF5File hdf5_file(_name, "r");
  const std::size_t zero = 0;
  vector.resize(1,&zero);

  // Interpolate value
  if (interpolate)
  {
    // Find closest pair
    const std::pair<std::size_t, std::size_t> index_pair = find_closest_pair(t, _vector_times, _name, "vector");
    const std::size_t i0 = index_pair.first;
    const std::size_t i1 = index_pair.second;

    // Special case: same index
    if (i0 == i1)
    {
      hdf5_file.read(vector, "/Vector/" + boost::lexical_cast<std::string>(i0));
      log(PROGRESS, "Reading vector value at t = %g.", _vector_times[0]);
      return;
    }

    log(PROGRESS, "Interpolating vector value at t = %g in interval [%g, %g].",
        t, _vector_times[i0], _vector_times[i1]);

    // Read vectors
    GenericVector& x0(vector);
    boost::shared_ptr<GenericVector> x1 = x0.factory().create_vector();
    hdf5_file.read(x0, "/Vector/" + boost::lexical_cast<std::string>(i0));
    hdf5_file.read(*x1, "/Vector/" + boost::lexical_cast<std::string>(i1));

    // Check that the vectors have the same size
    if (x0.size() != x1->size())
    {
      dolfin_error("TimeSeries.cpp",
                   "interpolate vector value in time series",
                   "Vector sizes don't match (%d and %d)",
                   x0.size(), x1->size());
    }

    // Compute weights for linear interpolation
    const double dt = _vector_times[i1] - _vector_times[i0];
    dolfin_assert(std::abs(dt) > DOLFIN_EPS);
    const double w0 = (_vector_times[i1] - t) / dt;
    const double w1 = 1.0 - w0;

    // Interpolate
    x0 *= w0;
    x0.axpy(w1, *x1);
  }

  // Read closest value
  else
  {
    // Find closest index
    const std::size_t index = find_closest_index(t, _vector_times, _name, "vector");

    log(PROGRESS, "Reading vector at t = %g (close to t = %g).",
        _vector_times[index], t);

    // Read vector
    hdf5_file.read(vector, "/Vector/" + boost::lexical_cast<std::string>(index));
  }
}
//-----------------------------------------------------------------------------
void TimeSeriesHDF5::retrieve(Mesh& mesh, double t) const
{
  // Get index closest to given time
  const std::size_t index = find_closest_index(t, _mesh_times, _name, "mesh");

  log(PROGRESS, "Reading mesh at t = %g (close to t = %g).",
      _mesh_times[index], t);

  // Read mesh
  HDF5File hdf5_file(_name, "r");
  hdf5_file.read(mesh, "/Mesh/" + boost::lexical_cast<std::string>(index));

}
//-----------------------------------------------------------------------------
std::vector<double> TimeSeriesHDF5::vector_times() const
{
  return _vector_times;
}
//-----------------------------------------------------------------------------
std::vector<double> TimeSeriesHDF5::mesh_times() const
{
  return _mesh_times;
}
//-----------------------------------------------------------------------------
void TimeSeriesHDF5::clear()
{
  _vector_times.clear();
  _mesh_times.clear();
  _cleared = true;
}
//-----------------------------------------------------------------------------
std::string TimeSeriesHDF5::str(bool verbose) const
{
  std::stringstream s;

  if (verbose)
  {
    s << str(false) << std::endl << std::endl;

    s << "Vectors:";
    for (std::size_t i = 0; i < _vector_times.size(); ++i)
      s << "  " << i << ": " << _vector_times[i];
    s << std::endl;

    s << "Meshes:";
    for (std::size_t i = 0; i < _mesh_times.size(); ++i)
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
bool TimeSeriesHDF5::monotone(const std::vector<double>& times)
{
  // If size of time series is 0 or 1 they are always monotone
  if (times.size() < 2)
    return true;

  // If size is 2
  if (times.size() == 2)
    return times[0] < times[1];

  for (std::size_t i = 0; i < times.size() - 2; i++)
  {
    if ((times[i + 2] - times[i + 1])*(times[i + 1] - times[i]) <= 0.0)
      return false;
  }
  return true;
}
//-----------------------------------------------------------------------------
std::size_t TimeSeriesHDF5::find_closest_index(double t,
                                            const std::vector<double>& times,
                                            std::string series_name,
                                            std::string type_name)
{
  // Get closest pair
  const std::pair<std::size_t, std::size_t> index_pair = find_closest_pair(t, times, series_name, type_name);
  const std::size_t i0 = index_pair.first;
  const std::size_t i1 = index_pair.second;

  // Check which is closer
  const std::size_t i = (std::abs(t - times[i0]) < std::abs(t - times[i1]) ? i0 : i1);
  dolfin_debug2("Using closest value t[%d] = %g", i, times[i]);

  return i;
}
//-----------------------------------------------------------------------------
std::pair<std::size_t, std::size_t>
TimeSeriesHDF5::find_closest_pair(double t,
                              const std::vector<double>& times,
                              std::string series_name,
                              std::string type_name)
{
  //for (std::size_t i = 0; i < times.size(); i++) cout << " " << times[i]; cout << endl;

  // Must have at least one value stored
  if (times.empty())
  {
    dolfin_error("TimeSeries.cpp",
                 "to retrieve data from time seris",
                 "No %s stored in time series",
                 type_name.c_str());
  }

  // Special case: just one value stored
  if (times.size() == 1)
  {
    dolfin_debug("Series has just one value, returning index 0.");
    return std::make_pair(0, 0);
  }

  // Check whether series is reversed
  const bool reversed = times[0] > times[1];

  // Find lower bound. Note that lower_bound() returns first item
  // larger than t or end of vector if no such item exists.
  std::vector<double>::const_iterator lower;
  if (reversed)
    lower = std::lower_bound(times.begin(), times.end(), t, std::greater<double>());
  else
    lower = std::lower_bound(times.begin(), times.end(), t, std::less<double>());

  // Set indexlower and upper bound
  std::size_t i0 = 0;
  std::size_t i1 = 0;
  if (lower == times.begin())
    i0 = i1 = lower - times.begin();
  else if (lower == times.end())
    i0 = i1 = lower - times.begin() - 1;
  else
  {
    i0 = lower - times.begin() - 1;
    i1 = i0 + 1;
  }

  dolfin_debug1("Looking for value at time t = %g", t);
  dolfin_debug4("Neighboring values are t[%d] = %g and t[%d] = %g",
                i0, times[i0], i1, times[i1]);

  return std::make_pair(i0, i1);
}
//-----------------------------------------------------------------------------

#endif
