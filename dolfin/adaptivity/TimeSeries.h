// Copyright (C) 2009 Anders Logg
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
// First added:  2009-11-11
// Last changed: 2011-03-31

#ifndef __TIME_SERIES_H
#define __TIME_SERIES_H

#include <string>
#include <dolfin/common/Array.h>
#include <dolfin/common/Variable.h>

namespace dolfin
{

  // Forward declarations
  class GenericVector;
  class Mesh;

  /// This class stores a time series of objects to file(s) in a
  /// binary format which is efficient for reading and writing.
  ///
  /// When objects are retrieved, the object stored at the time
  /// closest to the given time will be used.
  ///
  /// A new time series will check if values have been stored to
  /// file before (for a series with the same name) and in that
  /// case reuse those values. If new values are stored, old
  /// values will be cleared.

  class TimeSeries : public Variable
  {
  public:

    /// Create empty time series
    TimeSeries(std::string name);

    /// Destructor
    ~TimeSeries();

    /// Store vector at given time
    void store(const GenericVector& vector, double t);

    /// Store mesh at given time
    void store(const Mesh& mesh, double t);

    /// Retrieve vector at given time
    void retrieve(GenericVector& vector, double t, bool interpolate=true) const;

    /// Retrieve mesh at given time
    void retrieve(Mesh& mesh, double t) const;

    /// Return array of sample times for vectors
    Array<double> vector_times() const;

    /// Return array of sample times for meshes
    Array<double> mesh_times() const;

    /// Clear time series
    void clear();

    /// Return filename for data
    static std::string filename_data(std::string series_name,
                                     std::string type_name, uint index);

    /// Return filename for times
    static std::string filename_times(std::string series_name, std::string type_name);

    /// Return informal string representation (pretty-print)
    std::string str(bool verbose) const;

    /// Default parameter values
    static Parameters default_parameters()
    {
      Parameters p("time_series");
      p.add("clear_on_write", true);
      return p;
    }

  private:

    // Check if values are strictly increasing
    static bool increasing(const std::vector<double>& times);

    // Find index closest to given time
    static uint find_closest_index(double t,
                                   const std::vector<double>& times,
                                   std::string series_name,
                                   std::string type_name);

    // Find index pair closest to given time
    static std::pair<uint, uint> find_closest_pair(double t,
                                                   const std::vector<double>& times,
                                                   std::string series_name,
                                                   std::string type_name);

    // Name of series
    std::string _name;

    // List of times
    std::vector<double> _vector_times;
    std::vector<double> _mesh_times;

    // True if series has been cleared
    bool _cleared;

  };

}

#endif
