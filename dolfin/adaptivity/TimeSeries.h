// Copyright (C) 2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-11-11
// Last changed: 2011-02-25

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
    void retrieve(GenericVector& vector, double t) const;

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
