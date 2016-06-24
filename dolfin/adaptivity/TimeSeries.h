// Copyright (C) 2009-2016 Anders Logg
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

#ifndef __TIME_SERIES_H
#define __TIME_SERIES_H

#ifdef HAS_HDF5

#include <string>
#include <vector>
#include <dolfin/common/MPI.h>
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
    ///
    /// *Arguments*
    ///     mpi_comm (MPI_Comm)
    ///         An MPI communicator
    ///     name (std::string)
    ///         The time series name
    TimeSeries(std::string name)
      : TimeSeries(MPI_COMM_WORLD, name) {}

    /// Create empty time series
    ///
    /// *Arguments*
    ///     mpi_comm (MPI_Comm)
    ///         An MPI communicator
    ///     name (std::string)
    ///         The time series name
    TimeSeries(MPI_Comm mpi_comm, std::string name);

    /// Destructor
    ~TimeSeries();

    /// Store vector at given time
    ///
    /// *Arguments*
    ///     vector (_GenericVector_)
    ///         The vector to be stored.
    ///     t (double)
    ///         The time.
    void store(const GenericVector& vector, double t);

    /// Store mesh at given time
    ///
    /// *Arguments*
    ///     mesh (_Mesh_)
    ///         The mesh to be stored.
    ///     t (double)
    ///         The time.
    void store(const Mesh& mesh, double t);

    /// Retrieve vector at given time
    ///
    /// *Arguments*
    ///     vector (_GenericVector_)
    ///         The vector (values to be retrieved).
    ///     t (double)
    ///         The time.
    ///     interpolate (bool)
    ///         Optional argument: If true (default), interpolate
    ///         time samples closest to t if t is not present.
    void retrieve(GenericVector& vector, double t,
                  bool interpolate=true) const;

    /// Retrieve mesh at given time
    ///
    /// *Arguments*
    ///     mesh (_Mesh_)
    ///         The mesh (values to be retrieved).
    ///     t (double)
    ///         The time.
    void retrieve(Mesh& mesh, double t) const;

    /// Return array of sample times for vectors
    ///
    /// *Returns*
    ///     std::vector<double>
    ///         The times.
    std::vector<double> vector_times() const;

    /// Return array of sample times for meshes
    ///
    /// *Returns*
    ///     std::vector<double>
    ///         The times.
    std::vector<double> mesh_times() const;

    /// Clear time series
    void clear();

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

    template <typename T>
      void store_object(MPI_Comm comm, const T& object, double t,
                        std::vector<double>& times,
                        std::string series_name,
                        std::string group_name);

    // Check if values are strictly increasing
    static bool monotone(const std::vector<double>& times);

    // Find index closest to given time
    static std::size_t find_closest_index(double t,
                                          const std::vector<double>& times,
                                          std::string series_name,
                                          std::string type_name);

    // Find index pair closest to given time
    static std::pair<std::size_t, std::size_t>
      find_closest_pair(double t, const std::vector<double>& times,
                        std::string series_name, std::string type_name);

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
#endif
