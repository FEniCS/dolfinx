// Copyright (C) 2003-2009 Anders Logg
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
// Modified by Garth N. Wells 2005.

#ifndef __LOG_STREAM_H
#define __LOG_STREAM_H

#include <complex>
#include <string>
#include <sstream>

namespace dolfin
{

  class Variable;
  class MeshEntity;
  class MeshEntityIterator;
  class Point;

  /// This class provides functionality similar to standard C++
  /// streams (std::cout, std::endl) for output but working through
  /// the DOLFIN log system.

  class LogStream
  {
  public:

    /// Stream types
    enum class Type {COUT, ENDL};

    /// Create log stream of given type
    LogStream(Type type);

    /// Destructor
    ~LogStream();

    /// Output for log stream
    LogStream& operator<< (const LogStream& stream);

    /// Output for string
    LogStream& operator<< (const std::string& s);

    /// Output for int
    LogStream& operator<< (int a);

    /// Output for unsigned int
    LogStream& operator<< (unsigned int a);

    /// Output for long int
    LogStream& operator<< (long int a);

    /// Output for long int
    LogStream& operator<< (long unsigned int a);

    /// Output for double
    LogStream& operator<< (double a);

    /// Output for std::complex<double>
    LogStream& operator<< (std::complex<double> z);

    /// Output for variable (calling str() method)
    LogStream& operator<< (const Variable& variable);

    /// Output for mesh entity (not subclass of Variable for
    /// efficiency)
    LogStream& operator<< (const MeshEntity& entity);

    /// Output for point (not subclass of Variable for efficiency)
    LogStream& operator<< (const Point& point);

    void setprecision(std::streamsize n);

  private:

    // Type of stream
    Type _type;

    // Buffer
    std::stringstream buffer;

  };

  /// dolfin::cout
  extern LogStream cout;

  /// dolfin::endl;
  extern LogStream endl;

}

#endif
