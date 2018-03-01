// Copyright (C) 2003-2009 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <complex>
#include <sstream>
#include <string>

namespace dolfin
{

namespace common
{
class Variable;
}

namespace geometry
{
class Point;
}

namespace log
{

/// This class provides functionality similar to standard C++
/// streams (std::cout, std::endl) for output but working through
/// the DOLFIN log system.

class LogStream
{
public:
  /// Stream types
  enum class Type
  {
    COUT,
    ENDL
  };

  /// Create log stream of given type
  LogStream(Type type);

  /// Destructor
  ~LogStream();

  /// Output for log stream
  LogStream& operator<<(const LogStream& stream);

  /// Output for string
  LogStream& operator<<(const std::string& s);

  /// Output for int
  LogStream& operator<<(int a);

  /// Output for unsigned int
  LogStream& operator<<(unsigned int a);

  /// Output for long int
  LogStream& operator<<(long int a);

  /// Output for long int
  LogStream& operator<<(long unsigned int a);

  /// Output for double
  LogStream& operator<<(double a);

  /// Output for std::complex<double>
  LogStream& operator<<(std::complex<double> z);

  /// Output for variable (calling str() method)
  LogStream& operator<<(const common::Variable& variable);

  /// Output for point (not subclass of common::Variable for efficiency)
  LogStream& operator<<(const geometry::Point& point);

  /// Set precisionPoi
  void setprecision(std::streamsize n);

private:
  // Type of stream
  Type _type;

  // Buffer
  std::stringstream buffer;
};
}

/// dolfin::cout
extern log::LogStream cout;

/// dolfin::endl;
extern log::LogStream endl;
}
