// Copyright (C) 2003-2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells 2005.
//
// First added:  2003-03-13
// Last changed: 2009-08-11

#ifndef __LOG_STREAM_H
#define __LOG_STREAM_H

#include <string>
#include <sstream>
#include <dolfin/common/types.h>
#include <dolfin/common/real.h>

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
    enum Type {COUT, ENDL};

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
    LogStream& operator<< (uint a);

    /// Output for double
    LogStream& operator<< (double a);

    /// Output for complex
    LogStream& operator<< (complex z);

    /// Output for variable (calling str() method)
    LogStream& operator<< (const Variable& variable);

    /// Output for mesh entity (not subclass of Variable for efficiency)
    LogStream& operator<< (const MeshEntity& entity);

    /// Output for mesh entity iterator (not subclass of Variable for efficiency)
    LogStream& operator<< (const MeshEntityIterator& iterator);

    /// Output for point (not subclass of Variable for efficiency)
    LogStream& operator<< (const Point& point);

#ifdef HAS_GMP
    /// Output for real
    LogStream& operator<< (real a);
#endif

  private:

    // Type of stream
    Type type;

    // Buffer
    std::stringstream buffer;

  };

  /// dolfin::cout
  extern LogStream cout;

  /// dolfin::endl;
  extern LogStream endl;

}

#endif
