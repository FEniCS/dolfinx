// Copyright (C) 2003-2007 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "LogStream.h"
#include "log.h"
#include <dolfin/common/Variable.h>
#include <dolfin/common/constants.h>
#include <dolfin/geometry/Point.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshEntity.h>

using namespace dolfin;

// Definition of the global dolfin::cout and dolfin::endl variables
LogStream dolfin::cout(LogStream::Type::COUT);
LogStream dolfin::endl(LogStream::Type::ENDL);

//-----------------------------------------------------------------------------
LogStream::LogStream(Type type) : _type(type)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
LogStream::~LogStream()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
LogStream& LogStream::operator<<(const LogStream& stream)
{
  if (stream._type == Type::ENDL)
  {
    // Send buffer to log system
    info(buffer.str());

    // Reset buffer
    buffer.str("");
  }
  else
    buffer << stream.buffer.str();

  return *this;
}
//-----------------------------------------------------------------------------
LogStream& LogStream::operator<<(const std::string& s)
{
  buffer << s;
  return *this;
}
//-----------------------------------------------------------------------------
LogStream& LogStream::operator<<(int a)
{
  buffer << a;
  return *this;
}
//-----------------------------------------------------------------------------
LogStream& LogStream::operator<<(unsigned int a)
{
  buffer << a;
  return *this;
}
//-----------------------------------------------------------------------------
LogStream& LogStream::operator<<(long int a)
{
  buffer << a;
  return *this;
}
//-----------------------------------------------------------------------------
LogStream& LogStream::operator<<(long unsigned int a)
{
  buffer << a;
  return *this;
}
//-----------------------------------------------------------------------------
LogStream& LogStream::operator<<(double a)
{
  buffer << a;
  return *this;
}
//-----------------------------------------------------------------------------
LogStream& LogStream::operator<<(std::complex<double> z)
{
  buffer << z.real() << " + " << z.imag() << "i";
  return *this;
}
//-----------------------------------------------------------------------------
LogStream& LogStream::operator<<(const common::Variable& variable)
{
  buffer << variable.str(false);
  return *this;
}
//-----------------------------------------------------------------------------
LogStream& LogStream::operator<<(const geometry::Point& point)
{
  buffer << point.str(false);
  return *this;
}
//-----------------------------------------------------------------------------
void LogStream::setprecision(std::streamsize n) { buffer.precision(n); }
//-----------------------------------------------------------------------------
