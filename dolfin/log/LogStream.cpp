// Copyright (C) 2003-2007 Anders Logg
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
// Modified by Garth N. Wells, 2009.

#include <dolfin/common/constants.h>
#include <dolfin/common/Variable.h>
#include <dolfin/geometry/Point.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshEntity.h>
#include <dolfin/mesh/MeshEntityIterator.h>
#include "log.h"
#include "LogStream.h"

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
LogStream& LogStream::operator<< (const LogStream& stream)
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
LogStream& LogStream::operator<< (const std::string& s)
{
  buffer << s;
  return *this;
}
//-----------------------------------------------------------------------------
LogStream& LogStream::operator<< (int a)
{
  buffer << a;
  return *this;
}
//-----------------------------------------------------------------------------
LogStream& LogStream::operator<< (unsigned int a)
{
  buffer << a;
  return *this;
}
//-----------------------------------------------------------------------------
LogStream& LogStream::operator<< (long int a)
{
  buffer << a;
  return *this;
}
//-----------------------------------------------------------------------------
LogStream& LogStream::operator<< (long unsigned int a)
{
  buffer << a;
  return *this;
}
//-----------------------------------------------------------------------------
LogStream& LogStream::operator<< (double a)
{
  buffer << a;
  return *this;
}
//-----------------------------------------------------------------------------
LogStream& LogStream::operator<< (std::complex<double> z)
{
  buffer << z.real() << " + " << z.imag() << "i";
  return *this;
}
//-----------------------------------------------------------------------------
LogStream& LogStream::operator<< (const Variable& variable)
{
  buffer << variable.str(false);
  return *this;
}
//-----------------------------------------------------------------------------
LogStream& LogStream::operator<< (const MeshEntity& entity)
{
  buffer << entity.str(false);
  return *this;
}
//-----------------------------------------------------------------------------
LogStream& LogStream::operator<< (const Point& point)
{
  buffer << point.str(false);
  return *this;
}
//-----------------------------------------------------------------------------
void LogStream::setprecision(std::streamsize n)
{
  buffer.precision(n);
}
//-----------------------------------------------------------------------------
