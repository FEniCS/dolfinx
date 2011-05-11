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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Garth N. Wells, 2009.
//
// First added:  2003-03-13
// Last changed: 2009-09-08

#include <dolfin/common/constants.h>
#include <dolfin/common/Variable.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshEntity.h>
#include <dolfin/mesh/MeshEntityIterator.h>
#include <dolfin/mesh/Point.h>
#include "log.h"
#include "LogStream.h"

using namespace dolfin;

// Definition of the global dolfin::cout and dolfin::endl variables
LogStream dolfin::cout(LogStream::COUT);
LogStream dolfin::endl(LogStream::ENDL);

//-----------------------------------------------------------------------------
LogStream::LogStream(Type type) : type(type)
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
  if (stream.type == ENDL)
  {
    // Send buffer to log system
    info(buffer.str());

    // Reset buffer
    buffer.str("");
  }
  else
    buffer << stream.buffer;

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
LogStream& LogStream::operator<< (uint a)
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
LogStream& LogStream::operator<< (complex z)
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
#ifdef HAS_GMP
LogStream& LogStream::operator<< (real a)
{
  char tmp[DOLFIN_LINELENGTH];
  gmp_snprintf(tmp, DOLFIN_LINELENGTH, "%.16Fg...", a.get_mpf_t());
  buffer << tmp;
  return *this;
}
#endif
//-----------------------------------------------------------------------------
void LogStream::setprecision(uint n)
{
  buffer.precision(n);
}
//-----------------------------------------------------------------------------
