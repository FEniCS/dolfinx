// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Garth N. Wells, 2006.
//
// First added:  2006-06-12
// Last changed: 2006-10-16

#include <dolfin/Point.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
dolfin::LogStream& dolfin::operator<< (LogStream& stream, const Point& p)
{
   stream << "[ Point x = " << p.x() << " y = " << p.y() << " z = " << p.z() << " ]";
   return stream;
}
//-----------------------------------------------------------------------------
