// Copyright (C) 2006-2009 Anders Logg
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Ola Skavhaug, 2007-2008.
// Modified by Johan Hake, 2008-2009.
//
// First added:  2006-04-16
// Last changed: 2009-09-02

//-----------------------------------------------------------------------------
// Global ignores
//-----------------------------------------------------------------------------
%ignore *::operator=;
%ignore *::operator[];
%ignore *::operator++;
%ignore *::operator<<(unsigned int);

%ignore operator<<;
%ignore operator dolfin::uint;
%ignore operator std::string;
%ignore operator bool;

// SWIG gets confused wrt overloading of these functions
//%ignore dolfin::MPI::send_recv;
//%ignore dolfin::MPI::gather;
//%ignore dolfin::MPI::distribute;
