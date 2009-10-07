/* -*- C -*- */
// Copyright (C) 2009 Johan Hake
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-05-10
// Last changed: 2009-09-23

//=============================================================================
// SWIG directives for the DOLFIN log kernel module (pre)
//
// The directives in this file are applied _before_ the header files of the
// modules has been loaded.
//=============================================================================

//-----------------------------------------------------------------------------
// Due to a SWIG bug when overloading a function that also use elipsis (...) 
// argument in C++, we need to ignore other overloaded functions. They are 
// reimplemented in log_post.i
//-----------------------------------------------------------------------------
%ignore dolfin::info(const Parameters& parameters, bool verbose=false);
%ignore dolfin::info(const Variable& variable, bool verbose=false);
%rename(_info) dolfin::info;

//-----------------------------------------------------------------------------
// Need to ignore these dues to SWIG confusion of overloaded functions
//-----------------------------------------------------------------------------
%ignore dolfin::Table::set(std::string,std::string,uint);

//-----------------------------------------------------------------------------
// Ignore operators so SWIG stop complaining
//-----------------------------------------------------------------------------
%ignore dolfin::TableEntry::operator std::string;
%ignore dolfin::Progress::operator++;
%ignore dolfin::Progress::operator=;
%ignore dolfin::Table::operator=;
%ignore dolfin::TableEntry::operator=;

//-----------------------------------------------------------------------------
// Ignore DOLFIN C++ stream handling
//-----------------------------------------------------------------------------
%ignore dolfin::LogStream;
%ignore dolfin::cout;
%ignore dolfin::endl;
