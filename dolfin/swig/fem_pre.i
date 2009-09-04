/* -*- C -*- */
// Copyright (C) 2009 Anders Logg
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth Wells, 2007-2009.
// Modified by Kent-Andre Mardal, 2008.
// Modified by Johan Hake, 2008-2009.
//
// First added:  2007-08-16
// Last changed: 2009-09-02

// ===========================================================================
// SWIG directives for the DOLFIN fem kernel module (pre)
//
// The directives in this file are applied _before_ the header files of the
// modules has been loaded.
// ===========================================================================

//-----------------------------------------------------------------------------
// To simplify handling of shared_ptr types in PyDOLFIN we ignore the reference
// version of constructors to these types
//-----------------------------------------------------------------------------
%ignore dolfin::EqualityBC::EqualityBC(const FunctionSpace&, uint);
%ignore dolfin::DirichletBC::DirichletBC(const FunctionSpace&, 
					 const Function&, 
					 const SubDomain&,
					 std::string method="topological");

%ignore dolfin::DirichletBC::DirichletBC(const FunctionSpace&, 
					 const Function&,
					 const MeshFunction<uint>&, 
					 uint,
					 std::string method="topological");

%ignore dolfin::DirichletBC::DirichletBC(const FunctionSpace&, 
					 const Function&, 
					 uint,
					 std::string method="topological");

%ignore dolfin::PeriodicBC::PeriodicBC(const FunctionSpace&, const SubDomain&);
