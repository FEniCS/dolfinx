// Copyright (C) 2003-2005 Johan Hoffman and Anders Logg
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
// Modified by Garth N. Wells 2005, 2010.
// Modified by Rolv E. Bredesen 2008.
// Modified by Benjamin Kehlet 2009.

// First added:  2003-05-06
// Last changed: 2010-09-03

#include <fstream>
#include <iostream>
#include <iomanip>

#include <dolfin/log/dolfin_log.h>
#include <dolfin/la/GenericVector.h>
#include <dolfin/la/GenericMatrix.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/function/Function.h>
#include <dolfin/ode/Sample.h>
#include "PythonFile.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
PythonFile::PythonFile(const std::string filename) : GenericFile(filename)
{
  type = "Python";

  std::string prefix = filename.substr(0, filename.rfind("."));
  filename_t = prefix + "_t.data";
  filename_u = prefix + "_u.data";
  filename_k = prefix + "_k.data";
  filename_r = prefix + "_r.data";
}
//-----------------------------------------------------------------------------
PythonFile::~PythonFile()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void PythonFile::operator<<(const Sample& sample)
{
  // First time
  if ( counter2 == 0 )
  {
    std::ofstream fp(filename.c_str());
    if (!fp.is_open())
      error("Unable to open file %s", filename.c_str());

    fp << "from numpy import fromfile" << std::endl << std::endl;
    fp << "t = fromfile('" << filename_t << "', sep=' ')" << std::endl;
    fp << "u = fromfile('" << filename_u << "', sep=' ')" << std::endl;
    fp << "k = fromfile('" << filename_k << "', sep=' ')" << std::endl;
    fp << "r = fromfile('" << filename_r << "', sep=' ')" << std::endl;
    fp << std::endl;
    fp << "u.shape = len(u) //" << sample.size() << ", " << sample.size() << std::endl;
    fp << "k.shape = len(k) //" << sample.size() << ", " << sample.size() << std::endl;
    fp << "r.shape = len(r) //" << sample.size() << ", " << sample.size() << std::endl;
    fp << std::endl;
    fp.close();
  }

  //sub files filemode:  append unless this is the first sample
  std::ios_base::openmode filemode = (counter2 == 0 ?
				        std::ios_base::out :
				        std::ios_base::out| std::ios_base::app);

  //get precision
  int prec = real_decimal_prec();

  // Open sub files
  std::ofstream fp_t(filename_t.c_str(), filemode);
  if (!fp_t.is_open())
    error("Unable to open file %s", filename_t.c_str());

  std::ofstream fp_u(filename_u.c_str(), filemode);
  if (!fp_u.is_open())
    error("Unable to open file %s", filename_u.c_str());

  std::ofstream fp_k(filename_k.c_str(), filemode);
  if (!fp_k.is_open())
    error("Unable to open file %s", filename_k.c_str());

  std::ofstream fp_r(filename_r.c_str(), filemode);
  if (!fp_r.is_open())
    error("Unable to open file %s", filename_r.c_str());

  // Save time
  fp_t << std::setprecision(prec) << sample.t() << std::endl;

  // Save solution
  for (uint i = 0; i < sample.size(); i++)
    fp_u << std::setprecision(prec) << sample.u(i) << " ";
  fp_u << std::endl;

  // Save time steps
  for (uint i = 0; i < sample.size(); i++)
    fp_k << std::setprecision(prec) << sample.k(i) << " ";
  fp_k << std::endl;

  // Save residuals
  for (uint i = 0; i < sample.size(); i++)
    fp_r << std::setprecision(prec) << sample.r(i) << " ";
  fp_r << std::endl;

  // Increase frame counter
  counter2++;
}
//-----------------------------------------------------------------------------
void PythonFile::operator<<(const std::pair<real, RealArrayRef> sample)
{  
  const real& t      = sample.first;
  const Array<real>& u = sample.second;

  // First time
  if ( counter2 == 0 )
  {
    std::ofstream fp(filename.c_str());
    if (!fp.is_open())
      error("Unable to open file %s", filename.c_str());

    fp << "from numpy import fromfile" << std::endl << std::endl;
    fp << "t = fromfile('" << filename_t << "', sep=' ')" << std::endl;
    fp << "u = fromfile('" << filename_u << "', sep=' ')" << std::endl;
    fp << std::endl;
    fp << "u.shape = len(u) //" << u.size() << ", " << u.size() << std::endl;
    fp << std::endl;
    fp.close();
  }

  // Sub files filemode:  append unless this is the first sample
  std::ios_base::openmode filemode = (counter2 == 0 ?
				      std::ios_base::out :
				      std::ios_base::out| std::ios_base::app);

  // Get precision
  const int prec = real_decimal_prec();

  // Open sub files
  std::ofstream fp_t(filename_t.c_str(), filemode);
  if (!fp_t.is_open())
    error("Unable to open file %s", filename_t.c_str());

  std::ofstream fp_u(filename_u.c_str(), filemode);
  if (!fp_u.is_open())
    error("Unable to open file %s", filename_u.c_str());

  // Save time
  fp_t << std::setprecision(prec) << t << std::endl;

  // Save solution
  for (uint i=0; i < u.size(); ++i)
    fp_u << std::setprecision(prec) << u[i] << " ";
  fp_u << std::endl;

  counter2++;
}
//-----------------------------------------------------------------------------
