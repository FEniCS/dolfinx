// Copyright (C) 2008 Benjamin Kehlet
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg, 2008.
//
// First added:  2008-06-11
// Last changed: 2008-08-08

#include <algorithm>
#include "ODESolution.h"
#include "Sample.h"
#include "ODE.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
ODESolution::ODESolution(ODE& ode) : 
  ode(ode),
  filename(tmpnam(0)),
  file(filename.c_str(), std::ios::out | std::ios::binary),
  bintree(std::vector<double>()),
  step(sizeof(double)*ode.size()),
  buffer_count(0),
  dataondisk(false)
{
  // Initalize cache
  cache_size = ode.get("ODE order");
  std::string m = ode.get("ODE method") ;
  if (m == "dg") ++cache_size;

  cache = new std::pair<double, uBLASVector>[cache_size];
  for (uint i = 0; i < cache_size; ++i) 
  {
    cache[i].first = -1;
    cache[i].second.resize(ode.size());
  }
  ringbufcounter = 0;

  // Initialize buffer
  buffer_size = ODESOLUTION_INITIAL_ALLOC - (ODESOLUTION_INITIAL_ALLOC % step);
  buffer = (double *) malloc(buffer_size);
  buffer_offset = 0;
}
//-----------------------------------------------------------------------------
ODESolution::~ODESolution() 
{
  delete[] cache;
  file.close();
  remove(filename.c_str());
  free(buffer);
}
//-----------------------------------------------------------------------------
void ODESolution::eval(const double t, uBLASVector& y) 
{
  // Scan the cache
  for (uint i = 0; i < cache_size; ++i) 
  {
    if (cache[i].first < 0) continue;
    
    if (cache[i].first == t) 
    {
      //found return cache[i]
      for (uint j = 0; j < ode.size(); j++) 
      {
        uBLASVector& c = cache[i].second;
        y[j] = c[j];
      }
      return;
    }
  }

  // Not found in cache
  std::vector<double>::iterator low = std::lower_bound(bintree.begin(), 
						     bintree.end(), 
						     t);
  uint b = uint(low-bintree.begin());
  uint a = b - 1;

  if ( b >= bintree.size() || b < 1 ) 
  {
    error("ODESolution, eval(%g) out of range", t);
  } 

  double t_a = bintree[a];
  double t_b = bintree[b];

  // Check if we need to read from disk
  if (a < buffer_offset || b > buffer_offset + buffer_count) 
  {  
    cout << "ODESolution: Fetching from disk" << endl;

    //put a in the middle of the buffer
    buffer_offset = (uint) std::max((int) (a - buffer_size/(step*2)), 0);
    buffer_count = std::min(buffer_size / step, static_cast<uint>(bintree.size() - static_cast<uint>(buffer_offset)));
    file.seekg(buffer_offset*step);
    file.read( (char *) buffer, buffer_count*step);
  }

  uBLASVector tmp(ode.size());
  for (unsigned int i = 0; i < ode.size(); i++) 
  {
    y[i]   = buffer[(a - buffer_offset)*ode.size() + i];
    tmp[i] = buffer[(b - buffer_offset)*ode.size() + i];
  }

  // Do linear interpolation
  interpolate(y, t_a, tmp, t_b, t, y);

  // Cache y
  cache[ringbufcounter].first = t;
  for (uint i = 0; i < ode.size(); i++) 
  {
    cache[ringbufcounter].second[i] = y[i];
  }

  ringbufcounter = (ringbufcounter + 1) % cache_size;
}
//-----------------------------------------------------------------------------
void ODESolution::add_sample(Sample& sample) 
{  
  cout << "Adding sample at t = " << sample.t() << endl;
  
  bintree.push_back(sample.t());
  
  // Check if there is allocated memory for another entry in the buffer
  if (step*(buffer_count+1) > buffer_size) 
  {
    // Check if we can just extend the allocated memory
    if (buffer_size*2 <= ODESOLUTION_MAX_ALLOC) 
    {
      // Extend the memory
      buffer_size *= 2;
      buffer = (double *) realloc(buffer, buffer_size);
    }
    else
    {
      // No more memory available, dump to disk
      cout << "ODESolution: Writing to disk" << endl;
      file.write((char *) buffer, step*buffer_count);
      buffer_offset += buffer_count;
      buffer_count = 0;
      dataondisk = true;
    }
  }

  for (uint i = 0; i < sample.size(); ++i) 
    buffer[buffer_count*ode.size()+i] = sample.u(i);

  ++buffer_count;
}
//-----------------------------------------------------------------------------
void ODESolution::flush() 
{
  if (dataondisk && buffer_count > 0) 
  {
    cout << "ODESolution: Writing last chunk to disk" << endl;
    file.write((char *) buffer, buffer_count*step);
  }

  file.close();
  file.open(filename.c_str(), std::ios::in | std::ios::binary);  
}
//-----------------------------------------------------------------------------
void ODESolution::interpolate(const uBLASVector& v1, 
                              const double t1, 
                              const uBLASVector& v2, 
                              const double t2, 
                              const double t, 
                              uBLASVector& result) 
{
  double h = t2-t1;
  for (uint i = 0; i < ode.size(); i++) 
  {
    result[i] = v1[i] + (t-t1)*((v2[i]-v1[i])/h);
  }
}
//-----------------------------------------------------------------------------
