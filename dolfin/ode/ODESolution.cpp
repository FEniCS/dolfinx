// Copyright (C) 2008 Benjamin Kehlet
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg, 2008.
//
// First added:  2008-06-11
// Last changed: 2008-10-06

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
  cache(0),
  bintree(std::vector<real>()),
  step(sizeof(real)*ode.size()),
  buffer(0), tmp(0),
  buffer_count(0),
  dataondisk(false)
{
  // Initalize cache
  cache_size = ode.get("ODE order");
  std::string m = ode.get("ODE method") ;
  if (m == "dg") ++cache_size;

  cache = new std::pair<real, real*>[cache_size];
  for (uint i = 0; i < cache_size; ++i) 
  {
    cache[i].first = -1;
    cache[i].second = new real[ode.size()];
  }
  ringbufcounter = 0;

  // Initialize buffer
  buffer_size = ODESOLUTION_INITIAL_ALLOC - (ODESOLUTION_INITIAL_ALLOC % step);
  buffer = (real *) malloc(buffer_size);
  buffer_offset = 0;

  // Initialize temporary array
  tmp = new real[ode.size()];
}
//-----------------------------------------------------------------------------
ODESolution::~ODESolution() 
{
  for (uint i = 0; i < cache_size; ++i)
    delete [] cache[i].second;
  delete [] cache;
  file.close();
  remove(filename.c_str());
  free(buffer);
  delete [] tmp;
}
//-----------------------------------------------------------------------------
void ODESolution::eval(const real t, real* y)
{
  // Scan the cache
  for (uint i = 0; i < cache_size; ++i) 
  {
    // Empty position, skip
    if (cache[i].first < 0)
      continue;
    
    // Copy values
    if (cache[i].first == t) 
    {
      real_set(ode.size(), y, cache[i].second);
      return;
    }
  }

  // Find position in buffer
  std::vector<real>::iterator upper = std::upper_bound(bintree.begin(), 
                                                         bintree.end(), 
                                                         t);
  uint b = uint(upper - bintree.begin());
  uint a = b - 1;

  if (b >= bintree.size() - 1)
  {
    if (t > bintree[bintree.size()-1])
    {
      error("ODESolution, eval(%g) out of range", to_double(t));
    }
    else
    {
      //t = max(t_in_solution)
      get_from_buffer(y, bintree.size() - 1);
    }
  } 

  // Read from buffer
  real t_b = bintree[b];
  real t_a = bintree[a];
  get_from_buffer(y, a);
  get_from_buffer(tmp, b);

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
  //cout << "Adding sample at t = " << sample.t() << endl;
  
  bintree.push_back(sample.t());
  
  // Check if there is allocated memory for another entry in the buffer
  if (step*(buffer_count+1) > buffer_size) 
  {
    // Check if we can just extend the allocated memory
    if (buffer_size*2 <= ODESOLUTION_MAX_ALLOC) 
    {
      // Extend the memory
      buffer_size *= 2;
      buffer = (real *) realloc(buffer, buffer_size);
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
  {
    #ifndef HAS_GMP
      //TODO: Fix this so it works with GMP
      buffer[buffer_count*ode.size()+i] = sample.u(i);
    #endif
  }

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
void ODESolution::interpolate(const real* v0, const real t0,
                              const real* v1, const real t1,
                              const real t, real* result)
{
  const real h = t1 - t0;
  for (uint i = 0; i < ode.size(); i++)
  {
    result[i] = v0[i] + (t - t0) * (v1[i] - v0[i]) / h;
  }
}
//-----------------------------------------------------------------------------
void ODESolution::get_from_buffer(real* u, uint index)
{
  // Check if we need to read from disk
  if (index < buffer_offset || index > buffer_offset + buffer_count) 
  {  
    cout << "ODESolution: Fetching from disk" << endl;
    
    //put index in the middle of the buffer
    buffer_offset = (uint) std::max((int) (index - buffer_size/(step*2)), 0);
    buffer_count = std::min(buffer_size / step, 
			    static_cast<uint>(bintree.size() - static_cast<uint>(buffer_offset)));
    file.seekg(buffer_offset*step);
    file.read( (char *) buffer, buffer_count*step);
  }

  // Read from buffer
  for (unsigned int i = 0; i < ode.size(); ++i) 
    u[i] = buffer[(index - buffer_offset)*ode.size() + i];
}
//-----------------------------------------------------------------------------
