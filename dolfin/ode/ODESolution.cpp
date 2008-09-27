// Copyright (C) 2008 Benjamin Kehlet
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-06-11
// Last changed: 2008-08-08

#include "ODESolution.h"
#include "Sample.h"
#include "ODE.h"
#include <algorithm>

using namespace dolfin;

//-----------------------------------------------------------------------------
ODESolution::ODESolution(ODE& ode) : 
  ode(ode),
  filename(tmpnam(0)),
  file(filename, std::ios::out | std::ios::binary),
  count(0),
  bintree(std::vector<real>()),
  step(sizeof(real)*ode.size()),
  dataondisk(false)
{

  //initalize the cache
  cachesize = ode.get("ODE order");
  std::string m = ode.get("ODE method") ;
  if (m == "dg") ++cachesize;

  cache = new std::pair<real, uBLASVector>[cachesize];
  for (uint i = 0; i < cachesize; ++i) {
    cache[i].first = -1;
    cache[i].second.resize(ode.size());
  }
  ringbufcounter = 0;

  //initialize buffer
  buffersize = ODESOLUTION_INITIAL_ALLOC - (ODESOLUTION_INITIAL_ALLOC % step);
  buffer = (real *) malloc(buffersize);
  bufferoffset = 0;

}
//-----------------------------------------------------------------------------
ODESolution::~ODESolution() {
  delete[] cache;
  file.close();
  remove(filename);
  free(buffer);
}

void ODESolution::addSample(Sample& sample) {
  
  bintree.push_back(sample.t());
  
  //check if there is allocated memory for another entry in the buffer
  if (step*(count+1) > buffersize) {
    //memory buffer not big enough

    //check if we can just extend the allocated memory
    if (buffersize*2 <= ODESOLUTION_MAX_ALLOC) {
      //extend the memory
      buffersize *= 2;
      buffer = (real *) realloc(buffer, buffersize);
      cout << "ODESolution: Reallocating memory. New size: " << buffersize << endl;
    } else {
      // No more memory available. Dump to disk
      cout << "ODESolution: Writing to disk" << endl;
      file.write((char *) buffer, step*count);
      bufferoffset += count;
      count = 0;
      dataondisk = true;
    }
  }

  for (uint i = 0; i < sample.size(); ++i) {
    buffer[count*ode.size()+i] = sample.u(i);
  }

  ++count;
}
//-----------------------------------------------------------------------------
void ODESolution::makeIndex() {

  if (dataondisk && count > 0) {
    cout << "ODESolution: Writing last chunk to disk" << endl;
    file.write((char *) buffer, count*step);
  }

  //cout << "Switching, bufferoffset: " << bufferoffset << ", count: " << count << ", size: " << size << endl;

  file.close();
  file.open(filename, std::ios::in | std::ios::binary);  
}
//-----------------------------------------------------------------------------
void ODESolution::eval(const real t, uBLASVector& y) {

  //scan the cache
  for (uint i = 0; i < cachesize; ++i) {
    if (cache[i].first < 0) {
      continue;
    }
    
    if (cache[i].first == t) {
      //found return cache[i]
      for (uint j = 0; j < ode.size(); j++) {
        uBLASVector& c = cache[i].second;
        y[j] = c[j];
      }
      //std::cout << "t=" << t << " " << std::flush;
      //printVector(y);

      return;
    }
  }

  //Not found in cache
  std::vector<real>::iterator low = std::lower_bound(bintree.begin(), 
						     bintree.end(), 
						     t);
  uint b = uint(low-bintree.begin());
  uint a = b-1;

  if (b >= bintree.size()) 
  {
    b = bintree.size() - 1;
    a = bintree.size() - 2;
  } 
  else if (b < 1) 
  {
    b = 1;
    a = 0;
  }

  real t_a = bintree[a];
  real t_b = bintree[b];

  //check if we need to read from disk
  if (a < bufferoffset || b > bufferoffset + count) {
    
    cout << "ODESolution: Fetching from disk" << endl;

    //printf("t=%f, a=%d \n", t, a);

    //put a in the middle of the buffer
    bufferoffset = (uint) std::max((int) (a - buffersize/(step*2)), 0);
    count = std::min(buffersize/step, bintree.size()-bufferoffset); 
    file.seekg(bufferoffset*step);
    file.read( (char *) buffer, count*step);

    //cout << "Bufferoffset: " << bufferoffset << ", count: " << count << endl;
    //exit(1);
  }

  uBLASVector tmp(ode.size());
  
  for (unsigned int i = 0; i < ode.size(); i++) {
    y[i]   = buffer[(a - bufferoffset)*ode.size() + i];
    tmp[i] = buffer[(b - bufferoffset)*ode.size() + i];
  }

  lerp(y, t_a, tmp, t_b, t, y);

  //cache y
  cache[ringbufcounter].first = t;
  for (uint i = 0; i < ode.size(); i++) {
    cache[ringbufcounter].second[i] = y[i];
  }
  ringbufcounter = (ringbufcounter+1)%cachesize;
  
  //std::cout << "t="<<t<<" " << std::flush;
  //printVector(y);

}
//-----------------------------------------------------------------------------
void ODESolution::lerp(const uBLASVector& v1, 
			      const real t1, 
			      const uBLASVector& v2, 
			      const real t2, 
			      const real t, 
			      uBLASVector& result) {
  real h = t2-t1;
  for (uint i = 0; i < ode.size(); i++) {
    result[i] = v1[i] + (t-t1)*((v2[i]-v1[i])/h);
  }
}
//-----------------------------------------------------------------------------
//REMOVE AFTER TESTING
void ODESolution::printVector(const uBLASVector& u) {
  for (unsigned int i=0; i < u.size(); i++) {
    printf("%.15f ", u[i]);
  }
  printf("\n");
}
