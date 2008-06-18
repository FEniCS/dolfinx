// Copyright (C) 2008 Benjamin Kehlet
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-06-11
// Last changed: 2008-06-17

#include "ODESolution.h"
#include "Sample.h"
#include "ODE.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
ODESolution::ODESolution(ODE& ode) : 
  ode(ode),
  filename(tmpnam(0)),
  file(filename, std::ios::out | std::ios::binary),
  count(0),
  bintree(std::vector<real>()),
  step(sizeof(real)*(ode.size()+1))
{

  //initalize the cache
  cachesize = ode.get("ODE order");
  std::string m = ode.get("ODE method") ;
  if (m == "dg") ++cachesize;

  cache = new std::pair<real, uBlasVector>[cachesize];
  for (uint i = 0; i < cachesize; ++i) {
    cache[i].first = -1;
    cache[i].second.init(ode.size());
  }
  ringbufcounter = 0;

}
//-----------------------------------------------------------------------------
ODESolution::~ODESolution() {
  delete[] cache;
  file.close();
  remove(filename);
}

void ODESolution::addSample(Sample& sample) {
  real tmp = sample.t();
  
  bintree.push_back(tmp);  

  file.write((char *) &tmp, sizeof(real));

  for (uint i = 0; i < sample.size(); ++i) {
    tmp = sample.u(i);
    file.write((char *) &tmp, sizeof(real));
  }

  ++count;
}
//-----------------------------------------------------------------------------
void ODESolution::makeIndex() {
  file.close();
  file.open(filename, std::ios::in | std::ios::binary);  
}
//-----------------------------------------------------------------------------
void ODESolution::eval(const real t, uBlasVector& y) {
  //cout << "eval(" << t << ")" << endl;

  //scan the cache
  for (uint i = 0; i < cachesize; ++i) {
    if (cache[i].first < 0) {
      continue;
    }
    
    if (cache[i].first == t) {
      //found return cache[i]
      for (uint j = 0; j < ode.size(); j++) {
        uBlasVector& c = cache[i].second;
        y[j] = c[j];
      }
      //std::cout << "t=" << t << " " << std::flush;
      //printVector(y);

      return;
    }
  }

  //Not found in cache

  std::vector<double>::iterator low = std::lower_bound(bintree.begin(), bintree.end(), t);
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
  uBlasVector tmp(ode.size());
  
  file.seekg(a*step+sizeof(real), std::ios_base::beg);
  for (unsigned int i = 0; i < ode.size(); i++) {
    real buf;
    file.read( (char *) &buf, sizeof(real));
    y[i] = buf;
  }

  file.seekg(b*step+sizeof(real), std::ios_base::beg);
  for (unsigned int i = 0; i < ode.size(); i++) {
    real buf;
    file.read( (char *) &buf, sizeof(real));
    tmp[i] = buf;
  }
       
  interpolate(y, t_a, tmp, t_b, t, y);

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
void ODESolution::interpolate(const uBlasVector& v1, 
			      const real t1, 
			      const uBlasVector& v2, 
			      const real t2, 
			      const real t, 
			      uBlasVector& result) {
  real h = t2-t1;
  for (uint i = 0; i < ode.size(); i++) {
    result[i] = v1[i] + (t-t1)*((v2[i]-v1[i])/h);
  }
}
//-----------------------------------------------------------------------------
//REMOVE AFTER TESTING
void ODESolution::printVector(const uBlasVector& u) {
  for (unsigned int i=0; i < u.size(); i++) {
    printf("%.15f ", u[i]);
  }
  printf("\n");
}
