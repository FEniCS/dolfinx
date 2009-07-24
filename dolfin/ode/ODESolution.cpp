// Copyright (C) 2008 Benjamin Kehlet
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg, 2008.
// Modified by Garth N. Wells, 2009.
//
// First added:  2008-06-11
// Last changed: 2009-07-12

#include "ODESolution.h"
#include "MonoAdaptiveTimeSlab.h"
#include <algorithm>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>

using namespace dolfin;

//-----------------------------------------------------------------------------
ODESolution::ODESolution(uint N) : 
  trial(0),
  N(N),
  nodal_size(0),
  T(0.0),
  initialized(false),
  read_mode(false),
  data_on_disk(false),
  filename("odesolution"),
  number_of_files(0)
{
  cout << "Creating ODESolution of size " << N << endl;
}
//-----------------------------------------------------------------------------
ODESolution::ODESolution(std::string filename, uint number_of_files) : 
  trial(0),
  N(N),
  nodal_size(0),
  T(0.0),
  initialized(false),
  data_on_disk(true)
{
  real tmp; 
  std::ifstream file(filename.c_str());
  file >> N;
  file >> T;
  file >> nodal_size;
  file >> tmp;

  trial = new Lagrange(nodal_size);

  for (uint i = 0; i < nodal_size; ++i) 
  {
    file >> tmp;
    trial->set(i, tmp);
  }

  initialized = true;

  file.close();


  //load the last file into memory
  read_file(number_of_files);

}
//-----------------------------------------------------------------------------
ODESolution::~ODESolution() 
{
  for (uint i = 0; i < cache_size; ++i)
    delete [] cache[i].second;
  delete [] cache;
}
//-----------------------------------------------------------------------------
void ODESolution::init(const Lagrange& trial_space) 
{
  if (initialized) error("ODESolution initialized twice");
  trial = new Lagrange(trial_space);
  nodal_size = trial->size();
  max_timeslabs = max_filesize/(real_decimal_prec()*(nodal_size*N + 2));

  // Initalize cache
  cache_size = nodal_size+1;
  cache = new std::pair<real, real*>[cache_size];
  for (uint i = 0; i < cache_size; ++i) 
  {
    cache[i].first = -1;
    cache[i].second = new real[N];
  }
  ringbufcounter = 0;

  initialized = true;

}
//-----------------------------------------------------------------------------
void ODESolution::add_timeslab(const real& a, const real& b, real* nodal_values) 
{
  //Public method. Does some checks and calls add_data
  if (!initialized) error("ODE Solution not initialized");
  if (read_mode) error("ODE Solution in read mode");
  assert(b-a > 0);
  assert(real_abs(T-a) < real_epsilon());

  if (data.size() > max_timeslabs) {
    save_to_file();
  }

  //cout << "Adding timeslab, a=" << a << ", b=" << b << endl;
  add_data(a, b, nodal_values);

  T = b;
}
//-----------------------------------------------------------------------------
void ODESolution::flush() {
  if (data_on_disk)
    save_to_file();

  read_mode = true;
}

//-----------------------------------------------------------------------------
void ODESolution::eval(real t, real* y) 
{
  //cout << "ODESolution::eval(" << t << ")" << endl;
  if (!read_mode) error("Can not evaluate solution");
  if(t > T) error("Requested t > T");
  
  // Scan the cache
  for (uint i = 0; i < cache_size; ++i) 
  {
    // Empty position, skip
    if (cache[i].first < 0)
      continue;
    
    // Copy values
    if (cache[i].first == t) 
    {
      real_set(N, y, cache[i].second);
      return;
    }
  }
  


  if (data_on_disk && (t < a_in_memory() || t > b_in_memory()))
  {
    // need to fetch data from disk. Get file number
    std::vector<real>::iterator lower = std::upper_bound(file_table.begin(),
							 file_table.end(),
							 t);
    uint index = lower-file_table.begin()-1;
    
    read_file(index);
  }

  // Find position in buffer
  std::vector<Timeslabdata>::iterator lower = std::upper_bound(data.begin(), 
  							       data.end(), 
							       t,
							       t_cmp);
  uint index = lower-data.begin()-1;

  Timeslabdata a = data[index];
  real t_a = std::tr1::get<0>(a);
  real k = std::tr1::get<1>(a);
  real* values = std::tr1::get<2>(a);
  real tau = (t-t_a)/k;

  //cout << "a = " << t_a << ", k = " << k << ", tau = " << tau << endl;

  assert(tau <= 1.0+real_epsilon());


  for (uint i = 0; i < N; ++i) 
  {
    y[i] = 0.0;

    for (uint j = 0; j < nodal_size; j++)
      y[i] += values[i*nodal_size + j] * trial->eval(j, tau);
  }

  //store in cache
  cache[ringbufcounter].first = t;
  real_set(N, cache[ringbufcounter].second, y);
  ringbufcounter = (ringbufcounter + 1) % cache_size;
}
//----------------------------------------------------------------------------- 
void ODESolution::set_filename(std::string filename)
{
  if (data_on_disk)
    error("Filename cannot be changed after data is written to file");

  this->filename = filename;
}
//----------------------------------------------------------------------------- 
void ODESolution::save_to_file() 
{
  //if (read_mode) error("Can not save to file in read mode");

  if (data.size() == 0) return;

  std::stringstream f(filename, std::ios_base::app | std::ios_base::out);
  
  if (number_of_files > 0) 
  {
    f << "_" << (number_of_files);
  }

  //cout << "Saving solution data to file: " << f.str() << ". a = " << a_in_memory() << ", b = " << b_in_memory() << endl;

  file_table.push_back(a_in_memory());

  std::ofstream file(f.str().c_str(), std::ios::out);
  file << std::setprecision(real_decimal_prec());
  
  // write T, the number of nodal points and the decimal precision to the file
  file << N << " " << " " << nodal_size << " " << real_decimal_prec() << std::endl;

  // write the nodal points
  for (uint i = 0; i < nodal_size; ++i) 
  {
    file << trial->point(i) << " ";
  }
  file << std::endl;

  file << "end_of_header" << std::endl;

  // then write the timeslab data
  for (std::vector<Timeslabdata>::iterator it = data.begin(); it != data.end(); ++it) 
  {
    real& t = std::tr1::get<0>(*it);
    real& k = std::tr1::get<1>(*it);
    real* values = std::tr1::get<2>(*it);

    file << std::setprecision(real_decimal_prec()) << t << " " << k << " ";
    for (uint i = 0; i < N*nodal_size; ++i) 
    {
	file << values[i] << " ";
    }
    file << std::endl;
    delete [] values;
  }

  file.close();

  number_of_files++;
  data.clear();
  data_on_disk = true;
  
}
//-----------------------------------------------------------------------------
void ODESolution::read_file(uint file_number)
{

  if (data.size() > 0) 
  {
    for (std::vector<Timeslabdata>::iterator it = data.begin(); it != data.end(); ++it) 
    {
      real* values = std::tr1::get<2>(*it);
      delete [] values;
    }
    data.clear();
  }

  std::stringstream f(filename, std::ios_base::app | std::ios_base::out);
  if (file_number > 0) {
    f << "_" << file_number;
  }

  std::ifstream file(f.str().c_str());


  //skip the header
  uint nodal_size;
  real tmp; 
  file >> nodal_size; // variable used for temporary storage
  file >> nodal_size;
  file >> tmp; //reading precision

  for (uint i = 0; i < nodal_size; ++i) 
    file >> tmp;

  std::string marker;
  file >> marker;
  
  real a;
  real k;
  real values[nodal_size*N];

  while (true) {
    file >> a;
    file >> k;

    if (file.eof()) break;

    for (uint i = 0; i < N*nodal_size; ++i) 
    {
      file >> values[i];
    }

    add_data(a, a+k, values);
  }  

  file.close();
  //cout << "Done fetching from file " << file_number << ". In memory: " << a_in_memory() << " <--> " << b_in_memory() << endl;
}
//-----------------------------------------------------------------------------
void ODESolution::add_data(const real& a, const real& b, real* nodal_values) 
{
  //Private method. Called from either add_timeslab or read_file

  //allocate memory and copy nodal values
  real* nv = new real[nodal_size*N];
  real_set(nodal_size*N, nv, nodal_values);

  data.push_back(Timeslabdata(a, 
			      b-a,
			      nv));

}
//----------------------------------------------------------------------------- 
bool ODESolution::t_cmp(const real& t, const Timeslabdata& a) {
  real a_t = std::tr1::get<0>(a);
  return (t < a_t);
}
//----------------------------------------------------------------------------- 
void ODESolution::disp() 
{
  cout << "--- ODE solution ------------------------------" << endl;
  cout << "Size = " << N << endl;
  cout << "T = " << T << endl;
  cout << "Number of nodal points = " << nodal_size << endl;
  cout << "Nodal points: ";
  for (uint i = 0; i < nodal_size; i++)
    cout << " " << trial->point(i);
  cout << endl;
  cout << "Number of timeslabs = " << (uint) data.size() << endl;
  cout << "----------------------------------------------------------" << endl;  
}
