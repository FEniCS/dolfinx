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
ODESolution::ODESolution() : 
  trial(0),
  N(0),
  nodal_size(0),
  T(0.0),
  no_timeslabs(0),
  quadrature_weights(0),
  initialized(false),
  read_mode(false),
  data_on_disk(false),
  filename("odesolution"),
  number_of_files(0)
{
}
//-----------------------------------------------------------------------------
ODESolution::ODESolution(std::string filename, uint number_of_files) : 
  trial(0),
  N(0),
  nodal_size(0),
  T(0.0),
  no_timeslabs(0),
  initialized(false),
  data_on_disk(true),
  filename(filename)
{
  cout << "Reading ODESolution from file" << endl;

  real tmp; 
  std::ifstream file;
  no_timeslabs = open_and_read_header(file, 0u);
  file.close();


  //collect number of timeslabs and first t value from all files
  for (uint i=1; i < number_of_files; i++) 
  {
    std::ifstream file;
    uint size = open_and_read_header(file, i);
    real t;
    file >> t;
    file_table.push_back(t);
    file.close();

    file_offset_table.push_back(no_timeslabs);
    no_timeslabs += size;
  }

  cout << "Timeslabs: " << no_timeslabs << endl;

  uint count=0;
  for (std::vector<real>::iterator it=file_table.begin(); it != file_table.end(); it++)  {
    cout <<  "Filetable: " << count << ": " << *it << endl;
    count++;
  }

  //load the last file into memory
  read_file(number_of_files-1);

  ODESolution_data& last = data[data.size()-1];
  T = last.a+last.k;
  read_mode = true;

  cout << "Done reading ODESolution from file. T=" << T << endl;
}
//-----------------------------------------------------------------------------
ODESolution::~ODESolution() 
{
  for (uint i = 0; i < cache_size; ++i)
    delete [] cache[i].second;
  delete [] cache;

  if (quadrature_weights) delete [] quadrature_weights;
  if (trial) delete trial;
}
//-----------------------------------------------------------------------------
void ODESolution::init(uint N, const Lagrange& trial_space, const real* quad_weights) 
{
  if (initialized) error("ODESolution initialized twice");

  this->N = N;

  trial = new Lagrange(trial_space);
  nodal_size = trial->size();
  max_timeslabs = max_filesize/(real_decimal_prec()*(nodal_size*N + 2));

  quadrature_weights = new real[nodal_size];
  real_set(nodal_size, quadrature_weights, quad_weights);

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
void ODESolution::add_timeslab(const real& a, const real& b, const real* nodal_values) 
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

  no_timeslabs++;
  T = b;
}
//-----------------------------------------------------------------------------
void ODESolution::flush() {
  if (data_on_disk)
    save_to_file();

  read_mode = true;
}

//-----------------------------------------------------------------------------
void ODESolution::eval(const real& t, real* y) 
{
  //cout << "ODESolution::eval(" << t << ")" << endl;
  if (!read_mode) error("Can not evaluate solution");
  if(t > T) error("Requested t > T. t=%f, T=%f", to_double(t), to_double(T));
  
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
  
  // Read data from disk if necesary.
  if (data_on_disk && (t < a_in_memory() || t > b_in_memory()))
  {
    //get file number
    std::vector<real>::iterator lower = std::upper_bound(file_table.begin(),
							 file_table.end(),
							 t);
    uint index = lower-file_table.begin();
    
    read_file(index);
  }

  // Find position in buffer
  std::vector<ODESolution_data>::iterator lower = std::upper_bound(data.begin(), 
  							       data.end(), 
							       t,
							       t_cmp);
  uint index = lower-data.begin()-1;

  ODESolution_data& a = data[index];
  real tau = (t-a.a)/a.k;

  //cout << "a = " << t_a << ", k = " << k << ", tau = " << tau << endl;

  assert(tau <= 1.0+real_epsilon());


  for (uint i = 0; i < N; ++i) 
  {
    y[i] = 0.0;

    for (uint j = 0; j < nodal_size; j++)
      y[i] += a.nv[i*nodal_size + j] * trial->eval(j, tau);
  }

  //store in cache
  cache[ringbufcounter].first = t;
  real_set(N, cache[ringbufcounter].second, y);
  ringbufcounter = (ringbufcounter + 1) % cache_size;
}
//----------------------------------------------------------------------------- 
ODESolution_data& ODESolution::get_timeslab(uint index) 
{
  if (index >= no_timeslabs)
    error("Requested timeslabs %u out of range %u", index, no_timeslabs);

  if (data_on_disk && index > a_index+data.size())
  {
    std::vector<uint>::iterator lower = std::upper_bound(file_offset_table.begin(),
							 file_offset_table.end(),
							 index);
    uint fileno = lower-file_offset_table.begin();
    
    read_file(fileno);
  }

  return data[index-a_index];
}
//----------------------------------------------------------------------------- 
const real* ODESolution::get_weights() const {
  return quadrature_weights;
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
  if (data.size() == 0) return;

  std::stringstream f(filename, std::ios_base::app | std::ios_base::out);  
  if (number_of_files > 0) 
  {
    f << "_" << (number_of_files);
  }

  cout << "Saving solution data to file: " << f.str() << ". a = " << a_in_memory() << ", b = " << b_in_memory() << endl;

  file_table.push_back(a_in_memory());
  file_offset_table.push_back(no_timeslabs - data.size());

  std::ofstream file(f.str().c_str());
  if (!file.is_open()) 
    error("Unable to open file: %s", f.str().c_str());

  file << std::setprecision(real_decimal_prec());
  
  // write number of timeslabs, size of system, the number of nodal points and the decimal precision to the file
  file  << data.size() << " " << N << " " << nodal_size << " " << real_decimal_prec() << std::endl;

  // write the nodal points
  for (uint i = 0; i < nodal_size; ++i) 
  {
    file << trial->point(i) << " ";
  }
  file << std::endl;

  // write the nodal weights
  for (uint i = 0; i < nodal_size; ++i) 
  {
    file << quadrature_weights[i] << " ";
  }
  file << std::endl;

  file << "end_of_header" << std::endl;

  // then write the timeslab data
  for (std::vector<ODESolution_data>::iterator it = data.begin(); it != data.end(); ++it) 
  {
    //real& t = std::tr1::get<0>(*it);
    //real& k = std::tr1::get<1>(*it);
    //real* values = std::tr1::get<2>(*it);

    file << std::setprecision(real_decimal_prec()) << it->a << " " << it->k << " ";
    for (uint i = 0; i < N*nodal_size; ++i) 
    {
	file << it->nv[i] << " ";
    }
    file << std::endl;
  }

  file.close();

  number_of_files++;
  data.clear();
  data_on_disk = true;
  
}
//-----------------------------------------------------------------------------
dolfin::uint ODESolution::open_and_read_header(std::ifstream& file, uint filenumber)
{
  std::stringstream f(filename, std::ios_base::app | std::ios_base::out);
  if (filenumber > 0)
    f << "_" << filenumber;

  file.open(f.str().c_str());
  if (!file.is_open())
    error("Unable to read file: %s", f.str().c_str());

  uint timeslabs;
  file >> timeslabs;

  uint tmp;
  real tmp_real;

  if (initialized) {
    file >> tmp;
    if (tmp != N) 
      error("Wrong N size of system in file: %s", f.str().c_str());
    file >> tmp;
    if (tmp != nodal_size) 
      error("Wrong nodal size in file: %s", f.str().c_str());    
    file >> tmp;

    //skip nodal points and quadrature weights
    for (uint i=0; i < nodal_size*2; ++i)
      file >> tmp_real;
  } else {
    uint _N;
    file >> _N;
    file >> nodal_size;
    file >> tmp;

    //read nodal points
    Lagrange l(nodal_size-1);
    for (uint i=0; i < nodal_size; ++i) {
      file >> tmp_real;
      l.set(i, tmp_real);
    }

    real q_weights[nodal_size];
    for (uint i=0; i < nodal_size; ++i) {
      file >> q_weights[i];
    }

    init(_N, l, q_weights);
  }

  std::string marker;
  file >> marker;
  if (marker != "end_of_header")
    error("in file %s: End of header marker: %s", f.str().c_str(), marker.c_str());

  return timeslabs;
}
//-----------------------------------------------------------------------------
void ODESolution::read_file(uint file_number)
{

  if (data.size() > 0) 
    data.clear();


  //open file and read the header
  std::ifstream file;
  uint timeslabs = open_and_read_header(file, file_number);
  
  real a;
  real k;
  real values[nodal_size*N];

  uint count = 0;
  while (true) {
    file >> a;
    file >> k;

    if (file.eof()) break;

    for (uint i = 0; i < N*nodal_size; ++i) 
    {
      file >> values[i];
    }

    add_data(a, a+k, values);
    count++;
  }

  file.close();

  if (count != timeslabs)
    error("File data in file %u inconsistent with header. Header said: %d, read %d", 
	  file_number, timeslabs, count);

  //cout << "Done fetching from file " << file_number << 
  //  ". In memory: " << a_in_memory() << " <---> " << b_in_memory() << endl;
}
//-----------------------------------------------------------------------------
void ODESolution::add_data(const real& a, const real& b, const real* nodal_values) 
{
  //Private method. Called from either add_timeslab or read_file
  data.push_back(ODESolution_data(a, 
				  b-a,
				  nodal_size,
				  N,
				  nodal_values));
}
//----------------------------------------------------------------------------- 
bool ODESolution::t_cmp(const real& t, const ODESolution_data& a) {
  return (t < a.a);
}
//----------------------------------------------------------------------------- 
void ODESolution::disp() 
{
  cout << "--- ODE solution ------------------------------" << endl;
  if (initialized)
  {
    cout << "Size = " << N << endl;
    cout << "T = " << T << endl;
    cout << "Number of nodal points = " << nodal_size << endl;
    cout << "Nodal points: ";
    for (uint i = 0; i < nodal_size; i++)
      cout << " " << trial->point(i);
    cout << endl;
    cout << "Number of timeslabs = " << (uint) data.size() << endl;
  } else 
  {
    cout << "Not initialized" << endl;
  }

  cout << "----------------------------------------------------------" << endl;  
}
//----------------------------------------------------------------------------- 
ODESolution::iterator ODESolution::begin()
{
  return ODESolution::iterator(*this);
}
//----------------------------------------------------------------------------- 
ODESolution::iterator ODESolution::end()
{
  return ODESolution::iterator(*this, no_timeslabs);
}

