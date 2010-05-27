// Copyright (C) 2008 Benjamin Kehlet
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg, 2008.
// Modified by Garth N. Wells, 2009.
//
// First added:  2008-06-11
// Last changed: 2009-11-11

#include "ODESolution.h"
#include "MonoAdaptiveTimeSlab.h"
#include <algorithm>
#include <iostream>
#include <ios>
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
  dirty(false),
  filename("odesolution")
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
  dirty(false),
  filename(filename)
{
  std::ifstream file;
  uint timeslabs_in_file = open_and_read_header(file, 0u);
  file.close();

  //collect number of timeslabs and first t value from all files
  for (uint i=0; i < number_of_files; i++)
  {
    std::ifstream file;
    timeslabs_in_file = open_and_read_header(file, i);
    real t;
    file >> t;
    file_table.push_back( std::pair<real, uint>(t, no_timeslabs) );

    no_timeslabs += timeslabs_in_file;

    // if this is the last file, read the last line to extract the 
    // endtime value
    if (i == number_of_files-1) 
    {
      // read backwards from the end of the file to find the last
      // newline
      // FIXME: Is there a better way to do this? Some library function 
      // doing the same as the command tail
      char buf[1001];
      file.seekg (0, std::ios::end);
      int pos = file.tellg();
      pos -= 1001;
      file.seekg(pos, std::ios::beg);
      while (true) 
      {
	file.read(buf, 1000);
	buf[1000] = '\0';

	std::string buf_string(buf);
	size_t newline_pos =  buf_string.find_last_of("\n");
	if (newline_pos != std::string::npos) 
	{
	  file.seekg(pos + newline_pos);
	  real max_a;
	  real k;
	  file >> max_a;
	  file >> k;
	  T = max_a + k;
	  break;
	} else 
	{
	  pos -= 1000;
	  file.seekg(pos, std::ios::beg);

	}
      }
    }

    file.close();    
  }

  // save an invalid fileno and dummy timeslabs to trigger reading 
  // a file on first eval
  fileno_in_memory = file_table.size() + 1;
  data.push_back(ODESolutionData(-1, 0, 0, 0, NULL));
  data.push_back(ODESolutionData(-2, 0, 0, 0, NULL));

  // load the last file into memory
  //read_file(file_table.size()-1);
  

  //ODESolutionData& last = data[data.size()-1];
  //T = last.a+last.k;
  read_mode = true;
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
    data.clear();
  }

  //cout << "Adding timeslab, a=" << a << ", b=" << b << endl;
  add_data(a, b, nodal_values);

  dirty = true;
  no_timeslabs++;
  T = b;
}
//-----------------------------------------------------------------------------
void ODESolution::flush() {
  if (read_mode) error("Cannot flush. ODESolution already in read mode");
  if (data_on_disk)
  {
    save_to_file();
  }

  read_mode = true;
}

//-----------------------------------------------------------------------------
void ODESolution::eval(const real& t, real* y)
{
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
    std::vector< std::pair<real, uint> >::iterator lower = std::lower_bound(file_table.begin(),
							       file_table.end(),
							       t,
							       real_filetable_cmp);
    uint index = lower-file_table.begin()-1;

    read_file(index);
  }

  // Find position in buffer
  std::vector<ODESolutionData>::iterator lower = std::lower_bound(data.begin(),
								  data.end(),
								  t,
								  real_data_cmp);
  uint index = lower-data.begin()-1;

  ODESolutionData& a = data[index];
  real tau = (t-a.a)/a.k;

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
ODESolutionData& ODESolution::get_timeslab(uint index)
{

  if (index >= no_timeslabs)
    error("Requested timeslabs %u out of range %u", index, no_timeslabs);

  if ( data_on_disk && (index > b_index_in_memory() || index < a_index_in_memory()))
  {
    std::vector< std::pair<real, uint> >::iterator lower = std::lower_bound(file_table.begin(),
									    file_table.end(),
									    index,
									    uint_filetable_cmp);
    uint fileno = lower-file_table.begin();

    read_file(fileno);
  }

  assert(index-a_index_in_memory() < data.size());


  return data[index - a_index_in_memory()];
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
  if (!dirty) {
    return;
  }

  std::stringstream f(filename, std::ios_base::app | std::ios_base::out);
  if (file_table.size() > 0)
  {
    f << "_" << ( file_table.size());
  }

  file_table.push_back( std::pair<real, uint> (a_in_memory(), no_timeslabs - data.size()) );

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
  for (std::vector<ODESolutionData>::iterator it = data.begin(); it != data.end(); ++it)
  {
    file << std::setprecision(real_decimal_prec()) << it->a << " " << it->k << " ";
    for (uint i = 0; i < N*nodal_size; ++i)
    {
	file << it->nv[i] << " ";
    }
    file << std::endl;
  }
  file.close();

  data_on_disk = true;
  fileno_in_memory = file_table.size()-1;
  dirty = false;

  // Note: Don't clear data from memory. Caller should take care of this

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

  fileno_in_memory = file_number;

  //cout << "Done reading file" << file_number << endl;

}
//-----------------------------------------------------------------------------
void ODESolution::add_data(const real& a, const real& b, const real* nodal_values)
{
  //Private method. Called from either add_timeslab or read_file
  data.push_back(ODESolutionData(a,
				 b-a,
				 nodal_size,
				 N,
				 nodal_values));
}
//-----------------------------------------------------------------------------
bool ODESolution::real_data_cmp( const ODESolutionData& a, const real& t ) {
  return (a.a <= t);
}
//-----------------------------------------------------------------------------
bool ODESolution::real_filetable_cmp(const std::pair<real, uint>& a, const real& t) {
  return (a.first <= t);
}
//-----------------------------------------------------------------------------
bool ODESolution::uint_filetable_cmp(const std::pair<real, uint>& a, const uint& i) {
  return ( a.second < i);
}
//-----------------------------------------------------------------------------
std::string ODESolution::str(bool verbose) const
{
  std::stringstream s;

  if (!initialized)
  {
    s << "ODESolution: Not initialized";
  } else
  {
    if (verbose)
    {

      s << "Size = " << N << std::endl;
      s << "T = " << T << std::endl;
      s << "Number of nodal points = " << nodal_size << std::endl;
      s << "Nodal points: ";
      for (uint i = 0; i < nodal_size; i++)
	s << " " << trial->point(i);
      s << std::endl;
      s << "Number of timeslabs = " << (uint) data.size() << std::endl;
    } else
    {
      s << "<ODESolution of size" << N << " on interval [0,"<< endtime() << "]>";
    }
  }

  return s.str();
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
//-----------------------------------------------------------------------------

