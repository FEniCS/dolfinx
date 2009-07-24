// Copyright (C) 2008 Benjamin Kehlet
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg, 2008.
// 
// First added:  2008-06-11
// Last changed: 2009-07-12

#ifndef __ODESOLUTION_H
#define __ODESOLUTION_H

#include <vector>
#include <tr1/tuple>
#include <dolfin/common/types.h>
#include <dolfin/common/real.h>
#include "Method.h"

namespace dolfin
{

  /// ODESolution stores the solution from the ODE solver, primarily to
  /// be able to solve the dual problem. A number of interpolated values 
  /// is cached, since the ODE solver repeatedly requests evaluation of 
  /// the same t.
  /// 
  /// The samples are stored in memory if possible, otherwise stored
  /// in a temporary file and fetched from disk in blocks when needed.
  ///
  /// Since GMP at the moment doesn't support saving binary operands
  /// on disk this class uses ascii files for storage. 
  /// Storing binary operands on disk is fortunately planned on the next major 
  /// release of GMP.

  //a, k, nodal values
  typedef std::tr1::tuple<real, real, real*> Timeslabdata;

  class Lagrange;

  class ODESolution
  {
  public:

    /// Create solution data for given ODE
    ODESolution(uint N);
    ODESolution(std::string filename, uint number_of_files = 1); //load data from file

    /// Destructor
    ~ODESolution();

    //set the trial space (must be called before starting to add data)
    void init(const Lagrange& trial);

    // Add solution data. Must be in write_mode
    void add_timeslab(const real& a, const real& b, real* values);

    /// Make object ready for evaluating
    void flush();

    /// Evaluate (interpolate) value of solution at given time    
    void eval(const real t, real* y);

    void set_filename(std::string filename);
    void save_to_file();

    void disp();

    inline uint nsize() const {return nodal_size;}
    inline real endtime() const {return T;}

  private:
    Lagrange* trial;
    uint N; //number of components
    uint nodal_size;
    real T; //endtime. Updated when new timeslabs are added 
    std::vector<Timeslabdata> data; //data in memory
    std::vector<real> file_table; //table mapping t values to files

    std::pair<real, real*>* cache;    
    uint cache_size;
    uint ringbufcounter;

    bool initialized;
    bool read_mode; //true when 

    // Stuff related to file storage
    bool data_on_disk;
    uint max_timeslabs; //number of timeslabs pr file and in memory
    std::string filename;
    int number_of_files;

    static const uint max_filesize = 1000000000; //approx 1GB

    void read_file(uint file_number);
    void add_data(const real& a, const real& b, real* data);

    inline real a_in_memory() {return std::tr1::get<0>(data[0]);}
    inline real b_in_memory() {return std::tr1::get<0>(data[data.size()-1])+std::tr1::get<1>(data[data.size()-1]);}

    //callback function used by std::lower_bound() when searching
    static bool t_cmp( const real& t, const Timeslabdata& a);
  };
}


#endif
