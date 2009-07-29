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
#include <iterator>
#include <utility>
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


  class Lagrange;
  class ODESolution_iterator;


//----------------------------------------------------------------------------- 
//-------------------------------- timeslabdata -------------------------------
//----------------------------------------------------------------------------- 
  class ODESolution_data
  {

  public:
  ODESolution_data(const real& a, const real& k, uint nodal_size, uint N, const real* values) :
    a(a), k(k) 
    { 
      size = nodal_size*N;
      nv = new real[nodal_size*N];
      real_set(size, nv, values); 
    }

   //copy constructor
    ODESolution_data(const ODESolution_data& cp)
    {
      a = cp.a;
      k = cp.k;
      size = cp.size;
      nv = new real[size];
      real_set(size, nv, cp.nv);
    }
    
    ~ODESolution_data() {
      delete [] nv; 
    }

    inline real b() {return a+k;}

    real a;
    real k;
    real* nv;
    uint size;

  };

//----------------------------------------------------------------------------- 
//-------------------------------- ODESolution---------------------------------
//----------------------------------------------------------------------------- 

  class ODESolution
  {
  public:

    /// Create solution data for given ODE
    ODESolution();
    ODESolution(std::string filename, uint number_of_files = 1); //load data from file

    /// Destructor
    ~ODESolution();

    //must be called before starting to add data
    void init(uint N, const Lagrange& trial, const real* quad_weights);

    // Add solution data. Must be in write_mode
    void add_timeslab(const real& a, const real& b, const real* values);

    /// Make object ready for evaluating, set to read mode
    void flush();

    /// Evaluate (interpolate) value of solution at given time    
    void eval(const real& t, real* y);

    /// Get timeslab (used when iterating)
    ODESolution_data& get_timeslab(uint index);

    /// Get pointer to weights
    const real* get_weights() const;

    void set_filename(std::string filename);
    void save_to_file();

    void disp();

    inline uint size() const {return no_timeslabs;}
    inline uint nsize() const {return nodal_size;}
    inline real endtime() const {return T;}

    //iterator
    typedef ODESolution_iterator iterator;
    iterator begin();
    iterator end();

  private:
    Lagrange* trial;
    uint N; //number of components
    uint nodal_size;
    real T; //endtime. Updated when new timeslabs are added 

    uint no_timeslabs;
    std::vector<ODESolution_data> data; //data in memory
    std::vector<real> file_table; //table mapping t values to files
    std::vector<uint> file_offset_table;
    uint fileno_in_memory; //which file is currently in memory
    uint a_index; //offset of memory

    real* quadrature_weights;

    //cache
    std::pair<real, real*>* cache;    
    uint cache_size;
    uint ringbufcounter;

    bool initialized;
    bool read_mode;

    // Stuff related to file storage
    static const uint max_filesize = 1000000000; //approx 1GB
    bool data_on_disk;
    uint max_timeslabs; //number of timeslabs pr file and in memory
    std::string filename;
    int number_of_files;
    void read_file(uint file_number);
    dolfin::uint open_and_read_header(std::ifstream& file, uint filenumber); 

    void add_data(const real& a, const real& b, const real* data);

    inline real a_in_memory() {return data[0].a;}
    inline real b_in_memory() {return data[data.size()-1].a + data[data.size()-1].k;}

    //callback function used by std::lower_bound() when searching
    static bool t_cmp( const real& t, const ODESolution_data& a);
  };

//----------------------------------------------------------------------------- 
//-------------------------------- iterator -----------------------------------
//----------------------------------------------------------------------------- 
  class ODESolution_iterator : 
    public std::iterator<std::input_iterator_tag, ODESolution_data*>
  {
  public:
    ODESolution_iterator(ODESolution& u) : u(u), index(0) {}
    ODESolution_iterator(ODESolution& u, uint index) : u(u), index(index) {}
    ODESolution_iterator(const ODESolution_iterator& it) :   
      u(it.get_ODESolution()), index(it.get_index()) {}

    ODESolution_iterator& operator++() {++index;return *this;}
    void operator++(int) {++index;}

    uint get_index() const {return index;}

    ODESolution& get_ODESolution() const {return u;};

    bool operator==(const ODESolution_iterator& rhs) {return index == rhs.get_index();}
    bool operator!=(const ODESolution_iterator& rhs) {return index != rhs.get_index();}

    ODESolution_data& operator*() {return u.get_timeslab(index);}

  private:
    ODESolution& u;
    uint index;

  };


}
#endif
