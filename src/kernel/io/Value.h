#ifndef __VALUE_H
#define __VALUE_H

#include <stdio.h>
#include <iostream.h>
#include <assert.h>
#include <stdlib.h>

class Value
{
  
  friend std::ostream &operator<<(std::ostream &os, const Value &v){

	 std::cout << "t = " << v.t << ": [";
	 
	 for (int i=0;i<v.size;i++)
		std::cout << " " << v.values[i];
	 
	 std::cout << " ]";
	 return os;
  };
  
public:

  Value(int size);
  Value(const Value &v);

  ~Value();

  int    Size    ();
  double Time    ();
  double Get     (int pos);
  char  *Label   (int pos);
  
  void Set      (int pos, double val);
  void SetTime  (double t);
  void SetLabel (int pos, const char *string);
  
  bool Save(FILE *fp);
  bool Read(FILE *fp);
  
private:

  int size;
  double *values;
  char **labels;

  
  double t;
  
};

#endif
