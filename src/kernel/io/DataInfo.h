#ifndef __DATA_INFO_H
#define __DATA_INFO_H

class DataInfo{
public:

  DataInfo(const char *description, int no_data, int *dimensions);
  ~DataInfo();

  void SetLabel(int i, const char *name, const char *label);
  
  int Size();
  int Dim(int i);

  const char *Description();
  const char *Label(int i);
  const char *Name(int i);
  
private:

  int    no_data;
  int   *dimensions;
  char  *description;
  char **labels;
  char **names;  

};

#endif
