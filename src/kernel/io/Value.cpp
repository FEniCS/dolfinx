#include "Value.h"
#include "kw_constants.h"

//-----------------------------------------------------------------------------
Value::Value(int size)
{
  assert(size > 0);
  
  this->size = size;
  values = new double[size];

  for (int i=0;i<size;i++)
	 values[i] = 0.0;

  labels = new (char *)[size];
  for (int i=0;i<size;i++)
	 labels[i] = new char[DOLFIN_WORDLENGTH];
  
  t = 0.0;
}
//-----------------------------------------------------------------------------
Value::~Value()
{
  delete values;

  for (int i=0;i<size;i++)
	 delete labels[i];
  delete labels;
}
//-----------------------------------------------------------------------------
int Value::Size()
{
  return ( size );
}
//-----------------------------------------------------------------------------
double Value::Time()
{
  return ( t );
}
//-----------------------------------------------------------------------------
double Value::Get(int pos)
{
  assert(pos >= 0);
  assert(pos < size);
  
  return ( values[pos] );
}
//-----------------------------------------------------------------------------
char *Value::Label(int pos)
{
  assert(pos >= 0);
  assert(pos < size);

  return ( labels[pos] );
}
//-----------------------------------------------------------------------------
void Value::Set(int pos, double val)
{
  assert(pos >= 0);
  assert(pos < size);
  
  values[pos] = val;
}
//-----------------------------------------------------------------------------
void Value::SetTime(double t)
{
  this->t = t;
}
//-----------------------------------------------------------------------------
void Value::SetLabel(int pos, const char *string)
{
  for (int i=0;i<DOLFIN_WORDLENGTH;i++)
	 labels[pos][i] = string[i];
  labels[pos][DOLFIN_WORDLENGTH-1] = '\0';
}
//-----------------------------------------------------------------------------
bool Value::Save(FILE *fp)
{
  fprintf(fp,"%1.16e",t);

  // Binary
  //for (int i=0;i<size;i++)
  //	 fwrite(values+i,sizeof(double),1,fp);
  
  // ASCII
  for (int i=0;i<size;i++)
	 fprintf(fp," %1.16e",values[i]);

  fprintf(fp,"\n");

  return true;
}
//-----------------------------------------------------------------------------
bool Value::Read(FILE *fp)
{
  char c;
  double time, x;
  char line[32];
  
  // Step to t=...
  //while ( (c=getc(fp)) != EOF )
  //	 if ( c == 't' )
  //		break;

  // Check if it worked
  //if ( c != 't' )
  //	 return false;

  // Next should be '='
  //if ( (c=getc(fp)) != '=' )
  //	 return false;

  // Read time value
  //fgets(line,32,fp);
  //t = atof(line);

  // Read values
  //for (int i=0;i<size;i++){
  //	 if ( !fread(&x,sizeof(double),1,fp) )
  //		return false;
  //	 values[i] = x;
  // }

  fscanf(fp,"%e",&x);
  for (int i=0;i<(size-1);i++){
	 fscanf(fp," %e",&x);
	 values[i] = x;
  }
  fscanf(fp," %e\n",&x);
  values[size-1] = x;
  
  return true;
}
//-----------------------------------------------------------------------------
