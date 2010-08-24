#include <stdio.h>
#include <stdlib.h>

void checkerror(int error);

int main(int argc, char **argv)
{
  FILE *fp_inp; /* inp file pointer */
  FILE *fp_dx;  /* dx file pointer */
  int npoints;  /* number of points */
  int nelems;   /* number of elements */
  int nvars;    /* number of variables */
  int nconns;   /* number of connections */
  int dummy1, dummy2;
  float x,y,z;
  int p1,p2,p3,p4;
  int i,j;
  char element_type[64];
  float *data;
  int   *objects_size;
  int    objects_n;
  
  /* Check arguments */
  if ( argc != 3 ){
	 printf("Usage: inp2dx datafile.inp datafile.dx\n");
	 exit(1);
  }

  /* Open inp file */
  fp_inp = fopen(argv[1],"r");
  if ( !fp_inp ){
	 printf("Unable to open inp file %s for reading.\n",argv[1]);
	 exit(1);
  }

  /* Open dx file */
  fp_dx = fopen(argv[2],"w");
  if ( !fp_dx ){
	 printf("Unable to open dx file %s for writing.\n",argv[2]);
	 exit(1);
  }
  
  /* Read data from inp file */
  fscanf(fp_inp,"%d %d %d %d %d\n",&npoints,&nelems,&nvars,&dummy1,&dummy2);
  printf("Number of points:    %d\n",npoints);
  printf("Number of elements:  %d\n",nelems);
  printf("Number of variables: %d\n",nvars);

  /* Allocate memory for data */
  data         = (float *) malloc( nvars*sizeof(float) );
  objects_size = (int *)   malloc( nvars*sizeof(int) );
  
  /* A warning */
  printf("I assume the elements are tetrahedrons...\n");
  
  /* Positions */
  printf("Reading positions...");
  fflush(stdout);
  fprintf(fp_dx,"# A list of all node positions\n");
  fprintf(fp_dx,"object 1 class array type float rank 1 shape 3 items %d data follows\n",npoints);
  
  for (i=0;i<npoints;i++){
	 
	 /* Read values */
	 fscanf(fp_inp,"%d %f %f %f\n",&dummy1,&x,&y,&z);
	 
	 if ( i % (npoints/20) == 0 ){
		printf(".");
		fflush(stdout);
	 }
	 
	 /* Save values */
	 fprintf(fp_dx,"%f %f %f\n",x,y,z);
	 
  }  printf("\n");
  
  /* Elements */
  printf("Reading elements....");
  fflush(stdout);
  fprintf(fp_dx,"\n# A list of all elements\n");
  fprintf(fp_dx,"object 2 class array type int rank 1 shape 4 items %d data follows\n",nelems);
  for (i=0;i<nelems;i++){
	 
	 /* Read values */
	 fscanf(fp_inp,"%d %d %s",&dummy1,&dummy2,&element_type);
	 fscanf(fp_inp,"%d %d %d %d\n",&p1,&p2,&p3,&p4);
	 
	 if ( i % (nelems/20) == 0 ){
		printf(".");
		fflush(stdout);
	 }
	 
	 /* Save values */
	 fprintf(fp_dx,"%d %d %d %d\n",p1-1,p2-1,p3-1,p4-1);
	 
  }
  fprintf(fp_dx,"attribute \"element type\" string \"tetrahedra\"\n");
  fprintf(fp_dx,"attribute \"ref\" string \"positions\"\n");
  printf("\n");
  
  /* Read object sizes */
  fscanf(fp_inp,"%d",&objects_n);
  if ( objects_n == 1 )
	 printf("Found %d object.\n",objects_n);
  else
	 printf("Found %d objects.\n",objects_n);
  for (i=0;i<(objects_n-1);i++){
	 fscanf(fp_inp,"%d",objects_size+i);
	 printf("  Object %d has %d components.\n",i+1,objects_size[i]);
  }
  fscanf(fp_inp,"%d\n",objects_size+i);
  printf("  Object %d has %d components.\n",i+1,objects_size[i]);
  printf("Warning: Skipping component information.\n");
  
  /* Skip a few lines */
  for (i=0;i<(objects_n);i++)
	 while ( getc(fp_inp) != '\n' );
  
  /* Values */  
  printf("Reading values......");
  fflush(stdout);
  fprintf(fp_dx,"\n# Values at nodal points\n");
  fprintf(fp_dx,"object 3 class array type float rank %d shape %d items %d data follows\n",
			 (nvars > 1 ? 1 : 0 ),nvars,npoints);
  for (i=0;i<npoints;i++){
	 
	 /* Read values */
	 fscanf(fp_inp,"%d",&dummy1);
	 for (j=0;j<(nvars-1);j++)
		fscanf(fp_inp,"%f ",data+j);
	 fscanf(fp_inp,"%f\n",data+j);
	 
	 if ( i % (npoints/20) == 0 ){
		printf(".");
		fflush(stdout);
	 }
	 
	 /* Save values */
	 for (j=0;j<(nvars-1);j++)
		fprintf(fp_dx,"%f ",data[j]);
	 fprintf(fp_dx,"%f\n",data[j]);
	 
  }
  fprintf(fp_dx,"attribute \"dep\" string \"positions\"\n");
  printf("\n");

  /* Write a few lines to make Open DX understand the data */
  fprintf(fp_dx,"\n# The Open DX field\n");
  fprintf(fp_dx,"object \"irregular positions irregular connections\" class field\n");
  fprintf(fp_dx,"component \"positions\" value 1\n");
  fprintf(fp_dx,"component \"connections\" value 2\n");
  fprintf(fp_dx,"component \"data\" value 3\n");
  fprintf(fp_dx,"end\n");

  /* Delete memory for data */
  free(data);
  free(objects_size);
  
  /* Close files */
  fclose(fp_inp);
  fclose(fp_dx);
  
  return 0;
}
