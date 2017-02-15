
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>

#define UNDERSCORE 1
#define USE_NAIVE_BLAS 
#define NO_NEX_EXITT 1
#define GLOBAL_LONG_LONG 1
#define PREFIX jl_

#define MPI 1

#include "gslib.h"

typedef struct {
  struct crs_data *A;

  uint numLocalRows;
  uint nnz;

  ulong *rowIds;
  uint   *Ai; // row coordinates of non-zeros (local indexes)
  uint   *Aj; // column coordinates  of non-zeros (local indexes)
  double *Avals; // values of non-zeros

  double *x;
  double *rhs;

  char* iintType;
  char* dfloatType;

} crs_t;

void * xxtSetup(uint  numLocalRows, 
                void* rowIds,
                uint  nnz, 
                void* Ai,
                void* Aj,
                void* Avals,
                int   nullSpace,
                char* iintType, 
                char* dfloatType) {

  int np, myId, n;
  struct comm com;
  crs_t *crsA = (crs_t*) calloc(1, sizeof(crs_t));

  MPI_Comm_size(MPI_COMM_WORLD,&np);
  comm_init(&com,(comm_ext) MPI_COMM_WORLD);

  myId = com.id;

  crsA->numLocalRows = numLocalRows;
  crsA->nnz = nnz;

  if (strcmp(dfloatType,"float")) { //float
    crsA->Avals = (double *) malloc(nnz*sizeof(double));
    crsA->x     = (double *) malloc(numLocalRows*sizeof(double));
    crsA->rhs   = (double *) malloc(numLocalRows*sizeof(double));
    float *AvalsFloat = (float *) Avals;
    for (n=0;n<nnz;n++) crsA->Avals[n] = (double) AvalsFloat[n];
  } else { //double
    crsA->Avals = (double*) Avals;
  }

  if (strcmp(iintType,"int")) { //int
    crsA->rowIds = (ulong*) malloc(numLocalRows*sizeof(ulong));
    int *rowIdsInt = (int*) rowIds;
    for (n=0;n<numLocalRows;n++) crsA->rowIds[n] = (ulong) rowIdsInt[n];
  } else { //long
    crsA->rowIds = (ulong *) rowIds;
  }

  crsA->Ai = (uint*) Ai; //this will break if iint = long
  crsA->Aj = (uint*) Aj;
  
  crsA->iintType = iintType;
  crsA->dfloatType = dfloatType;

  crsA->A = crs_setup(crsA->numLocalRows,
		      crsA->rowIds,
		      crsA->nnz,
		      crsA->Ai,
		      crsA->Aj,
		      crsA->Avals,
		      nullSpace,
		      &com);

  crs_stats(crsA->A);

  return (void *) crsA;
}

int xxtSolve(void* x,
             void* A,
             void* rhs) {

  int n;
  
  crs_t *crsA = (crs_t *) A;

  if (strcmp(crsA->dfloatType,"float")) {
    float *xFloat   = (float *) x;
    float *rhsFloat = (float *) rhs;
    for (n=0;n<crsA->numLocalRows;n++) {
      crsA->x[n]   = (double) xFloat[n];
      crsA->rhs[n] = (double) rhsFloat[n];
    }
  } else {
    crsA->x   = (double*) x;
    crsA->rhs = (double*) rhs;
  }

  crs_solve(crsA->x,crsA->A,crsA->rhs);

  return 0;
}

int xxtFree(void* A) {
  crs_t *crsA = (crs_t *) A;

  crs_free(crsA->A);

  if (strcmp(crsA->dfloatType,"float")) { 
    free(crsA->Avals);
    free(crsA->x);  
    free(crsA->rhs);
  }

  if (strcmp(crsA->iintType,"int")) { 
    free(crsA->rowIds);
  }

  return 0;
}