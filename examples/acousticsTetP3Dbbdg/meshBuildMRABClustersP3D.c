
#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#include "mesh3D.h"

typedef struct {
  iint id;
  iint level;
  dfloat weight;
  iint N;

  // 8 for maximum number of vertices per element in 3D
  iint v[8];
  dfloat EX[8], EY[8], EZ[8];

  iint cRank;
  iint cId;
} cElement_t;

typedef struct {
  iint Nelements;
  iint offSet;
} cluster_t;

int compareCluster(const void *a, const void *b) {
  cElement_t *na = (cElement_t *) a;
  cElement_t *nb = (cElement_t *) b;

  if (na->cRank < nb->cRank) return -1;
  if (nb->cRank < na->cRank) return +1;

  if (na->cId < nb->cId) return -1;
  if (nb->cId < na->cId) return +1;  

  return 0;
}


void meshBuildMRABClustersP3D(mesh_t *mesh, iint lev, dfloat *weights, iint *levels,
            iint *Nclusters, cluster_t **clusters, iint *Nelements, cElement_t **elements) {

  iint rank, size;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // minimum {vertex id % size}
  iint *Nsend = (iint*) calloc(size, sizeof(iint));
  iint *Nrecv = (iint*) calloc(size, sizeof(iint));
  iint *Ncount = (iint*) calloc(size, sizeof(iint));
  iint *sendOffsets = (iint*) calloc(size, sizeof(iint));
  iint *recvOffsets = (iint*) calloc(size, sizeof(iint));
  iint *sendCounts = (iint*) calloc(size, sizeof(iint));


  //build element struct
  *elements = (cElement_t *) calloc(mesh->Nelements+mesh->totalHaloPairs,sizeof(cElement_t));
  for (iint e=0;e<mesh->Nelements;e++) {
    (*elements)[e].id = e;
    (*elements)[e].level = 0.;
    if (levels) (*elements)[e].level = levels[e];
    (*elements)[e].N = mesh->N[e];
    
    (*elements)[e].weight = 1.;
    if (weights) (*elements)[e].weight = weights[e];

    for(iint n=0;n<mesh->Nverts;++n){
      (*elements)[e].v[n] = mesh->EToV[e*mesh->Nverts+n];
      (*elements)[e].EX[n] = mesh->EX[e*mesh->Nverts+n];
      (*elements)[e].EY[n] = mesh->EY[e*mesh->Nverts+n];
      (*elements)[e].EZ[n] = mesh->EZ[e*mesh->Nverts+n];
    }

    //initialize the clustering numbering
    (*elements)[e].cId = e; 
    (*elements)[e].cRank = rank;
  }

  cElement_t *sendBuffer = (cElement_t *) calloc(mesh->totalHaloPairs,sizeof(cElement_t));

  //propagate clusters 
  int allDone = 0;
  int rankDone, done;
  while(!allDone) {
    meshHaloExchange(mesh, sizeof(cElement_t), *elements, sendBuffer, *elements + mesh->Nelements);

    rankDone = 1;
    //local clustering
    done = 0;
    while(!done) {
      done = 1;
      for (iint e=0;e<mesh->Nelements;e++) {
        for (iint f=0;f<mesh->Nfaces;f++) {
          iint eP = mesh->EToE[e*mesh->Nfaces +f];
          if (eP>-1) {
            if (((*elements)[eP].level<lev+1)||((*elements)[e].level<lev+1)){
              if (compareCluster(*elements+eP,*elements+e)<0) {
                (*elements)[e].cRank = (*elements)[eP].cRank;
                (*elements)[e].cId   = (*elements)[eP].cId;
                done = 0;
                rankDone = 0;
              }
            }
          }
        }
      }    
    }

    MPI_Allreduce(&rankDone, &allDone, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    allDone /= size;
  }

  //clusters have been built
  //transfer them to their owning rank

  qsort((*elements), mesh->Nelements, sizeof(cElement_t), compareCluster);

  //set up exchange along MPI interfaces
  for (iint r=0;r<size;r++)
    Nsend[r] = 0;

  for(iint e=0;e<mesh->Nelements;++e)
    ++Nsend[(*elements)[e].cRank];

  // find send offsets
  sendOffsets[0] = 0;
  for(iint r=1;r<size;++r)
    sendOffsets[r] = sendOffsets[r-1] + Nsend[r-1];
  
  // exchange byte counts 
  MPI_Alltoall(Nsend, 1, MPI_IINT,
         Nrecv, 1, MPI_IINT,
         MPI_COMM_WORLD);
  
  // count incoming faces
  iint allNrecv = 0;
  for(iint r=0;r<size;++r){
    allNrecv += Nrecv[r];
    Nrecv[r] *= sizeof(cElement_t);
    Nsend[r] *= sizeof(cElement_t);
    sendOffsets[r] *= sizeof(cElement_t);
  }
  for(iint r=1;r<size;++r)
    recvOffsets[r] = recvOffsets[r-1] + Nrecv[r-1];

  // buffer for recvied elements
  cElement_t *recvElements = (cElement_t*) calloc(allNrecv, sizeof(cElement_t));  

  // exchange parallel faces
  MPI_Alltoallv(*elements, Nsend, sendOffsets, MPI_CHAR,
      recvElements, Nrecv, recvOffsets, MPI_CHAR,
      MPI_COMM_WORLD);

  free(*elements);
  *elements = recvElements;
  *Nelements = allNrecv;

  qsort((*elements), *Nelements, sizeof(cElement_t), compareCluster);

  //build cluster lists
  // the lists are already sorted by cluster, so we just scan for different indices
  *Nclusters = 0;
  if (*Nelements) {
    (*Nclusters)++;
    for (iint e=1;e<*Nelements;e++) {
      if ((*elements)[e].cId != (*elements)[e-1].cId) (*Nclusters)++;
    }

    *clusters = (cluster_t *) calloc(*Nclusters,sizeof(cluster_t));

    iint cnt  = 0;
    iint ecnt = 1;
    (*clusters)[0].Nelements = 1;
    (*clusters)[0].offSet = 0;
    for (iint e=1;e<*Nelements;e++) {
      if ((*elements)[e].cId != (*elements)[e-1].cId) {
        cnt++;
        (*clusters)[cnt].offSet = e;
        (*clusters)[cnt].Nelements = 1;
      } else {
        (*clusters)[cnt].Nelements++;
      }
    }
  }
}