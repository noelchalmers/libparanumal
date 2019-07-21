/*

The MIT License (MIT)

Copyright (c) 2017 Tim Warburton, Noel Chalmers, Jesse Chan, Ali Karakus

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*/

#include "mesh.hpp"
#include "mesh2D.hpp"
#include "mesh3D.hpp"

void meshTri3D::BuildBasisCoarsen(dfloat**R, occa::memory& o_R, int Nf, int Nc){
  mesh_t *mesh_p = (mesh_t*) this;
  meshTri2D* trimesh = (meshTri2D*) mesh_p;
  trimesh->meshTri2D::BuildBasisCoarsen(R, o_R, Nf, Nc);
}

void meshTri2D::BuildBasisCoarsen(dfloat**R, occa::memory& o_R, int Nf, int Nc) {

  int NpFine   = ((Nf+1)*(Nf+2))/2;
  int NpCoarse = ((Nc+1)*(Nc+2))/2;

  dfloat *P    = (dfloat *) calloc(NpFine*NpCoarse,sizeof(dfloat));
  dfloat *Ptmp = (dfloat *) calloc(NpFine*NpCoarse,sizeof(dfloat));

  //initialize P as identity (which it is for SPARSE)
  for (int i=0;i<NpCoarse;i++) P[i*NpCoarse+i] = 1.0;

  for (int n=Nc;n<Nf;n++) {
    int Npp1 = ((n+2)*(n+3))/2;;
    int Npp   = ((n+1)*(n+2))/2;

    //copy P
    for (int i=0;i<Npp*NpCoarse;i++) Ptmp[i] = P[i];

    //get the raise op from the node file
    char fname[BUFSIZ];
    sprintf(fname, LIBP_DIR "/nodes/triangleN%02d.dat", n);

    FILE *fp = fopen(fname, "r");

    if (!fp) {
      stringstream ss;
      ss << "Cannot open file: " << fname;
      LIBP_ABORT(ss.str())
    }

    int Nrows, Ncols;
    dfloat *InterpRaise;
    readDfloatArray(comm, fp, "Nodal degree raise matrix", &(InterpRaise), &Nrows, &Ncols);

    //Multiply Ptmp by the raise op
    for (int i=0;i<Npp1;i++) {
      for (int j=0;j<NpCoarse;j++) {
        P[i*NpCoarse + j] = 0.;
        for (int k=0;k<Npp;k++) {
          P[i*NpCoarse + j] += InterpRaise[i*Npp+k]*Ptmp[k*NpCoarse + j];
        }
      }
    }

    fclose(fp);
    free(InterpRaise);
  }

  //the coarsen matrix is P^T
  *R = (dfloat *) calloc(NpFine*NpCoarse,sizeof(dfloat));
  for (int i=0;i<NpCoarse;i++) {
    for (int j=0;j<NpFine;j++) {
      (*R)[i*NpFine+j] = P[j*NpCoarse+i];
    }
  }
  o_R = device.malloc(NpFine*NpCoarse*sizeof(dfloat), *R);


  free(P); free(Ptmp);
}