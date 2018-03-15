#include "acousticsQuad2D.h"

void acousticsUpdate2D(mesh2D *mesh, dfloat rka, dfloat rkb){
  
  // Low storage Runge Kutta time step update
  for(int n=0;n<mesh->Nelements*mesh->Np*mesh->Nfields;++n){

    mesh->resq[n] = rka*mesh->resq[n] + mesh->dt*mesh->rhsq[n];
    
    mesh->q[n] += rkb*mesh->resq[n];
  }
}

