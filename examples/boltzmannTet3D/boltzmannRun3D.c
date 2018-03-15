#include "boltzmann3D.h"

void boltzmannRun3D(mesh3D *mesh, char *options){

  // Allocate MPI send buffer
  int haloBytes = mesh->totalHaloPairs*mesh->Np*mesh->Nfields*sizeof(dfloat);
  dfloat *sendBuffer = (dfloat*) malloc(haloBytes);
  dfloat *recvBuffer = (dfloat*) malloc(haloBytes);

  occa::initTimer(mesh->device);
    for(int tstep=0;tstep<mesh->NtimeSteps;++tstep){

      if(strstr(options, "LSERK")){
      boltzmannLserkStep3D(mesh, tstep, haloBytes, sendBuffer, recvBuffer, options);
      }

      if(strstr(options, "LSIMEX")){
      boltzmannLsimexStep3D(mesh, tstep, haloBytes, sendBuffer, recvBuffer,options);
      }
      
      if(strstr(options, "SARK3")){
       boltzmannSark3Step3D(mesh, tstep, haloBytes, sendBuffer, recvBuffer,options);
      }

      if(strstr(options, "SAAB3")){
       boltzmannSaab3Step3D(mesh, tstep, haloBytes, sendBuffer, recvBuffer,options);
      }

     if(strstr(options, "REPORT")){
      if((tstep%mesh->errorStep)==0){
        boltzmannReport3D(mesh, tstep,options);
      }
     }
    }
  
  //For Final Time
  boltzmannReport3D(mesh, mesh->NtimeSteps,options);

  occa::printTimer();

  // Deallocate Halo MPI storage
  free(recvBuffer);
  free(sendBuffer);
}



