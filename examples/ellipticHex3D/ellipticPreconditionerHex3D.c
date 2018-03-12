#include "ellipticHex3D.h"

void ellipticPreconditioner3D(solver_t *solver,
            dfloat lambda,
            occa::memory &o_r,
            occa::memory &o_z,
            const char *options){

  mesh_t *mesh = solver->mesh;
  precon_t *precon = solver->precon;

  if (strstr(options, "FULLALMOND")||strstr(options, "MULTIGRID")||strstr(options, "SEMFEM")) {

    occaTimerTic(mesh->device,"parALMOND");
    parAlmondPrecon(precon->parAlmond, o_z, o_r);
    occaTimerToc(mesh->device,"parALMOND");

  } else if(strstr(options, "OAS")){
    
    //patch solve
    //ellipticPatchSmootherHex3D(solver,o_r,o_z,options);
    smoothHex3D(precon->OASsmoothArgs, o_r, o_z,true);

    occaTimerTic(mesh->device,"coarseGrid");

    // Z1*Z1'*PL1*(Z1*z1) = (Z1*rL)  HMMM
    occaTimerTic(mesh->device,"coarsenKernel");
    precon->coarsenKernel(mesh->Nelements, precon->o_V1, o_r, precon->o_r1);
    occaTimerToc(mesh->device,"coarsenKernel");

    occaTimerTic(mesh->device,"ALMOND");
    parAlmondPrecon(precon->parAlmond, precon->o_z1, precon->o_r1);
    occaTimerToc(mesh->device,"ALMOND");

    // prolongate from P1 to PN, adding patch and coarse solves together
    occaTimerTic(mesh->device,"prolongateKernel");
    precon->prolongateKernel(mesh->Nelements, precon->o_V1, precon->o_z1, solver->o_z);
    occaTimerToc(mesh->device,"prolongateKernel");

    occaTimerToc(mesh->device,"coarseGrid");

  } else if(strstr(options, "JACOBI")){

    dlong Ntotal = mesh->Np*mesh->Nelements;
    // Jacobi preconditioner
    occaTimerTic(mesh->device,"dotDivideKernel");
    solver->dotMultiplyKernel(Ntotal, o_r, precon->o_invDiagA, o_z);
    occaTimerToc(mesh->device,"dotDivideKernel");
  
  }  else{ // turn off preconditioner
    o_z.copyFrom(o_r); 
  }
}
