#include "ins2D.h"

// complete a time step using LSERK4
void insHelmholtzStepSS2D(ins_t *ins, iint tstep,  iint haloBytes,
			dfloat * sendBuffer, dfloat * recvBuffer, 
			char   * options){
  
  mesh2D *mesh = ins->mesh; 
  solver_t *solver = ins->vSolver; 

  dfloat t = tstep*ins->dt + ins->dt;
  
  iint Ntotal = mesh->Nelements+mesh->totalHaloPairs;
  iint rhsPackingMode = (strstr(options, "VECTORHELMHOLTZ")) ? 1:0;
  
  // if(strstr(options,"BROKEN")){

  if(strstr(options,"SUBCYCLING")){
     // compute all forcing i.e. f^(n+1) - grad(Pr)
    ins->helmholtzRhsForcingKernel(mesh->Nelements,
                                   ins->dt,
                                   ins->g0,
                                   mesh->o_vgeo,
                                   mesh->o_MM,
                                   ins->o_Ut,
                                   ins->o_Vt,
                                   ins->o_rhsU,
                                   ins->o_rhsV);
  }
  else{
    ins->helmholtzRhsForcingKernel(mesh->Nelements,
                                   ins->dt,
                                   ins->g0,
                                   mesh->o_vgeo,
                                   mesh->o_MM,
                                   ins->o_Ut,
                                   ins->o_Vt,
                                   ins->o_rhsU,
                                   ins->o_rhsV);
  }

// } else{


// // *(pxB) = 2.f*OCCA_PI*occaCos(2.f*OCCA_PI*y)*occaSin(2.f*OCCA_PI*x)*occaExp(-p_nu*8.f*OCCA_PI*OCCA_PI*t); \
// // *(pyB) = 2.f*OCCA_PI*occaSin(2.f*OCCA_PI*y)*occaCos(2.f*OCCA_PI*x)*occaExp(-p_nu*8.f*OCCA_PI*OCCA_PI*t); \



// }
  
  ins->helmholtzRhsIpdgBCKernel(mesh->Nelements,
				                        rhsPackingMode,
                                mesh->o_vmapM,
                                mesh->o_vmapP,
                                ins->tau,
                                t,
                                mesh->o_x,
                                mesh->o_y,
                                mesh->o_vgeo,
                                mesh->o_sgeo,
                                mesh->o_EToB,
                                mesh->o_DrT,
                                mesh->o_DsT,
                                mesh->o_LIFTT,
                                mesh->o_MM,
                                ins->o_rhsU,
                                ins->o_rhsV);

  
   
    ins->o_Ut.copyFrom(ins->o_U,Ntotal*mesh->Np*sizeof(dfloat),0,ins->index*Ntotal*mesh->Np*sizeof(dfloat));
    ins->o_Vt.copyFrom(ins->o_V,Ntotal*mesh->Np*sizeof(dfloat),0,ins->index*Ntotal*mesh->Np*sizeof(dfloat));

    int Niter = 0; 
    printf("Solving for Ux: Niter= ");
    Niter = ellipticSolveTri2D( solver, ins->lambda, ins->velTOL, ins->o_rhsU, ins->o_Ut, ins->vSolverOptions);
    printf("%d \n",Niter);

    printf("Solving for Uy: Niter= ");
    Niter = ellipticSolveTri2D(solver, ins->lambda, ins->velTOL, ins->o_rhsV, ins->o_Vt, ins->vSolverOptions);
    printf("%d \n",Niter);
   
    //copy into next stage's storage
    ins->index = (ins->index+1)%3; //hard coded for 3 stages
    //
    ins->o_Ut.copyTo(ins->o_U,Ntotal*mesh->Np*sizeof(dfloat),ins->index*Ntotal*mesh->Np*sizeof(dfloat),0);
    ins->o_Vt.copyTo(ins->o_V,Ntotal*mesh->Np*sizeof(dfloat),ins->index*Ntotal*mesh->Np*sizeof(dfloat),0);  

}
