#include <stdio.h>
#include <stdlib.h>
#include "mesh3D.h"

void meshGeometricFactors3D(mesh3D *mesh){

  /* unified storage array for geometric factors */
  mesh->Nvgeo = 10;
  mesh->vgeo = (dfloat*) calloc(mesh->Nelements*mesh->Nvgeo, 
				sizeof(dfloat));
  dfloat minJ = 1e9, maxJ = -1e9;
  for(iint e=0;e<mesh->Nelements;++e){ /* for each element */

    /* find vertex indices and physical coordinates */
    iint id = e*mesh->Nverts;

    /* vertex coordinates */
    dfloat xe1 = mesh->EX[id+0], ye1 = mesh->EY[id+0], ze1 = mesh->EZ[id+0];
    dfloat xe2 = mesh->EX[id+1], ye2 = mesh->EY[id+1], ze2 = mesh->EZ[id+1];
    dfloat xe3 = mesh->EX[id+2], ye3 = mesh->EY[id+2], ze3 = mesh->EZ[id+2];
    dfloat xe4 = mesh->EX[id+3], ye4 = mesh->EY[id+3], ze4 = mesh->EZ[id+3];

    /* Jacobian matrix */
    dfloat xr = 0.5*(xe2-xe1), xs = 0.5*(xe3-xe1), xt = 0.5*(xe4-xe1);
    dfloat yr = 0.5*(ye2-ye1), ys = 0.5*(ye3-ye1), yt = 0.5*(ye4-ye1);
    dfloat zr = 0.5*(ze2-ze1), zs = 0.5*(ze3-ze1), zt = 0.5*(ze4-ze1);

    /* compute geometric factors for affine coordinate transform*/
    dfloat J = xr*(ys*zt-zs*yt) - yr*(xs*zt-zs*xt) + zr*(xs*yt-ys*xt);
    
    dfloat rx =  (ys*zt - zs*yt)/J, ry = -(xs*zt - zs*xt)/J, rz =  (xs*yt - ys*xt)/J;
    dfloat sx = -(yr*zt - zr*yt)/J, sy =  (xr*zt - zr*xt)/J, sz = -(xr*yt - yr*xt)/J;
    dfloat tx =  (yr*zs - zr*ys)/J, ty = -(xr*zs - zr*xs)/J, tz =  (xr*ys - yr*xs)/J;

    if(J<0) printf("bugger: got negative geofac\n");
    minJ = mymin(minJ,J);
    maxJ = mymax(maxJ,J);
    
    /* store geometric factors */
    mesh->vgeo[mesh->Nvgeo*e + RXID] = rx;
    mesh->vgeo[mesh->Nvgeo*e + RYID] = ry;
    mesh->vgeo[mesh->Nvgeo*e + RZID] = rz;
    mesh->vgeo[mesh->Nvgeo*e + SXID] = sx;
    mesh->vgeo[mesh->Nvgeo*e + SYID] = sy;
    mesh->vgeo[mesh->Nvgeo*e + SZID] = sz;
    mesh->vgeo[mesh->Nvgeo*e + TXID] = tx;
    mesh->vgeo[mesh->Nvgeo*e + TYID] = ty;
    mesh->vgeo[mesh->Nvgeo*e + TZID] = tz;
    mesh->vgeo[mesh->Nvgeo*e +  JID] = J;
    //    printf("geo: %g,%g,%g - %g,%g,%g - %g,%g,%g\n",
    //	   rx,ry,rz, sx,sy,sz, tx,ty,tz);
  }

  printf("minJ = %g, maxJ = %g\n", minJ, maxJ);
}