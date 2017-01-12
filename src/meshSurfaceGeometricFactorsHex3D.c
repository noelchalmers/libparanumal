#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "mesh3D.h"

/* compute outwards facing normals, surface Jacobian, and volume Jacobian for all face nodes */
void meshSurfaceGeometricFactorsHex3D(mesh3D *mesh){

  /* unified storage array for geometric factors */
  mesh->Nsgeo = 5;
  mesh->sgeo = (dfloat*) calloc(mesh->Nelements*mesh->Nsgeo*mesh->Nfp*mesh->Nfaces, 
				sizeof(dfloat));
  
  for(iint e=0;e<mesh->Nelements;++e){ /* for each element */

    /* find vertex indices and physical coordinates */
    iint id = e*mesh->Nverts;

    dfloat *xe = mesh->EX + id;
    dfloat *ye = mesh->EY + id;
    dfloat *ze = mesh->EZ + id;
    
    for(iint f=0;f<mesh->Nfaces;++f){ // for each face
      
      for(iint i=0;i<mesh->Nfp;++i){  // for each node on face

	/* volume index of face node */
	iint n = mesh->faceNodes[f*mesh->Nfp+i];

	/* local node coordinates */
	dfloat rn = mesh->r[n]; 
	dfloat sn = mesh->s[n];
	dfloat tn = mesh->t[n];
	
	/* Jacobian matrix */
	dfloat J = xr*(ys*zt-zs*yt) - yr*(xs*zt-zs*xt) + zr*(xs*yt-ys*xt);
	
	dfloat xr = 0.25*( (1-tn)*(1-sn)*(xe[1]-xe[0]) + (1-tn)*(1+sn)*(xe[2]-xe[3]) + (1+tn)*(1-sn)*(xe[5]-xe[4]) + (1+tn)*(1+sn)*(xe[6]-xe[7]) );
	dfloat xs = 0.25*( (1-tn)*(1-rn)*(xe[3]-xe[0]) + (1-tn)*(1+rn)*(xe[2]-xe[1]) + (1+tn)*(1-rn)*(xe[7]-xe[4]) + (1+tn)*(1+rn)*(xe[6]-xe[5]) );
	dfloat xt = 0.25*( (1-rn)*(1-sn)*(xe[4]-xe[0]) + (1+rn)*(1-sn)*(xe[5]-xe[1]) + (1+rn)*(1+sn)*(xe[6]-xe[2]) + (1-rn)*(1+sn)*(xe[7]-xe[3]) );
	
	dfloat yr = 0.25*( (1-tn)*(1-sn)*(ye[1]-ye[0]) + (1-tn)*(1+sn)*(ye[2]-ye[3]) + (1+tn)*(1-sn)*(ye[5]-ye[4]) + (1+tn)*(1+sn)*(ye[6]-ye[7]) );
	dfloat ys = 0.25*( (1-tn)*(1-rn)*(ye[3]-ye[0]) + (1-tn)*(1+rn)*(ye[2]-ye[1]) + (1+tn)*(1-rn)*(ye[7]-ye[4]) + (1+tn)*(1+rn)*(ye[6]-ye[5]) );
	dfloat yt = 0.25*( (1-rn)*(1-sn)*(ye[4]-ye[0]) + (1+rn)*(1-sn)*(ye[5]-ye[1]) + (1+rn)*(1+sn)*(ye[6]-ye[2]) + (1-rn)*(1+sn)*(ye[7]-ye[3]) );
	
	dfloat zr = 0.25*( (1-tn)*(1-sn)*(ze[1]-ze[0]) + (1-tn)*(1+sn)*(ze[2]-ze[3]) + (1+tn)*(1-sn)*(ze[5]-ze[4]) + (1+tn)*(1+sn)*(ze[6]-ze[7]) );
	dfloat zs = 0.25*( (1-tn)*(1-rn)*(ze[3]-ze[0]) + (1-tn)*(1+rn)*(ze[2]-ze[1]) + (1+tn)*(1-rn)*(ze[7]-ze[4]) + (1+tn)*(1+rn)*(ze[6]-ze[5]) );
	dfloat zt = 0.25*( (1-rn)*(1-sn)*(ze[4]-ze[0]) + (1+rn)*(1-sn)*(ze[5]-ze[1]) + (1+rn)*(1+sn)*(ze[6]-ze[2]) + (1-rn)*(1+sn)*(ze[7]-ze[3]) );

	dfloat rx =  (ys*zt - zs*yt)/J, ry = -(xs*zt - zs*xt)/J, rz =  (xs*yt - ys*xt)/J;
	dfloat sx = -(yr*zt - zr*yt)/J, sy =  (xr*zt - zr*xt)/J, sz = -(xr*yt - yr*xt)/J;
	dfloat tx =  (yr*zs - zr*ys)/J, ty = -(xr*zs - zr*xs)/J, tz =  (xr*ys - yr*xs)/J;
	
	/* face f normal and length */
	dfloat nx, ny, nz, d;
	switch(f){
	case 0: nx = -tx; ny = -ty; nz = -tz; break;
	case 1: nx = -sx; ny = -sy; nz = -sz; break;
	case 2: nx = +rx; ny = +ry; nz = +rz; break;
	case 3: nx = +sx; ny = +sy; nz = +sz; break;
	case 4: nx = -rx; ny = -ry; nz = -rz; break;
	case 5: nx = +tx; ny = +ty; nz = +tz; break;
	}

	dfloat sJ = sqrt(nx*nx+ny*ny+nz*nz);
	nx /= sJ; ny /= sJ; nz /= sJ;
	sJ *= J;
	
	/* output index */
	iint base = mesh->Nsgeo*(mesh->Nfaces*mesh->Nfp*e + mesh->Nfp*f + i);

	/* store normal, surface Jacobian, and reciprocal of volume Jacobian */
	mesh->sgeo[base+NXID] = nx/d;
	mesh->sgeo[base+NYID] = ny/d;
	mesh->sgeo[base+NZID] = nz/d;
	mesh->sgeo[base+SJID] = d/2.;
	mesh->sgeo[base+IJID] = 1./J;
      }
    }
  }
}
