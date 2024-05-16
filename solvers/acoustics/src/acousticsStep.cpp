/*

The MIT License (MIT)

Copyright (c) 2017-2022 Tim Warburton, Noel Chalmers, Jesse Chan, Ali Karakus

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

#include "acoustics.hpp"
#include "timer.hpp"

dfloat acoustics_t::MaxWaveSpeed(){
  //wavespeed is constant 1 everywhere
  const dfloat vmax = 1.0;
  return vmax;
}

//evaluate ODE rhs = f(q,t)
void acoustics_t::rhsf(deviceMemory<dfloat>& o_Q, deviceMemory<dfloat>& o_RHS, const dfloat T){

#if 0
  printf("Ndofs = %d\n", mesh.Nelements*mesh.Np*Nfields);

  int wavesize = 1;
  if (platform.device.mode() == "CUDA") wavesize = 32;
  if (platform.device.mode() == "HIP") wavesize = 64;

  size_t shmemLimit = 64*1024 - 32; //64 KB
  if (platform.device.mode() == "CUDA") shmemLimit = 48*1024; //48 KB


  // OCCA build stuff
  properties_t kernelInfo = mesh.props; //copy base occa properties

  //add boundary data to kernel info
  std::string dataFileName;
  settings.getSetting("DATA FILE", dataFileName);
  kernelInfo["includes"] += dataFileName;

  kernelInfo["defines/" "p_Nfields"]= Nfields;

  const dfloat p_half = 1./2.;
  kernelInfo["defines/" "p_half"]= p_half;

  int maxNodes = std::max(mesh.Np, (mesh.Nfp*mesh.Nfaces));
  kernelInfo["defines/" "p_maxNodes"]= maxNodes;
  kernelInfo["defines/" "p_Lambda2"]= 0.5;

  // set kernel name suffix
  std::string suffix = mesh.elementSuffix();
  std::string oklFilePrefix = DACOUSTICS "/okl/";
  std::string oklFileSuffix = ".okl";

  std::string fileName, kernelName;

  // kernels from volume file
  fileName   = oklFilePrefix + "acousticsVolume" + suffix + oklFileSuffix;
  kernelName = "acousticsVolume" + suffix;

  kernelInfo["defines/KERNEL_NUMBER"] = 0;

  int maxElementsPerBlock = 1024/mesh.Np;
  int maxElementsPerThread = 10;

  double maxBW = 0;
  double maxGflops = 0;
  int bestElementsPerBlock = 0;
  int bestElementsPerThread = 0;

  for (int elementsPerBlock=1;elementsPerBlock<=maxElementsPerBlock;elementsPerBlock++) {
    properties_t props = kernelInfo;

    //Count number of waves
    int Nwaves = (mesh.Np*elementsPerBlock+wavesize-1)/wavesize;
    //increment elementsPerBlock to fill wave count
    if (maxElementsPerBlock>1)
      while ((mesh.Np*(elementsPerBlock+1)+wavesize-1)/wavesize == Nwaves) elementsPerBlock++;

    int blockSize = mesh.Np*elementsPerBlock;
    if (blockSize>1024) break;

    for (int elementsPerThread=1; elementsPerThread<=maxElementsPerThread;elementsPerThread++) {

      //Check shmem use
      size_t shmem = sizeof(dfloat)*elementsPerThread*elementsPerBlock*mesh.Np*Nfields;
      if (shmem > shmemLimit) continue;

      if (maxElementsPerBlock>1)  props["defines/p_NelementsPerBlk"] = elementsPerBlock;
      if (maxElementsPerThread>1) props["defines/p_NelementsPerThd"] = elementsPerThread;

      // Ax kernel
      volumeKernel =  platform.buildKernel(fileName, kernelName,
                                         props);

      for(int n=0;n<5;++n){ //warmup
        volumeKernel(mesh.Nelements,
                     mesh.o_vgeo,
                     mesh.o_D,
                     o_Q,
                     o_RHS);
      }

      int Ntrials = 10;
      timePoint_t start = GlobalPlatformTime(platform);
      for (int n=0;n<Ntrials;++n) {
        volumeKernel(mesh.Nelements,
                     mesh.o_vgeo,
                     mesh.o_D,
                     o_Q,
                     o_RHS);
      }
      timePoint_t end = GlobalPlatformTime(platform);
      double time = ElapsedTime(start, end)/Ntrials;

      size_t Nbytes = sizeof(dfloat) * mesh.Nelements * mesh.Np * Nfields * 2 +
                      sizeof(dfloat) * mesh.Nelements * 9;

      size_t Nflops = static_cast<size_t>(mesh.Nelements) * mesh.Np * mesh.Np * 2 * 6 + //3 derivatives per field
                      static_cast<size_t>(mesh.Nelements) * mesh.Np * 30;

      double BW = Nbytes/(1.0e9 * time);
      double gflops = Nflops/(1.0e9 * time);

      printf("Volume kernel, N=%d, Np=%d, blocksize=%d, elementsPerBlock = %d, elementsPerThread = %d: time = %f, BW = %f GB/s, GFLOPS = %f\n",
             mesh.N, mesh.Np, mesh.Np*elementsPerBlock, elementsPerBlock, elementsPerThread,
             time, static_cast<double>(Nbytes)/(1.e9*time), static_cast<double>(Nflops)/(1.e9*time));

      if (BW > maxBW) {
        maxBW = BW;
        maxGflops = gflops;
        bestElementsPerBlock = elementsPerBlock;
        bestElementsPerThread = elementsPerThread;
      }
    }
  }


  printf("Volume kernel: N=%2d, BW = %6.1f, GFLOPS = %6.1f", mesh.N, maxBW, maxGflops);
  if (maxElementsPerBlock>1)  printf(", bestElementsPerBlock = %d", bestElementsPerBlock);
  if (maxElementsPerThread>1) printf(", bestElementsPerThread = %d", bestElementsPerThread);
  printf("\n");

  kernelInfo["defines/KERNEL_NUMBER"] = 1;

  // Ax kernel
  volumeKernel =  platform.buildKernel(fileName, kernelName,
                                     kernelInfo);

  for(int n=0;n<5;++n){ //warmup
    volumeKernel(mesh.Nelements,
                 mesh.o_vgeo,
                 mesh.o_D,
                 o_Q,
                 o_RHS);
  }

  int Ntrials = 10;
  timePoint_t start = GlobalPlatformTime(platform);
  for (int n=0;n<Ntrials;++n) {
    volumeKernel(mesh.Nelements,
                 mesh.o_vgeo,
                 mesh.o_D,
                 o_Q,
                 o_RHS);
  }
  timePoint_t end = GlobalPlatformTime(platform);
  double time = ElapsedTime(start, end)/Ntrials;

  size_t Nbytes = sizeof(dfloat) * mesh.Nelements * mesh.Np * Nfields * 2 +
                  sizeof(dfloat) * mesh.Nelements * 9;

  size_t Nflops = static_cast<size_t>(mesh.Nelements) * mesh.Np * mesh.Np * 2 * 6 + //3 derivatives per field
                  static_cast<size_t>(mesh.Nelements) * mesh.Np * 30;

  printf("Volume MFMA kernel, N=%d, BW = %f GB/s, GFLOPS = %f\n",
         mesh.N, static_cast<double>(Nbytes)/(1.e9*time), static_cast<double>(Nflops)/(1.e9*time));


  // kernels from volume file
  fileName   = oklFilePrefix + "acousticsSurface" + suffix + oklFileSuffix;
  kernelName = "acousticsSurface" + suffix;

  kernelInfo["defines/KERNEL_NUMBER"] = 0;

  maxElementsPerBlock = 1024/maxNodes;
  maxElementsPerThread = 10;

  maxBW = 0;
  maxGflops = 0;
  bestElementsPerBlock = 0;
  bestElementsPerThread = 0;

  for (int elementsPerBlock=1;elementsPerBlock<=maxElementsPerBlock;elementsPerBlock++) {
    properties_t props = kernelInfo;

    //Count number of waves
    int Nwaves = (maxNodes*elementsPerBlock+wavesize-1)/wavesize;
    //increment elementsPerBlock to fill wave count
    if (maxElementsPerBlock>1)
      while ((maxNodes*(elementsPerBlock+1)+wavesize-1)/wavesize == Nwaves) elementsPerBlock++;

    int blockSize = maxNodes*elementsPerBlock;
    if (blockSize>1024) break;

    for (int elementsPerThread=1; elementsPerThread<=maxElementsPerThread;elementsPerThread++) {

      //Check shmem use
      size_t shmem = sizeof(dfloat)*elementsPerThread*elementsPerBlock*mesh.Nfaces*mesh.Nfp*Nfields;
      if (shmem > shmemLimit) continue;

      // if (maxElementsPerBlock>1)
        props["defines/p_NelementsPerBlk"] = elementsPerBlock;
      // if (maxElementsPerThread>1)
        props["defines/p_NelementsPerThd"] = elementsPerThread;

      // Ax kernel
      surfaceKernel =  platform.buildKernel(fileName, kernelName,  props);

      for(int n=0;n<5;++n){ //warmup
        surfaceKernel(mesh.NinternalElements,
                  mesh.o_internalElementIds,
                  mesh.o_sgeo,
                  mesh.o_LIFT,
                  mesh.o_vmapM,
                  mesh.o_vmapP,
                  mesh.o_EToB,
                  T,
                  mesh.o_x,
                  mesh.o_y,
                  mesh.o_z,
                  o_Q,
                  o_RHS);
      }

      int Ntrials = 10;
      timePoint_t start = GlobalPlatformTime(platform);
      for (int n=0;n<Ntrials;++n) {
        surfaceKernel(mesh.NinternalElements,
                  mesh.o_internalElementIds,
                  mesh.o_sgeo,
                  mesh.o_LIFT,
                  mesh.o_vmapM,
                  mesh.o_vmapP,
                  mesh.o_EToB,
                  T,
                  mesh.o_x,
                  mesh.o_y,
                  mesh.o_z,
                  o_Q,
                  o_RHS);
      }
      timePoint_t end = GlobalPlatformTime(platform);
      double time = ElapsedTime(start, end)/Ntrials;

      size_t Nbytes = (sizeof(dfloat) +2*sizeof(dlong)) * mesh.Nelements * mesh.Nfp * mesh.Nfaces * Nfields +
                      2*sizeof(dfloat) * mesh.Nelements * mesh.Np * Nfields +
                      sizeof(dfloat) * mesh.Nelements * mesh.Nfaces * 5 + sizeof(dlong) * mesh.Nelements * mesh.Nfaces;

      size_t Nflops = static_cast<size_t>(mesh.Nelements) * mesh.Np * mesh.Nfaces * mesh.Nfp * 2 *Nfields;

      double BW = Nbytes/(1.0e9 * time);
      double gflops = Nflops/(1.0e9 * time);

      printf("Surface kernel, N=%d, Np=%d, blocksize=%d, elementsPerBlock = %d, elementsPerThread = %d: time = %f, BW = %f GB/s, GFLOPS = %f\n",
             mesh.N, mesh.Np, maxNodes*elementsPerBlock, elementsPerBlock, elementsPerThread,
             time, static_cast<double>(Nbytes)/(1.e9*time), static_cast<double>(Nflops)/(1.e9*time));

      if (BW > maxBW) {
        maxBW = BW;
        maxGflops = gflops;
        bestElementsPerBlock = elementsPerBlock;
        bestElementsPerThread = elementsPerThread;
      }
    }
  }


  printf("Surface kernel: N=%2d, BW = %6.1f, GFLOPS = %6.1f", mesh.N, maxBW, maxGflops);
  if (maxElementsPerBlock>1)  printf(", bestElementsPerBlock = %d", bestElementsPerBlock);
  if (maxElementsPerThread>1) printf(", bestElementsPerThread = %d", bestElementsPerThread);
  printf("\n");

  kernelInfo["defines/KERNEL_NUMBER"] = 1;

  // Ax kernel
  surfaceKernel =  platform.buildKernel(fileName, kernelName,  kernelInfo);

  for(int n=0;n<5;++n){ //warmup
    surfaceKernel(mesh.NinternalElements,
                  mesh.o_internalElementIds,
                  mesh.o_sgeo,
                  mesh.o_LIFT,
                  mesh.o_vmapM,
                  mesh.o_vmapP,
                  mesh.o_EToB,
                  T,
                  mesh.o_x,
                  mesh.o_y,
                  mesh.o_z,
                  o_Q,
                  o_RHS);
  }

  Ntrials = 10;
  start = GlobalPlatformTime(platform);
  for (int n=0;n<Ntrials;++n) {
    surfaceKernel(mesh.NinternalElements,
                  mesh.o_internalElementIds,
                  mesh.o_sgeo,
                  mesh.o_LIFT,
                  mesh.o_vmapM,
                  mesh.o_vmapP,
                  mesh.o_EToB,
                  T,
                  mesh.o_x,
                  mesh.o_y,
                  mesh.o_z,
                  o_Q,
                  o_RHS);
  }
  end = GlobalPlatformTime(platform);
  time = ElapsedTime(start, end)/Ntrials;

  Nbytes = (sizeof(dfloat) +2*sizeof(dlong)) * mesh.Nelements * mesh.Nfp * mesh.Nfaces * Nfields +
                      2*sizeof(dfloat) * mesh.Nelements * mesh.Np * Nfields +
                      sizeof(dfloat) * mesh.Nelements * mesh.Nfaces * 5 + sizeof(dlong) * mesh.Nelements * mesh.Nfaces;

  Nflops = static_cast<size_t>(mesh.Nelements) * mesh.Np * mesh.Nfaces * mesh.Nfp * 2 *Nfields;


  printf("Surface MFMA kernel, N=%d, BW = %f GB/s, GFLOPS = %f\n",
         mesh.N, static_cast<double>(Nbytes)/(1.e9*time), static_cast<double>(Nflops)/(1.e9*time));

  exit(-1);
#endif

  // extract q halo on DEVICE
  traceHalo.ExchangeStart(o_Q, 1);

  volumeKernel(mesh.Nelements,
               mesh.o_vgeo,
               mesh.o_D,
               o_Q,
               o_RHS);

  if (mesh.NinternalElements)
    surfaceKernel(mesh.NinternalElements,
                  mesh.o_internalElementIds,
                  mesh.o_sgeo,
                  mesh.o_LIFT,
                  mesh.o_vmapM,
                  mesh.o_vmapP,
                  mesh.o_EToB,
                  T,
                  mesh.o_x,
                  mesh.o_y,
                  mesh.o_z,
                  o_Q,
                  o_RHS);

  traceHalo.ExchangeFinish(o_Q, 1);

  if (mesh.NhaloElements)
    surfaceKernel(mesh.NhaloElements,
                  mesh.o_haloElementIds,
                  mesh.o_sgeo,
                  mesh.o_LIFT,
                  mesh.o_vmapM,
                  mesh.o_vmapP,
                  mesh.o_EToB,
                  T,
                  mesh.o_x,
                  mesh.o_y,
                  mesh.o_z,
                  o_Q,
                  o_RHS);
}
