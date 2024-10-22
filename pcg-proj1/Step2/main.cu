/**
 * @file      main.cu
 *
 * @author    Tomáš Matuš \n
 *            Faculty of Information Technology \n
 *            Brno University of Technology \n
 *            xmatus37@fit.vutbr.cz
 *
 * @brief     PCG Assignment 1
 *
 * @version   2024
 *
 * @date      04 October   2023, 09:00 (created) \n
 */

#include <cmath>
#include <cstdio>
#include <chrono>
#include <string>

#include "nbody.cuh"
#include "h5Helper.h"

/**
 * @brief CUDA error checking macro
 * @param call CUDA API call
 */
#define CUDA_CALL(call) \
  do { \
    const cudaError_t _error = (call); \
    if (_error != cudaSuccess) \
    { \
      std::fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__, cudaGetErrorString(_error)); \
      std::exit(EXIT_FAILURE); \
    } \
  } while(0)

/**
 * Main rotine
 * @param argc
 * @param argv
 * @return
 */
int main(int argc, char **argv)
{
  if (argc != 10)
  {
    std::printf("Usage: nbody <N> <dt> <steps> <threads/block> <write intesity> <reduction threads> <reduction threads/block> <input> <output>\n");
    std::exit(1);
  }

  // Number of particles
  const unsigned N                   = static_cast<unsigned>(std::stoul(argv[1]));
  // Length of time step
  const float    dt                  = std::stof(argv[2]);
  // Number of steps
  const unsigned steps               = static_cast<unsigned>(std::stoul(argv[3]));
  // Number of thread blocks
  const unsigned simBlockDim         = static_cast<unsigned>(std::stoul(argv[4]));
  // Write frequency
  const unsigned writeFreq           = static_cast<unsigned>(std::stoul(argv[5]));
  // number of reduction threads
  const unsigned redTotalThreadCount = static_cast<unsigned>(std::stoul(argv[6]));
  // Number of reduction threads/blocks
  const unsigned redBlockDim         = static_cast<unsigned>(std::stoul(argv[7]));

  // Size of the simulation CUDA grid - number of blocks
  const unsigned simGridDim = (N + simBlockDim - 1) / simBlockDim;
  // Size of the reduction CUDA grid - number of blocks
  const unsigned redGridDim = (redTotalThreadCount + redBlockDim - 1) / redBlockDim;

  // Log benchmark setup
  std::printf("       NBODY GPU simulation\n"
              "N:                       %u\n"
              "dt:                      %f\n"
              "steps:                   %u\n"
              "threads/block:           %u\n"
              "blocks/grid:             %u\n"
              "reduction threads/block: %u\n"
              "reduction blocks/grid:   %u\n",
              N, dt, steps, simBlockDim, simGridDim, redBlockDim, redGridDim);

  const std::size_t recordsCount = (writeFreq > 0) ? (steps + writeFreq - 1) / writeFreq : 0;

  Particles hParticles{};

  /********************************************************************************************************************/
  /*                              TODO: CPU side memory allocation (pinned)                                           */
  /********************************************************************************************************************/

  hParticles.posX = new float[N];
  hParticles.posY = new float[N];
  hParticles.posZ = new float[N];
  hParticles.velX = new float[N];
  hParticles.velY = new float[N];
  hParticles.velZ = new float[N];
  hParticles.weight = new float[N];

  /********************************************************************************************************************/
  /*                              TODO: Fill memory descriptor layout                                                 */
  /********************************************************************************************************************/
  /*
   * Caution! Create only after CPU side allocation
   * parameters:
   *                            Stride of two            Offset of the first
   *       Data pointer       consecutive elements        element in FLOATS,
   *                          in FLOATS, not bytes            not bytes
  */
  MemDesc md(hParticles.posX,                 1,                          0,
             hParticles.posY,                 1,                          0,
             hParticles.posZ,                 1,                          0,
             hParticles.velX,                 1,                          0,
             hParticles.velY,                 1,                          0,
             hParticles.velZ,                 1,                          0,
             hParticles.weight,                 1,                          0,
             N,
             recordsCount);

  // Initialisation of helper class and loading of input data
  H5Helper h5Helper(argv[8], argv[9], md);

  try
  {
    h5Helper.init();
    h5Helper.readParticleData();
  }
  catch (const std::exception& e)
  {
    std::fprintf(stderr, "Error: %s\n", e.what());
    return EXIT_FAILURE;
  }

  Particles dParticles[2]{};

  /********************************************************************************************************************/
  /*                                     TODO: GPU side memory allocation                                             */
  /********************************************************************************************************************/

  unsigned long bytesSize = N * sizeof(float);

  CUDA_CALL(cudaMalloc(&(dParticles[0].posX), bytesSize));
  CUDA_CALL(cudaMalloc(&(dParticles[0].posY), bytesSize));
  CUDA_CALL(cudaMalloc(&(dParticles[0].posZ), bytesSize));
  CUDA_CALL(cudaMalloc(&(dParticles[0].velX), bytesSize));
  CUDA_CALL(cudaMalloc(&(dParticles[0].velY), bytesSize));
  CUDA_CALL(cudaMalloc(&(dParticles[0].velZ), bytesSize));
  CUDA_CALL(cudaMalloc(&(dParticles[0].weight), bytesSize));

  CUDA_CALL(cudaMalloc(&(dParticles[1].posX), bytesSize));
  CUDA_CALL(cudaMalloc(&(dParticles[1].posY), bytesSize));
  CUDA_CALL(cudaMalloc(&(dParticles[1].posZ), bytesSize));
  CUDA_CALL(cudaMalloc(&(dParticles[1].velX), bytesSize));
  CUDA_CALL(cudaMalloc(&(dParticles[1].velY), bytesSize));
  CUDA_CALL(cudaMalloc(&(dParticles[1].velZ), bytesSize));
  CUDA_CALL(cudaMalloc(&(dParticles[1].weight), bytesSize));

  /********************************************************************************************************************/
  /*                                     TODO: Memory transfer CPU -> GPU                                             */
  /********************************************************************************************************************/

  CUDA_CALL(cudaMemcpy(dParticles[0].posX, hParticles.posX, bytesSize, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(dParticles[0].posY, hParticles.posY, bytesSize, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(dParticles[0].posZ, hParticles.posZ, bytesSize, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(dParticles[0].velX, hParticles.velX, bytesSize, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(dParticles[0].velY, hParticles.velY, bytesSize, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(dParticles[0].velZ, hParticles.velZ, bytesSize, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(dParticles[0].weight, hParticles.weight, bytesSize, cudaMemcpyHostToDevice));

  CUDA_CALL(cudaMemcpy(dParticles[1].posX, hParticles.posX, bytesSize, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(dParticles[1].posY, hParticles.posY, bytesSize, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(dParticles[1].posZ, hParticles.posZ, bytesSize, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(dParticles[1].velX, hParticles.velX, bytesSize, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(dParticles[1].velY, hParticles.velY, bytesSize, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(dParticles[1].velZ, hParticles.velZ, bytesSize, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(dParticles[1].weight, hParticles.weight, bytesSize, cudaMemcpyHostToDevice));

  /********************************************************************************************************************/
  /*                                  TODO: Set dynamic shared memory computation                                     */
  /********************************************************************************************************************/
  const std::size_t sharedMemSize = blockDim.x * sizeof(float) * 7;

  // Start measurement
  const auto start = std::chrono::steady_clock::now();

  for (unsigned s = 0u; s < steps; ++s)
  {
    const unsigned srcIdx = s % 2;        // source particles index
    const unsigned dstIdx = (s + 1) % 2;  // destination particles index

    /******************************************************************************************************************/
    /*                   TODO: GPU kernel invocation with correctly set dynamic memory size                           */
    /******************************************************************************************************************/

    calculateVelocity<<<simGridDim, simBlockDim,
                        sharedMemSize>>>(dParticles[srcIdx], dParticles[dstIdx], N, dt);
  }

  // Wait for all CUDA kernels to finish
  CUDA_CALL(cudaDeviceSynchronize());

  // End measurement
  const auto end = std::chrono::steady_clock::now();

  // Approximate simulation wall time
  const float elapsedTime = std::chrono::duration<float>(end - start).count();
  std::printf("Time: %f s\n", elapsedTime);

  const unsigned resIdx = steps % 2;    // result particles index

  /********************************************************************************************************************/
  /*                                     TODO: Memory transfer GPU -> CPU                                             */
  /********************************************************************************************************************/

  CUDA_CALL(cudaMemcpy(hParticles.posX, dParticles[resIdx].posX, bytesSize, cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaMemcpy(hParticles.posY, dParticles[resIdx].posY, bytesSize, cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaMemcpy(hParticles.posZ, dParticles[resIdx].posZ, bytesSize, cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaMemcpy(hParticles.velX, dParticles[resIdx].velX, bytesSize, cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaMemcpy(hParticles.velY, dParticles[resIdx].velY, bytesSize, cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaMemcpy(hParticles.velZ, dParticles[resIdx].velZ, bytesSize, cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaMemcpy(hParticles.weight, dParticles[resIdx].weight, bytesSize, cudaMemcpyDeviceToHost));

  // Compute reference center of mass on CPU
  const float4 refCenterOfMass = centerOfMassRef(md);

  std::printf("Reference center of mass: %f, %f, %f, %f\n",
              refCenterOfMass.x,
              refCenterOfMass.y,
              refCenterOfMass.z,
              refCenterOfMass.w);

  std::printf("Center of mass on GPU: %f, %f, %f, %f\n", 0.f, 0.f, 0.f, 0.f);

  // Writing final values to the file
  h5Helper.writeComFinal(refCenterOfMass);
  h5Helper.writeParticleDataFinal();

  /********************************************************************************************************************/
  /*                                     TODO: GPU side memory deallocation                                           */
  /********************************************************************************************************************/

  CUDA_CALL(cudaFree(dParticles[0].posX));
  CUDA_CALL(cudaFree(dParticles[0].posY));
  CUDA_CALL(cudaFree(dParticles[0].posZ));
  CUDA_CALL(cudaFree(dParticles[0].velX));
  CUDA_CALL(cudaFree(dParticles[0].velY));
  CUDA_CALL(cudaFree(dParticles[0].velZ));
  CUDA_CALL(cudaFree(dParticles[0].weight));

  CUDA_CALL(cudaFree(dParticles[1].posX));
  CUDA_CALL(cudaFree(dParticles[1].posY));
  CUDA_CALL(cudaFree(dParticles[1].posZ));
  CUDA_CALL(cudaFree(dParticles[1].velX));
  CUDA_CALL(cudaFree(dParticles[1].velY));
  CUDA_CALL(cudaFree(dParticles[1].velZ));
  CUDA_CALL(cudaFree(dParticles[1].weight));

  /********************************************************************************************************************/
  /*                                     TODO: CPU side memory deallocation                                           */
  /********************************************************************************************************************/

  delete[] hParticles.posX;
  delete[] hParticles.posY;
  delete[] hParticles.posZ;
  delete[] hParticles.velX;
  delete[] hParticles.velY;
  delete[] hParticles.velZ;
  delete[] hParticles.weight;

}// end of main
//----------------------------------------------------------------------------------------------------------------------
