/**
 * @file      nbody.cu
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

#include <device_launch_parameters.h>

#include "nbody.cuh"

/* Constants */
constexpr float G                  = 6.67384e-11f;
constexpr float COLLISION_DISTANCE = 0.01f;

/**
 * CUDA kernel to calculate new particles velocity and position
 * @param pIn  - particles in
 * @param pOut - particles out
 * @param N    - Number of particles
 * @param dt   - Size of the time step
 */
__global__ void calculateVelocity(Particles pIn, Particles pOut, const unsigned N, float dt)
{
  /********************************************************************************************************************/
  /*  TODO: CUDA kernel to calculate new particles velocity and position, use shared memory to minimize memory access */
  /********************************************************************************************************************/
  extern __shared__ float sharedMem[];

  // shared mem vectors
  float* const sharedPosX = sharedMem;
  float* const sharedPosY = sharedMem + blockDim.x;
  float* const sharedPosZ = sharedMem + 2 * blockDim.x;
  float* const sharedVelX = sharedMem + 3 * blockDim.x;
  float* const sharedVelY = sharedMem + 4 * blockDim.x;
  float* const sharedVelZ = sharedMem + 5 * blockDim.x;
  float* const sharedWeight = sharedMem + 6 * blockDim.x;

  const unsigned threadID = threadIdx.x + blockIdx.x * blockDim.x;

  // tmp velocity
  float newGravitationVelX{};
  float newGravitationVelY{};
  float newGravitationVelZ{};
  float newCollisionVelX{};
  float newCollisionVelY{};
  float newCollisionVelZ{};

  // current point
  const bool bound = threadID < N;
  const float posX = (bound) ? pIn.posX[threadID] : 0.0f;
  const float posY = (bound) ? pIn.posY[threadID] : 0.0f;
  const float posZ = (bound) ? pIn.posZ[threadID] : 0.0f;
  const float velX = (bound) ? pIn.velX[threadID] : 0.0f;
  const float velY = (bound) ? pIn.velY[threadID] : 0.0f;
  const float velZ = (bound) ? pIn.velZ[threadID] : 0.0f;
  const float weight = (bound) ? pIn.weight[threadID] : 0.0f;

  unsigned tileCount = ceil((float)N / blockDim.x);
  for (unsigned i = 0; i < tileCount; i++) {
    unsigned threadOffset = i * blockDim.x + threadIdx.x;

    // load data to shared memory
    const bool tileBound = threadOffset < N;
    sharedPosX[threadIdx.x] = (tileBound) ? pIn.posX[threadOffset] : 0.0f;
    sharedPosY[threadIdx.x] = (tileBound) ? pIn.posY[threadOffset] : 0.0f;
    sharedPosZ[threadIdx.x] = (tileBound) ? pIn.posZ[threadOffset] : 0.0f;
    sharedVelX[threadIdx.x] = (tileBound) ? pIn.velX[threadOffset] : 0.0f;
    sharedVelY[threadIdx.x] = (tileBound) ? pIn.velY[threadOffset] : 0.0f;
    sharedVelZ[threadIdx.x] = (tileBound) ? pIn.velZ[threadOffset] : 0.0f;
    sharedWeight[threadIdx.x] = (tileBound) ? pIn.weight[threadOffset] : 0.0f;

    __syncthreads();

    // loop over all points in the tile
    for (unsigned j = 0u; j < blockDim.x; j++) {
      // distance between particles in dimensions
      const float dx = sharedPosX[j] - posX;
      const float dy = sharedPosY[j] - posY;
      const float dz = sharedPosZ[j] - posZ;

      // distance r between particles in 3D
      const float r2 = dx * dx + dy * dy + dz * dz;
      const float r = sqrt(r2);

      // gravity force of the two particles
      const float f = G * weight * sharedWeight[j] / r2 + __FLT_MIN__;

      // SUM(F^(i+1))
      if (r > COLLISION_DISTANCE) {
        newGravitationVelX += dx / r * f;
        newGravitationVelY += dy / r * f;
        newGravitationVelZ += dz / r * f;
      } else {
        const bool isColliding = r > 0.0f;
        newCollisionVelX += (isColliding)
                    ? (((weight * velX - sharedWeight[j] * velX + 2.f * sharedWeight[j] * sharedVelX[j]) / (weight + sharedWeight[j])) - velX)
                    : 0.f;
        newCollisionVelY += (isColliding)
                    ? (((weight * velY - sharedWeight[j] * velY + 2.f * sharedWeight[j] * sharedVelY[j]) / (weight + sharedWeight[j])) - velY)
                    : 0.f;
        newCollisionVelZ += (isColliding)
                    ? (((weight * velZ - sharedWeight[j] * velZ + 2.f * sharedWeight[j] * sharedVelZ[j]) / (weight + sharedWeight[j])) - velZ)
                    : 0.f;
      }

    }

    __syncthreads();
  }

  // Final results from the first kernel in step0
  if (bound) {
    newGravitationVelX *= dt / weight;
    newGravitationVelY *= dt / weight;
    newGravitationVelZ *= dt / weight;

    const float nextStepVelX = velX + newGravitationVelX + newCollisionVelX;
    const float nextStepVelY = velY + newGravitationVelY + newCollisionVelY;
    const float nextStepVelZ = velZ + newGravitationVelZ + newCollisionVelZ;

    pOut.posX[threadID] = posX + nextStepVelX * dt;
    pOut.posY[threadID] = posY + nextStepVelY * dt;
    pOut.posZ[threadID] = posZ + nextStepVelZ * dt;

    pOut.velX[threadID] = nextStepVelX;
    pOut.velY[threadID] = nextStepVelY;
    pOut.velZ[threadID] = nextStepVelZ;
  }
}// end of calculate_gravitation_velocity
//----------------------------------------------------------------------------------------------------------------------

/**
 * CUDA kernel to calculate particles center of mass
 * @param p    - particles
 * @param com  - pointer to a center of mass
 * @param lock - pointer to a user-implemented lock
 * @param N    - Number of particles
 */
__global__ void centerOfMass(Particles p, float4* com, int* lock, const unsigned N)
{

}// end of centerOfMass
//----------------------------------------------------------------------------------------------------------------------

/**
 * CPU implementation of the Center of Mass calculation
 * @param particles - All particles in the system
 * @param N         - Number of particles
 */
__host__ float4 centerOfMassRef(MemDesc& memDesc)
{
  float4 com{};

  for (std::size_t i{}; i < memDesc.getDataSize(); i++)
  {
    const float3 pos = {memDesc.getPosX(i), memDesc.getPosY(i), memDesc.getPosZ(i)};
    const float  w   = memDesc.getWeight(i);

    // Calculate the vector on the line connecting current body and most recent position of center-of-mass
    // Calculate weight ratio only if at least one particle isn't massless
    const float4 d = {pos.x - com.x,
                      pos.y - com.y,
                      pos.z - com.z,
                      ((memDesc.getWeight(i) + com.w) > 0.0f)
                        ? ( memDesc.getWeight(i) / (memDesc.getWeight(i) + com.w))
                        : 0.0f};

    // Update position and weight of the center-of-mass according to the weight ration and vector
    com.x += d.x * d.w;
    com.y += d.y * d.w;
    com.z += d.z * d.w;
    com.w += w;
  }

  return com;
}// enf of centerOfMassRef
//----------------------------------------------------------------------------------------------------------------------
