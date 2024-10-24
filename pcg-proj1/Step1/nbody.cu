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
  /*          TODO: CUDA kernel to calculate new particles velocity and position, collapse previous kernels           */
  /********************************************************************************************************************/

  const unsigned threadID = threadIdx.x + blockIdx.x * blockDim.x;
  if (threadID >= N) {
    return;
  }

  float* const pPosX = pIn.posX;
  float* const pPosY = pIn.posY;
  float* const pPosZ = pIn.posZ;
  float* const pVelX = pIn.velX;
  float* const pVelY = pIn.velY;
  float* const pVelZ = pIn.velZ;
  float* const pweight = pIn.weight;

  float* const pOutPosX = pOut.posX;
  float* const pOutPosY = pOut.posY;
  float* const pOutPosZ = pOut.posZ;
  float* const pOutVelX = pOut.velX;
  float* const pOutVelY = pOut.velY;
  float* const pOutVelZ = pOut.velZ;

  // tmp velocity
  float newGravitationVelX{};
  float newGravitationVelY{};
  float newGravitationVelZ{};

  float newCollisionVelX{};
  float newCollisionVelY{};
  float newCollisionVelZ{};

  // current particle
  const float posX = pPosX[threadID];
  const float posY = pPosY[threadID];
  const float posZ = pPosZ[threadID];
  const float velX = pVelX[threadID];
  const float velY = pVelY[threadID];
  const float velZ = pVelZ[threadID];
  const float weight = pweight[threadID];

  for (unsigned j = 0u; j < N; j++) {
    // neighbour partcle
    const float otherPosX = pPosX[j];
    const float otherPosY = pPosY[j];
    const float otherPosZ = pPosZ[j];
    const float otherVelX = pVelX[j];
    const float otherVelY = pVelY[j];
    const float otherVelZ = pVelZ[j];
    const float otherWeight = pweight[j];

    // distance between particles in dimensions
    const float dx = otherPosX - posX;
    const float dy = otherPosY - posY;
    const float dz = otherPosZ - posZ;

    // distance r between particles in 3D
    const float r2 = dx * dx + dy * dy + dz * dz;
    const float r = std::sqrt(r2) + __FLT_MIN__;

    // gravity force of the two particles
    const float f = G * weight * otherWeight / r2 + __FLT_MIN__;

    // SUM(F^(i+1))
    if (r > COLLISION_DISTANCE) {
      newGravitationVelX += dx / r * f;
      newGravitationVelY += dy / r * f;
      newGravitationVelZ += dz / r * f;
    } else {
      const bool isColliding = r > 0.0f;

      newCollisionVelX += (isColliding)
                  ? (((weight * velX - otherWeight * velX + 2.f * otherWeight * otherVelX) / (weight + otherWeight)) - velX)
                  : 0.f;
      newCollisionVelY += (isColliding)
                  ? (((weight * velY - otherWeight * velY + 2.f * otherWeight * otherVelY) / (weight + otherWeight)) - velY)
                  : 0.f;
      newCollisionVelZ += (isColliding)
                  ? (((weight * velZ - otherWeight * velZ + 2.f * otherWeight * otherVelZ) / (weight + otherWeight)) - velZ)
                  : 0.f;
    }
  }

  // Final results from the first kernel in step0
  newGravitationVelX *= dt / weight;
  newGravitationVelY *= dt / weight;
  newGravitationVelZ *= dt / weight;

  const float nextStepVelX = velX + newGravitationVelX + newCollisionVelX;
  const float nextStepVelY = velY + newGravitationVelY + newCollisionVelY;
  const float nextStepVelZ = velZ + newGravitationVelZ + newCollisionVelZ;

  pOutPosX[threadID] = posX + nextStepVelX * dt;
  pOutPosY[threadID] = posY + nextStepVelY * dt;
  pOutPosZ[threadID] = posZ + nextStepVelZ * dt;

  pOutVelX[threadID] = nextStepVelX;
  pOutVelY[threadID] = nextStepVelY;
  pOutVelZ[threadID] = nextStepVelZ;

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
