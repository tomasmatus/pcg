/**
 * @file      nbody.cu
 *
 * @author    Name Surname \n
 *            Faculty of Information Technology \n
 *            Brno University of Technology \n
 *            xlogin00@fit.vutbr.cz
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
 * CUDA kernel to calculate gravitation velocity
 * @param p      - particles
 * @param tmpVel - temp array for velocities
 * @param N      - Number of particles
 * @param dt     - Size of the time step
 */
__global__ void calculateGravitationVelocity(Particles p, Velocities tmpVel, const unsigned N, float dt)
{
  /********************************************************************************************************************/
  /*              TODO: CUDA kernel to calculate gravitation velocity, see reference CPU version                      */
  /********************************************************************************************************************/

  const unsigned threadID = threadIdx.x + blockIdx.x * blockDim.x;
  if (threadID >= N) {
    return;
  }

  const float* pPosX = p.posX;
  const float* pPosY = p.posY;
  const float* pPosZ = p.posZ;
  const float* pweight = p.weight;

  float* const tmpVelX = tmpVel.velX;
  float* const tmpVelY = tmpVel.velY;
  float* const tmpVelZ = tmpVel.velZ;

  // tmp velocity
  float newVelX = 0.0f;
  float newVelY = 0.0f;
  float newVelZ = 0.0f;

  // current particle
  const float posX = pPosX[threadID];
  const float posY = pPosY[threadID];
  const float posZ = pPosZ[threadID];
  const float weight = pweight[threadID];

  for (unsigned i = 0; i < N; i++) {
    // neighbour partcle
    const float otherPosX = pPosX[i];
    const float otherPosY = pPosY[i];
    const float otherPosZ = pPosZ[i];
    const float otherWeight = pweight[i];

    // distance between particles in dimensions
    const float dx = otherPosX - posX;
    const float dy = otherPosY - posY;
    const float dz = otherPosZ - posZ;

    // distance r between particles in 3D
    const float r2 = dx * dx + dy * dy + dz * dz;
    const float r = std::sqrt(r2) + std::numeric_limits<float>::min();

    // gravity force of the two particles
    const float f = G * weight * otherWeight / r2 + std::numeric_limits<float>::min();

    // SUM(F^(i+1))
    newVelX += (r > COLLISION_DISTANCE) ? dx / r * f : 0.f;
    newVelY += (r > COLLISION_DISTANCE) ? dy / r * f : 0.f;
    newVelZ += (r > COLLISION_DISTANCE) ? dz / r * f : 0.f;
  }

  newVelX *= dt / weight;
  newVelY *= dt / weight;
  newVelZ *= dt / weight;

  tmpVelX[threadID] = newVelX;
  tmpVelY[threadID] = newVelY;
  tmpVelZ[threadID] = newVelZ;

}// end of calculate_gravitation_velocity
//----------------------------------------------------------------------------------------------------------------------

/**
 * CUDA kernel to calculate collision velocity
 * @param p      - particles
 * @param tmpVel - temp array for velocities
 * @param N      - Number of particles
 * @param dt     - Size of the time step
 */
__global__ void calculateCollisionVelocity(Particles p, Velocities tmpVel, const unsigned N, float dt)
{
  /********************************************************************************************************************/
  /*              TODO: CUDA kernel to calculate collision velocity, see reference CPU version                        */
  /********************************************************************************************************************/


}// end of calculate_collision_velocity
//----------------------------------------------------------------------------------------------------------------------

/**
 * CUDA kernel to update particles
 * @param p      - particles
 * @param tmpVel - temp array for velocities
 * @param N      - Number of particles
 * @param dt     - Size of the time step
 */
__global__ void updateParticles(Particles p, Velocities tmpVel, const unsigned N, float dt)
{
  /********************************************************************************************************************/
  /*             TODO: CUDA kernel to update particles velocities and positions, see reference CPU version            */
  /********************************************************************************************************************/


}// end of update_particle
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
