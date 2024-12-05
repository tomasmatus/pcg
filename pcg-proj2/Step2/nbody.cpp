/**
 * @file      nbody.cpp
 *
 * @author    Name Surname \n
 *            Faculty of Information Technology \n
 *            Brno University of Technology \n
 *            xlogin00@fit.vutbr.cz
 *
 * @brief     PCG Assignment 2
 *
 * @version   2023
 *
 * @date      04 October   2023, 09:00 (created) \n
 */

#include <cfloat>
#include <cmath>

#include "nbody.h"
#include "Vec.h"

/* Constants */
constexpr float G                  = 6.67384e-11f;
constexpr float COLLISION_DISTANCE = 0.01f;

/*********************************************************************************************************************/
/*                TODO: Fullfill Partile's and Velocitie's constructors, destructors and methods                     */
/*                                    for data copies between host and device                                        */
/*********************************************************************************************************************/

/**
 * @brief Constructor
 * @param N - Number of particles
 */
Particles::Particles(const unsigned N)
{
  this->N = N;
  posWei = new float4[N];
  vel = new float3[N];

  #pragma acc enter data copyin(this[0:1])
  #pragma acc enter data create(posWei[0:N])
  #pragma acc enter data create(vel[0:N])
}

/// @brief Destructor
Particles::~Particles()
{
  #pragma acc exit data delete(posWei[0:N])
  #pragma acc exit data delete(vel[0:N])
  #pragma acc exit data delete(this[0:1])

  delete[] posWei;
  delete[] vel;
}

/**
 * @brief Copy particles from host to device
 */
void Particles::copyToDevice()
{
  #pragma acc update device(posWei[0:N])
  #pragma acc update device(vel[0:N])
}

/**
 * @brief Copy particles from device to host
 */
void Particles::copyToHost()
{
  #pragma acc update host(posWei[0:N])
  #pragma acc update host(vel[0:N])
}

/*********************************************************************************************************************/

/**
 * Calculate velocity
 * @param pIn  - particles input
 * @param pOut - particles output
 * @param N    - Number of particles
 * @param dt   - Size of the time step
 */
void calculateVelocity(Particles& pIn, Particles& pOut, const unsigned N, float dt)
{
  /*******************************************************************************************************************/
  /*                    TODO: Calculate gravitation velocity, see reference CPU version,                             */
  /*                            you can use overloaded operators defined in Vec.h                                    */
  /*******************************************************************************************************************/

  #pragma acc parallel loop present(pIn, pOut)
  for (unsigned i = 0u; i < N; i++) {
    float3 newGravityVel{ 0 };
    float3 newCollisionVel{ 0 };
    const float3 curPos = { pIn.posWei[i].x, pIn.posWei[i].y, pIn.posWei[i].z };
    const float3 curVel = pIn.vel[i];
    const float curWeight = pIn.posWei[i].w;

    #pragma acc loop
    for (unsigned j = 0u; j < N; j++) {
      const float3 otherPos = { pIn.posWei[j].x, pIn.posWei[j].y, pIn.posWei[j].z };
      const float3 otherVel = pIn.vel[j];
      const float otherWeight = pIn.posWei[j].w;

      const float3 delta = otherPos - curPos;
      const float3 delta2 = delta * delta;

      const float r2 = delta2.x + delta2.y + delta2.z;
      const float r = std::sqrt(r2);

      const float f = G * curWeight * otherWeight / r2;

      newGravityVel += (r > COLLISION_DISTANCE) ? delta / r * f : 0.0f;
      newCollisionVel += (r > 0.0f && r < COLLISION_DISTANCE)
        ? (((curWeight * curVel - otherWeight * curVel + 2.0f * otherWeight * otherVel) / (curWeight + otherWeight)) - curVel)
        : 0.0f;
    }

    newGravityVel *= dt / curWeight;

    const float3 stepVel = curVel + newGravityVel + newCollisionVel;
    const float3 stepPos = curPos + stepVel * dt;

    pOut.vel[i] = stepVel;
    pOut.posWei[i].x = stepPos.x;
    pOut.posWei[i].y = stepPos.y;
    pOut.posWei[i].z = stepPos.z;
  }
}// end of calculate_gravitation_velocity
//----------------------------------------------------------------------------------------------------------------------

/**
 * Calculate particles center of mass
 * @param p         - particles
 * @param comBuffer - pointer to a center of mass buffer
 * @param N         - Number of particles
 */
void centerOfMass(Particles& p, float4* comBuffer, const unsigned N)
{
  /********************************************************************************************************************/
  /*                 TODO: Calculate partiles center of mass inside center of mass buffer                             */
  /********************************************************************************************************************/
  
  // round N to the nearest even number
  const unsigned maxN = (N % 2 == 0) ? N : N + 1;

  #pragma acc parallel loop gang present(p, comBuffer)
  for (unsigned i = 0u; i < maxN; i++) {
    comBuffer[i] = (i < N)
      ? p.posWei[i]
      : float4{ 0.0f };
  }

  comBuffer[N] = float4{ 0 };

  #pragma acc loop seq
  for (unsigned stride = maxN / 2; stride >= 1; stride /= 2) {
    #pragma acc parallel loop gang present(p, comBuffer)
    for (unsigned i = 0u; i < stride; i++) {
      if (i + stride >= N) {
        continue;
      }

      float4 com = comBuffer[i];
      float4 comOther = comBuffer[i + stride];

      const float4 d = {
        comOther.x - com.x,
        comOther.y - com.y,
        comOther.z - com.z,
        ((comOther.w + com.w) > 0.0f)
          ? (comOther.w / (comOther.w + com.w))
          : 0.0f
      };

      comBuffer[i].x += d.x * d.w;
      comBuffer[i].y += d.y * d.w;
      comBuffer[i].z += d.z * d.w;
      comBuffer[i].w += comOther.w;
    }
  }
}// end of centerOfMass
//----------------------------------------------------------------------------------------------------------------------

/**
 * CPU implementation of the Center of Mass calculation
 * @param particles - All particles in the system
 * @param N         - Number of particles
 */
float4 centerOfMassRef(MemDesc& memDesc)
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
