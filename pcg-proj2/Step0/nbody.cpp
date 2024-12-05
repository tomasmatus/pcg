/**
 * @file      nbody.cpp
 *
 * @author    Tomáš Matuš \n
 *            Faculty of Information Technology \n
 *            Brno University of Technology \n
 *            xmatus37@fit.vutbr.cz
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

/**
 * @brief Constructor
 * @param N - Number of particles
 */
Velocities::Velocities(const unsigned N)
{
  this->N = N;
  vel = new float3[N];

  #pragma acc enter data copyin(this[0:1])
  #pragma acc enter data create(vel[0:N])
}

/// @brief Destructor
Velocities::~Velocities()
{
  #pragma acc exit data delete(vel[0:N])
  #pragma acc exit data delete(this[0:1])

  delete[] vel;
}

/**
 * @brief Copy velocities from host to device
 */
void Velocities::copyToDevice()
{
  #pragma acc update device(vel[0:N])
}

/**
 * @brief Copy velocities from device to host
 */
void Velocities::copyToHost()
{
  #pragma acc update host(vel[0:N])
}

/*********************************************************************************************************************/

/**
 * Calculate gravitation velocity
 * @param p      - particles
 * @param tmpVel - temp array for velocities
 * @param N      - Number of particles
 * @param dt     - Size of the time step
 */
void calculateGravitationVelocity(Particles& p, Velocities& tmpVel, const unsigned N, float dt)
{
  /*******************************************************************************************************************/
  /*                    TODO: Calculate gravitation velocity, see reference CPU version,                             */
  /*                            you can use overloaded operators defined in Vec.h                                    */
  /*******************************************************************************************************************/

  #pragma acc parallel loop present(p, tmpVel) gang
  for (unsigned i = 0u; i < N; i++) {
    float3 newVel{ 0 };
    const float3 curPos = { p.posWei[i].x, p.posWei[i].y, p.posWei[i].z };
    const float curWeight = p.posWei[i].w;

    #pragma acc loop vector
    for (unsigned j = 0u; j < N; j++) {
      const float3 otherPos = { p.posWei[j].x, p.posWei[j].y, p.posWei[j].z };
      const float otherWeight = p.posWei[j].w;

      const float3 delta = otherPos - curPos;
      const float3 delta2 = delta * delta;

      const float r2 = delta2.x + delta2.y + delta2.z;
      const float r = std::sqrt(r2);

      const float f = G * curWeight * otherWeight / r2;

      newVel += (r > COLLISION_DISTANCE) ? delta / r * f : 0.0f;
    }

    newVel *= dt / curWeight;

    tmpVel.vel[i] = newVel;
  }

}// end of calculate_gravitation_velocity
//----------------------------------------------------------------------------------------------------------------------

/**
 * Calculate collision velocity
 * @param p      - particles
 * @param tmpVel - temp array for velocities
 * @param N      - Number of particles
 * @param dt     - Size of the time step
 */
void calculateCollisionVelocity(Particles& p, Velocities& tmpVel, const unsigned N, float dt)
{
  /*******************************************************************************************************************/
  /*                    TODO: Calculate collision velocity, see reference CPU version,                               */
  /*                            you can use overloaded operators defined in Vec.h                                    */
  /*******************************************************************************************************************/

  #pragma acc parallel loop present(p, tmpVel) gang
  for (unsigned i = 0u; i < N; i++) {
    float3 newVel{ 0 };
    const float3 curPos = { p.posWei[i].x, p.posWei[i].y, p.posWei[i].z };
    const float3 curVel = p.vel[i];
    const float curWeight = p.posWei[i].w;

    #pragma acc loop vector
    for (unsigned j = 0u; j < N; j++) {
      const float3 otherPos = { p.posWei[j].x, p.posWei[j].y, p.posWei[j].z };
      const float3 otherVel = p.vel[j];
      const float otherWeight = p.posWei[j].w;

      const float3 delta = otherPos - curPos;
      const float3 delta2 = delta * delta;

      const float r2 = delta2.x + delta2.y + delta2.z;
      const float r = std::sqrt(r2);

      newVel += (r > 0.0f && r < COLLISION_DISTANCE)
        ? (((curWeight * curVel - otherWeight * curVel + 2.0f * otherWeight * otherVel) / (curWeight + otherWeight)) - curVel)
        : 0.0f;
    }


    tmpVel.vel[i] += newVel;
  }
}// end of calculate_collision_velocity
//----------------------------------------------------------------------------------------------------------------------

/**
 * Update particles
 * @param p      - particles
 * @param tmpVel - temp array for velocities
 * @param N      - Number of particles
 * @param dt     - Size of the time step
 */
void updateParticles(Particles& p, Velocities& tmpVel, const unsigned N, float dt)
{
  /*******************************************************************************************************************/
  /*                    TODO: Update particles position and velocity, see reference CPU version,                     */
  /*                            you can use overloaded operators defined in Vec.h                                    */
  /*******************************************************************************************************************/

  #pragma acc parallel loop present(p, tmpVel)
  for (unsigned i = 0u; i < N; i++) {
    float3 curPos = { p.posWei[i].x, p.posWei[i].y, p.posWei[i].z };
    float3 curVel = p.vel[i];

    const float3 newVel = tmpVel.vel[i];

    curVel += newVel;
    curPos += curVel * dt;

    p.vel[i] = curVel;
    p.posWei[i].x = curPos.x;
    p.posWei[i].y = curPos.y;
    p.posWei[i].z = curPos.z;
  }

}// end of update_particle
//----------------------------------------------------------------------------------------------------------------------

/**
 * Calculate particles center of mass
 * @param p    - particles
 * @param com  - pointer to a center of mass
 * @param lock - pointer to a user-implemented lock
 * @param N    - Number of particles
 */
void centerOfMass(Particles& p, float4& com, int* lock, const unsigned N)
{

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
