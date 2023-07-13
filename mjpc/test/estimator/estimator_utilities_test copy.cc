// Copyright 2023 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <absl/random/random.h>
#include <mujoco/mujoco.h>

#include <vector>

#include "gtest/gtest.h"
#include "mjpc/estimators/estimator.h"
#include "mjpc/estimators/trajectory.h"
#include "mjpc/test/load.h"
#include "mjpc/test/simulation.h"
#include "mjpc/threadpool.h"
#include "mjpc/utilities.h"

namespace mjpc {
namespace {

// linear interpolation between configurations (t in [0, 1])
void LinearInterpolateConfiguration(const mjModel* model, double* qinterp,
                                    double* DqinterpDqpos1,
                                    double* DqinterpDqpos2, const double* qpos1,
                                    const double* qpos2, double t) {
  // check for endpoints
  if (t < 1.0e-6) {
    mju_copy(qinterp, qpos1, model->nq);
    return;
  } else if (t > 1.0 - 1.0e-6) {
    mju_copy(qinterp, qpos2, model->nq);
    return;
  }

  // slerp derivatives
  double DsDq1[9];
  double DsDq2[9];

  // dimension
  int nv = model->nv;

  // loop over joints
  for (int j = 0; j < model->njnt; j++) {
    // get addresses in qpos and qvel
    int padr = model->jnt_qposadr[j];
    int vadr = model->jnt_dofadr[j];

    switch (model->jnt_type[j]) {
      case mjJNT_FREE:
        for (int i = 0; i < 3; i++) {
          // interpolate
          qinterp[padr + i] = (1.0 - t) * qpos1[padr + i] + t * qpos2[padr + i];

          // -- interpolation derivative --//

          // wrt qpos1
          if (DqinterpDqpos1) {
            DqinterpDqpos1[(vadr + i) * nv + (vadr + i)] = 1.0 - t;
          }

          // wrt qpos2
          if (DqinterpDqpos2) {
            DqinterpDqpos2[(vadr + i) * nv + (vadr + i)] = t;
          }
        }
        padr += 3;
        vadr += 3;

        // continute with rotations
        [[fallthrough]];

      case mjJNT_BALL:
        // interpolate quaternion
        Slerp(qinterp + padr, qpos1 + padr, qpos2 + padr, t,
              DqinterpDqpos1 ? DsDq1 : NULL, DqinterpDqpos2 ? DsDq2 : NULL);

        // derivative wrt qpos1
        if (DqinterpDqpos1) {
          SetBlockInMatrix(DqinterpDqpos1, DsDq1, 1.0, nv, nv, 3, 3, vadr,
                           vadr);
        }
        // derivative wrt qpos2
        if (DqinterpDqpos2) {
          SetBlockInMatrix(DqinterpDqpos2, DsDq2, 1.0, nv, nv, 3, 3, vadr,
                           vadr);
        }

        break;

      case mjJNT_HINGE:
      case mjJNT_SLIDE:
        // interpolate
        qinterp[padr] = (1.0 - t) * qpos1[padr] + t * qpos2[padr];

        // wrt qpos1
        if (DqinterpDqpos1) {
          DqinterpDqpos1[vadr * nv + vadr] = 1.0 - t;
        }

        // wrt qpos2
        if (DqinterpDqpos2) {
          DqinterpDqpos2[vadr * nv + vadr] = t;
        }
    }
  }
}

/* 
// position
x1 = s1(q0_current, q1_current, t_current) 

dx1dq = [ds1dq0_current ds1dq1_current]

// velocity
v1(s0(q0_previous, q1_previous, t_previous), s1(q0_current, q1_current, t_current)) 
= diffPos(s1 - s0) / h

dv1dq = [(-ddpds0 * ds0dq0previous / h) (-ddpds0 * ds0dq1previous / h) (ddpds1 * ds1dq0current / h) (ddpds1 * ds1dq1current / h)]

// acceleration
a1(s0(q0_previous, q1_previous, t_previous), s1(q0_current, q1_current, t_current), s2(q0_next, q1_next, t_next))
= (diffPos2(s2 - s1) / h - diffPos1(s1 - s0) / h) / h

da1dq = [(-ddp1ds0 * ds0dq0previous / h^2) (-ddp1s0 * ds0dq1previous / h^2) ((-ddp2ds1 + ddp1ds1) * ds1dq0current) / h^2) ((-ddp2ds1 + ddp1ds1) * ds1dq1current) / h^2) (ddp2ds2 * ds2dq0next / h^2) (ddp2ds2 * ds2dq1next / h^2)]
*/
// sample configuration, velocities, and accelerations
void SampleConfigurationToVelocityAcceleration(
    mjModel* model, const EstimatorTrajectory<double>& input_configuration,
    const EstimatorTrajectory<double>& input_time, int input_length,
    const EstimatorTrajectory<double>& sample_time, int sample_length,
    EstimatorTrajectory<double>& configuration,
    EstimatorTrajectory<double>& velocity,
    EstimatorTrajectory<double>& acceleration,
    double* derivative_configuration_current,
    double* derivative_configuration_previous,
    double* derivative_configuration_next, int* bounds_current,
    int* bounds_previous, int* bounds_next) {
  // dimension
  int nq = model->nq, nv = model->nv;

  // convert to std::vector
  // TODO(taylor): use input time as is?
  std::vector<double> sequence(input_length);
  for (int i = 0; i < input_length; i++) {
    sequence[i] = input_time.Get(i)[0];
  }

  // tmp
  // TODO(taylor): allocate elsewhere?
  std::vector<double> configuration_previous_(nq);
  std::vector<double> configuration_next_(nq);
  std::vector<double> velocity_next_(nv);

  double* configuration_previous = configuration_previous_.data();
  double* configuration_next = configuration_next_.data();
  double* velocity_next = velocity_next_.data();

  // loop over configurations
  for (int k = 0; k < sample_length; k++) {
    // sample time
    double time_current = sample_time.Get(k)[0];
    double time_previous = time_current - model->opt.timestep;
    double time_next = time_current + model->opt.timestep;

    // unpack bounds
    int* b_current = bounds_current + 2 * k;
    int* b_previous = bounds_previous + 2 * k;
    int* b_next = bounds_next + 2 * k;

    // find intervals
    FindInterval(b_current, sequence, time_current, input_length);
    FindInterval(b_previous, sequence, time_previous, input_length);
    FindInterval(b_next, sequence, time_next, input_length);

    // -- elements at bounds -- //

    // current
    const double* q0_current = input_configuration.Get(b_current[0]);
    const double* q1_current = input_configuration.Get(b_current[1]);
    double h0_current = input_time.Get(b_current[0])[0];
    double h1_current = input_time.Get(b_current[1])[0];

    // previous
    const double* q0_previous = input_configuration.Get(b_previous[0]);
    const double* q1_previous = input_configuration.Get(b_previous[1]);
    double h0_previous = input_time.Get(b_previous[0])[0];
    double h1_previous = input_time.Get(b_previous[1])[0];

    // next
    const double* q0_next = input_configuration.Get(b_next[0]);
    const double* q1_next = input_configuration.Get(b_next[1]);
    double h0_next = input_time.Get(b_next[0])[0];
    double h1_next = input_time.Get(b_next[1])[0];

    // interpolate current configuration
    double t_current = (time_current - h0_current) / (h1_current - h0_current);
    double* configuration_current = configuration.Get(k);
    double* Dconfiguration_currentDq0 = derivative_configuration_current + k * (2 * nv * nv);
    double* Dconfiguration_currentDq1 = Dconfiguration_currentDq0 + nv * nv;
    LinearInterpolateConfiguration(
        model, configuration_current,
        derivative_configuration_current ? Dconfiguration_currentDq0 : NULL,
        derivative_configuration_current ? Dconfiguration_currentDq1 : NULL,
        q0_current, q1_current, t_current);

    // interpolate previous configuration
    double t_previous =
        (time_previous - h0_previous) / (h1_previous - h0_previous);
    double* Dconfiguration_previousDq0 =
        derivative_configuration_previous + k * (2 * nv * nv);
    double* Dconfiguration_previousDq1 = Dconfiguration_previousDq0 + nv * nv;
    LinearInterpolateConfiguration(
        model, configuration_previous,
        derivative_configuration_previous ? Dconfiguration_previousDq0 : NULL,
        derivative_configuration_previous ? Dconfiguration_previousDq1 : NULL,
        q0_previous, q1_previous, t_previous);

    // interpolate next configuration
    double t_next = (time_next - h0_next) / (h1_next - h0_next);
    double* Dconfiguration_nextDq0 =
        derivative_configuration_next + k * (2 * nv * nv);
    double* Dconfiguration_nextDq1 = Dconfiguration_nextDq0 + nv * nv;
    LinearInterpolateConfiguration(
        model, configuration_next,
        derivative_configuration_next ? Dconfiguration_nextDq0 : NULL,
        derivative_configuration_next ? Dconfiguration_nextDq1 : NULL, q0_next,
        q1_next, t_next);

    // compute current velocity
    double* velocity_current = velocity.Get(k);
    mj_differentiatePos(model, velocity_current, model->opt.timestep,
                        configuration_previous, configuration_current);

    // compute next velocity
    mj_differentiatePos(model, velocity_next, model->opt.timestep,
                        configuration_current, configuration_next);

    // compute acceleration
    double* acceleration_current = acceleration.Get(k);
    mju_sub(acceleration_current, velocity_next, velocity_current, nv);
    mju_scl(acceleration_current, acceleration_current,
            1.0 / model->opt.timestep, nv);

    // -- times + bounds -- //

    // sample index
    printf("sample (%i)\n", k);

    // previous
    printf("  time_previous = %.5f : bounds = [%i, %i]\n", time_previous,
           b_previous[0], b_previous[1]);
    printf("  configuration_previous = ");
    mju_printMat(configuration_previous, 1, nq);
    printf("  Dconfiguration_previousDq0 = ");
    mju_printMat(Dconfiguration_previousDq0, nv, nv);
    printf("  Dconfiguration_previousDq1 = ");
    mju_printMat(Dconfiguration_previousDq1, nv, nv);

    // current
    printf("  time_current = %.5f : bounds = [%i, %i]\n", time_current,
           b_current[0], b_current[1]);

    printf("  configuration_current = ");
    mju_printMat(configuration_current, 1, nq);
    printf("  Dconfiguration_currentDq0 = ");
    mju_printMat(Dconfiguration_currentDq0, nv, nv);
    printf("  Dconfiguration_currentDq1 = ");
    mju_printMat(Dconfiguration_currentDq1, nv, nv);

    // next
    printf("  time_next = %.5f : bounds = [%i, %i]\n", time_next,
           b_next[0], b_next[1]);

    printf("  configuration_next = ");
    mju_printMat(configuration_next, 1, nq);
    printf("  Dconfiguration_nextDq0 = ");
    mju_printMat(Dconfiguration_nextDq0, nv, nv);
    printf("  Dconfiguration_nextDq1 = ");
    mju_printMat(Dconfiguration_nextDq1, nv, nv);

    printf("\n");
  }
}

TEST(ConfigurationInterpolation, Particle1D) {
  // load model
  mjModel* model = LoadTestModel("estimator/particle/task1d.xml");
  mjData* data = mj_makeData(model);

  // dimensions
  int nq = model->nq, nv = model->nv;

  // -- simulate -- //
  const int T = 10;
  Simulation sim(model, T);
  auto controller = [](double* ctrl, double time) {
    ctrl[0] = mju_sin(10 * time);
    ctrl[1] = 10 * mju_cos(10 * time);
  };
  sim.Rollout(controller);

  // print configurations
  printf("rollout:\n");
  for (int t = 0; t < T; t++) {
    printf("configuration (time = %.4f) = ", sim.time.Get(t)[0]);
    mju_printMat(sim.qpos.Get(t), 1, nq);
  }

  // interpolation
  printf("interpolation times:\n");
  int T_sample = T - 3;
  EstimatorTrajectory<double> time_interp(1, T_sample);
  EstimatorTrajectory<double> configuration_interp(nq, T_sample);
  EstimatorTrajectory<double> velocity_interp(nv, T_sample);
  EstimatorTrajectory<double> acceleration_interp(nv, T_sample);

  for (int i = 0; i < T_sample; i++) {
    // time
    double time = 0.5 * (sim.time.Get(i + 1)[0] + sim.time.Get(i + 2)[0]);
    time_interp.Set(&time, i);
  }

  // bounds 
  std::vector<int> bounds_current(2 * T_sample);
  std::vector<int> bounds_previous(2 * T_sample);
  std::vector<int> bounds_next(2 * T_sample);

  // derivatives 
  std::vector<double> derivative_configuration_current(2 * nv * nv * T_sample);
  std::vector<double> derivative_configuration_previous(2 * nv * nv * T_sample);
  std::vector<double> derivative_configuration_next(2 * nv * nv * T_sample);

  // sample
  SampleConfigurationToVelocityAcceleration(
      model, sim.qpos, sim.time, T, time_interp, T_sample, configuration_interp,
      velocity_interp, acceleration_interp,
      derivative_configuration_current.data(),
      derivative_configuration_previous.data(),
      derivative_configuration_next.data(), bounds_current.data(),
      bounds_previous.data(), bounds_next.data());

  // delete data + model
  mj_deleteData(data);
  mj_deleteModel(model);
}

// TEST(RecursivePrior, ConditionMatrixDense) {
//   // dimensions
//   const int n = 3;
//   const int n0 = 1;
//   const int n1 = n - n0;

//   // symmetric matrix
//   double mat[n * n] = {1.0, 0.1, 0.01, 0.1, 1.0, 0.1, 0.01, 0.1, 1.0};

//   // scratch
//   double mat00[n0 * n0];
//   double mat10[n1 * n0];
//   double mat11[n1 * n1];
//   double tmp0[n1 * n0];
//   double tmp1[n1 * n1];
//   double res[n1 * n1];

//   // condition matrix
//   ConditionMatrix(res, mat, mat00, mat10, mat11, tmp0, tmp1, n, n0, n1);

//   // solution
//   double solution[4] = {0.99, 0.099, 0.099, 0.9999};

//   // test
//   double error[n1 * n1];
//   mju_sub(error, res, solution, n1 * n1);

//   EXPECT_NEAR(mju_norm(error, n1 * n1), 0.0, 1.0e-4);
// }

// TEST(RecursivePrior, ConditionMatrixBand) {
//   // dimensions
//   const int n = 4;
//   const int n0 = 3;
//   const int n1 = n - n0;
//   const int nband = 2;

//   // symmetric matrix
//   double mat[n * n] = {1.0, 0.1, 0.0, 0.0, 0.1, 1.0, 0.1, 0.0,
//                        0.0, 0.1, 1.0, 0.1, 0.0, 0.0, 0.1, 1.0};

//   // scratch
//   double mat00[n0 * n0];
//   double mat10[n1 * n0];
//   double mat11[n1 * n1];
//   double tmp0[n1 * n0];
//   double tmp1[n1 * n1];
//   double bandfactor[n0 * n0];
//   double res[n1 * n1];

//   // condition matrix
//   ConditionMatrix(res, mat, mat00, mat10, mat11, tmp0, tmp1, n, n0, n1,
//                   bandfactor, nband);

//   // solution
//   double solution[n1 * n1] = {0.98989796};

//   // test
//   double error[n1 * n1];
//   mju_sub(error, res, solution, n1 * n1);

//   EXPECT_NEAR(mju_norm(error, n1 * n1), 0.0, 1.0e-4);
// }

// TEST(QuaternionInterpolation, Slerp) {
//   // quaternions
//   double quat0[4] = {1.0, 0.0, 0.0, 0.0};
//   double quat1[4] = {0.7071, 0.0, 0.7071, 0.0};
//   mju_normalize4(quat1);
//   // printf("quat0 = \n");
//   // mju_printMat(quat0, 1, 4);
//   // printf("quat1 = \n");
//   // mju_printMat(quat1, 1, 4);

//   // -- slerp: t = 0 -- //
//   double t = 0.0;
//   double slerp0[4];
//   double jac00[9];
//   double jac01[9];
//   Slerp(slerp0, quat0, quat1, t, jac00, jac01);

//   // printf("slerp0 = \n");
//   // mju_printMat(slerp0, 1, 4);

//   // test
//   double error[4];
//   mju_sub(error, slerp0, quat0, 4);
//   EXPECT_NEAR(mju_norm(error, 4), 0.0, 1.0e-4);

//   // -- slerp: t = 1.0 -- //
//   t = 1.0;
//   double slerp1[4];
//   double jac10[9];
//   double jac11[9];
//   Slerp(slerp1, quat0, quat1, t, jac10, jac11);

//   // printf("slerp1 = \n");
//   // mju_printMat(slerp1, 1, 4);

//   // test
//   mju_sub(error, slerp1, quat1, 4);
//   EXPECT_NEAR(mju_norm(error, 4), 0.0, 1.0e-4);

//   // -- slerp: t = 0.5 -- //
//   t = 0.5;
//   double slerp05[4];
//   double jac050[9];
//   double jac051[9];
//   Slerp(slerp05, quat0, quat1, t, jac050, jac051);

//   // printf("slerp05 = \n");
//   // mju_printMat(slerp05, 1, 4);

//   // test
//   double slerp05_solution[4] = {0.92387953, 0.0, 0.38268343, 0.0};
//   mju_sub(error, slerp05, slerp05_solution, 4);
//   EXPECT_NEAR(mju_norm(error, 4), 0.0, 1.0e-4);

//   // ----- jacobians ----- //
//   double jac0fdT[9];
//   double jac1fdT[9];
//   mju_zero(jac0fdT, 9);
//   mju_zero(jac1fdT, 9);
//   double jac0fd[9];
//   double jac1fd[9];

//   // -- t = 0.5 -- //
//   t = 0.5;

//   // finite difference
//   double eps = 1.0e-6;
//   double nudge[3];
//   mju_zero(nudge, 3);

//   for (int i = 0; i < 3; i++) {
//     // perturb
//     mju_zero(nudge, 3);
//     nudge[i] += eps;

//     // quat0 perturb
//     double q0i[4];
//     double slerp0i[4];
//     mju_copy(q0i, quat0, 4);
//     mju_quatIntegrate(q0i, nudge, 1.0);
//     Slerp(slerp0i, q0i, quat1, t, NULL, NULL);
//     double* dif0 = jac0fdT + 3 * i;
//     mju_subQuat(dif0, slerp0i, slerp05);
//     mju_scl(dif0, dif0, 1.0 / eps, 3);

//     // quat1 perturb
//     double q1i[4];
//     double slerp1i[4];
//     mju_copy(q1i, quat1, 4);
//     mju_quatIntegrate(q1i, nudge, 1.0);
//     Slerp(slerp1i, quat0, q1i, t, NULL, NULL);
//     double* dif1 = jac1fdT + 3 * i;
//     mju_subQuat(dif1, slerp1i, slerp05);
//     mju_scl(dif1, dif1, 1.0 / eps, 3);
//   }

//   // transpose results
//   mju_transpose(jac0fd, jac0fdT, 3, 3);
//   mju_transpose(jac1fd, jac1fdT, 3, 3);

//   // error
//   double error_jac[9];

//   mju_sub(error_jac, jac050, jac0fd, 9);
//   EXPECT_NEAR(mju_norm(error_jac, 9) / 9, 0.0, 1.0e-3);

//   mju_sub(error_jac, jac051, jac1fd, 9);
//   EXPECT_NEAR(mju_norm(error_jac, 9) / 9, 0.0, 1.0e-3);

//   // printf("jac0fd = \n");
//   // mju_printMat(jac0fd, 3, 3);

//   // printf("jac050 = \n");
//   // mju_printMat(jac050, 3, 3);

//   // printf("jac1fd = \n");
//   // mju_printMat(jac1fd, 3, 3);

//   // printf("jac051 = \n");
//   // mju_printMat(jac051, 3, 3);

//   // -- t = 0.0 -- //
//   t = 0.0;

//   // finite difference

//   for (int i = 0; i < 3; i++) {
//     // perturb
//     mju_zero(nudge, 3);
//     nudge[i] += eps;

//     // quat0 perturb
//     double q0i[4];
//     double slerp0i[4];
//     mju_copy(q0i, quat0, 4);
//     mju_quatIntegrate(q0i, nudge, 1.0);
//     Slerp(slerp0i, q0i, quat1, t, NULL, NULL);
//     double* dif0 = jac0fdT + 3 * i;
//     mju_subQuat(dif0, slerp0i, slerp0);
//     mju_scl(dif0, dif0, 1.0 / eps, 3);

//     // quat1 perturb
//     double q1i[4];
//     double slerp1i[4];
//     mju_copy(q1i, quat1, 4);
//     mju_quatIntegrate(q1i, nudge, 1.0);
//     Slerp(slerp1i, quat0, q1i, t, NULL, NULL);
//     double* dif1 = jac1fdT + 3 * i;
//     mju_subQuat(dif1, slerp1i, slerp0);
//     mju_scl(dif1, dif1, 1.0 / eps, 3);
//   }

//   // transpose results
//   mju_transpose(jac0fd, jac0fdT, 3, 3);
//   mju_transpose(jac1fd, jac1fdT, 3, 3);

//   // error
//   mju_sub(error_jac, jac00, jac0fd, 9);
//   EXPECT_NEAR(mju_norm(error_jac, 9) / 9, 0.0, 1.0e-3);

//   mju_sub(error_jac, jac01, jac1fd, 9);
//   EXPECT_NEAR(mju_norm(error_jac, 9) / 9, 0.0, 1.0e-3);

//   // printf("jac0fd = \n");
//   // mju_printMat(jac0fd, 3, 3);

//   // printf("jac00 = \n");
//   // mju_printMat(jac00, 3, 3);

//   // printf("jac1fd = \n");
//   // mju_printMat(jac1fd, 3, 3);

//   // printf("jac01 = \n");
//   // mju_printMat(jac01, 3, 3);

//   // -- t = 1.0 -- //
//   t = 1.0;

//   // finite difference

//   for (int i = 0; i < 3; i++) {
//     // perturb
//     mju_zero(nudge, 3);
//     nudge[i] += eps;

//     // quat0 perturb
//     double q0i[4];
//     double slerp0i[4];
//     mju_copy(q0i, quat0, 4);
//     mju_quatIntegrate(q0i, nudge, 1.0);
//     Slerp(slerp0i, q0i, quat1, t, NULL, NULL);
//     double* dif0 = jac0fdT + 3 * i;
//     mju_subQuat(dif0, slerp0i, slerp1);
//     mju_scl(dif0, dif0, 1.0 / eps, 3);

//     // quat1 perturb
//     double q1i[4];
//     double slerp1i[4];
//     mju_copy(q1i, quat1, 4);
//     mju_quatIntegrate(q1i, nudge, 1.0);
//     Slerp(slerp1i, quat0, q1i, t, NULL, NULL);
//     double* dif1 = jac1fdT + 3 * i;
//     mju_subQuat(dif1, slerp1i, slerp1);
//     mju_scl(dif1, dif1, 1.0 / eps, 3);
//   }

//   // transpose results
//   mju_transpose(jac0fd, jac0fdT, 3, 3);
//   mju_transpose(jac1fd, jac1fdT, 3, 3);

//   // error
//   mju_sub(error_jac, jac10, jac0fd, 9);
//   EXPECT_NEAR(mju_norm(error_jac, 9) / 9, 0.0, 1.0e-3);

//   mju_sub(error_jac, jac11, jac1fd, 9);
//   EXPECT_NEAR(mju_norm(error_jac, 9) / 9, 0.0, 1.0e-3);

//   // printf("jac0fd = \n");
//   // mju_printMat(jac0fd, 3, 3);

//   // printf("jac10 = \n");
//   // mju_printMat(jac10, 3, 3);

//   // printf("jac1fd = \n");
//   // mju_printMat(jac1fd, 3, 3);

//   // printf("jac11 = \n");
//   // mju_printMat(jac11, 3, 3);
// }

// TEST(FiniteDifferenceVelocityAcceleration, Particle2D) {
//   // load model
//   mjModel* model = LoadTestModel("estimator/particle/task1D.xml");
//   mjData* data = mj_makeData(model);

//   // dimensions
//   int nq = model->nq, nv = model->nv;

//   // pool
//   ThreadPool pool(1);

//   // ----- simulate ----- //

//   // controller
//   auto controller = [](double* ctrl, double time) {
//     ctrl[0] = mju_sin(10 * time);
//     ctrl[1] = mju_cos(10 * time);
//   };

//   // trajectories
//   int T = 200;
//   std::vector<double> qpos(nq * T);
//   std::vector<double> qvel(nv * T);
//   std::vector<double> qacc(nv * T);

//   // reset
//   mj_resetData(model, data);

//   // rollout
//   for (int t = 0; t < T; t++) {
//     // set control
//     controller(data->ctrl, data->time);

//     // forward computes instantaneous qacc
//     mj_forward(model, data);

//     // cache
//     mju_copy(qpos.data() + t * nq, data->qpos, nq);
//     mju_copy(qvel.data() + t * nv, data->qvel, nv);
//     mju_copy(qacc.data() + t * nv, data->qacc, nv);

//     // step using mj_Euler since mj_forward has been called
//     // see mj_ step implementation here
//     // https://
//     // github.com/deepmind/mujoco/blob/main/src/engine/engine_forward.c#L831
//     mj_Euler(model, data);
//   }

//   // ----- estimator ----- //

//   // initialize
//   Estimator estimator;
//   estimator.Initialize(model);
//   estimator.SetConfigurationLength(T);
//   mju_copy(estimator.configuration.Data(), qpos.data(), nq * T);

//   // compute velocity, acceleration
//   estimator.ConfigurationEvaluation(pool);

//   // velocity error
//   std::vector<double> velocity_error(nv * (T - 1));
//   mju_sub(velocity_error.data(), estimator.velocity.Data() + nv,
//           qvel.data() + nv, nv * (T - 1));

//   // velocity test
//   EXPECT_NEAR(mju_norm(velocity_error.data(), nv * (T - 1)), 0.0, 1.0e-5);
//   EXPECT_NEAR(mju_norm(estimator.velocity.Data(), nv), 0.0, 1.0e-5);

//   // acceleration error
//   std::vector<double> acceleration_error(nv * (T - 2));
//   mju_sub(acceleration_error.data(), estimator.acceleration.Data() + nv,
//           qacc.data() + nv, nv * (T - 2));

//   // acceleration test
//   EXPECT_NEAR(mju_norm(acceleration_error.data(), nv * (T - 2)),
//   0.0, 1.0e-5); EXPECT_NEAR(mju_norm(estimator.acceleration.Data(), nv),
//   0.0, 1.0e-5); EXPECT_NEAR(mju_norm(estimator.acceleration.Data() + nv * (T
//   - 1), nv), 0.0,
//               1.0e-5);

//   // delete data + model
//   mj_deleteData(data);
//   mj_deleteModel(model);
// }

// TEST(FiniteDifferenceVelocityAcceleration, Box3D) {
//   // load model
//   mjModel* model = LoadTestModel("estimator/box/task0.xml");
//   mjData* data = mj_makeData(model);

//   // dimensions
//   int nq = model->nq, nv = model->nv, nu = model->nu, ns =
//   model->nsensordata;

//   // pool
//   ThreadPool pool(1);

//   // ----- simulate ----- //
//   // trajectories
//   int T = 5;
//   std::vector<double> qpos(nq * (T + 1));
//   std::vector<double> qvel(nv * (T + 1));
//   std::vector<double> qacc(nv * T);
//   std::vector<double> ctrl(nu * T);
//   std::vector<double> qfrc_actuator(nv * T);
//   std::vector<double> sensordata(ns * (T + 1));

//   // reset
//   mj_resetData(model, data);

//   // initialize TODO(taylor): improve initialization
//   double qpos0[7] = {0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0};
//   double qvel0[6] = {0.4, 0.05, -0.22, 0.01, -0.03, 0.24};
//   mju_copy(data->qpos, qpos0, nq);
//   mju_copy(data->qvel, qvel0, nv);

//   // rollout
//   for (int t = 0; t < T; t++) {
//     // control
//     mju_zero(data->ctrl, model->nu);

//     // forward computes instantaneous qacc
//     mj_forward(model, data);

//     // cache
//     mju_copy(qpos.data() + t * nq, data->qpos, nq);
//     mju_copy(qvel.data() + t * nv, data->qvel, nv);
//     mju_copy(qacc.data() + t * nv, data->qacc, nv);
//     mju_copy(ctrl.data() + t * nu, data->ctrl, nu);
//     mju_copy(qfrc_actuator.data() + t * nv, data->qfrc_actuator, nv);
//     mju_copy(sensordata.data() + t * ns, data->sensordata, ns);

//     // step using mj_Euler since mj_forward has been called
//     // see mj_ step implementation here
//     //
//     https://github.com/deepmind/mujoco/blob/main/src/engine/engine_forward.c#L831
//     mj_Euler(model, data);
//   }

//   // final cache
//   mju_copy(qpos.data() + T * nq, data->qpos, nq);
//   mju_copy(qvel.data() + T * nv, data->qvel, nv);

//   mj_forward(model, data);
//   mju_copy(sensordata.data() + T * ns, data->sensordata, ns);

//   // ----- estimator ----- //

//   // initialize
//   Estimator estimator;
//   estimator.Initialize(model);
//   mju_copy(estimator.configuration.Data(), qpos.data(), nq * (T + 1));

//   // compute velocity, acceleration
//   estimator.ConfigurationEvaluation(pool);

//   // velocity error
//   std::vector<double> velocity_error(nv * T);
//   mju_sub(velocity_error.data(), estimator.velocity.Data() + nv,
//           qvel.data() + nv, nv * (T - 1));

//   // velocity test
//   EXPECT_NEAR(mju_norm(velocity_error.data(), nv * (T - 1)) / (nv * (T - 1)),
//               0.0, 1.0e-3);

//   // acceleration error
//   std::vector<double> acceleration_error(nv * T);
//   mju_sub(acceleration_error.data(), estimator.acceleration.Data() + nv,
//           qacc.data() + nv, nv * (T - 2));

//   // acceleration test
//   EXPECT_NEAR(
//       mju_norm(acceleration_error.data(), nv * (T - 1)) / (nv * (T - 1)),
//       0.0, 1.0e-3);

//   // delete data + model
//   mj_deleteData(data);
//   mj_deleteModel(model);
// }

}  // namespace
}  // namespace mjpc
