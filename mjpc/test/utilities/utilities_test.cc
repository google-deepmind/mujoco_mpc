// Copyright 2022 DeepMind Technologies Limited
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

#include "mjpc/utilities.h"

#include <absl/random/random.h>
#include <mujoco/mujoco.h>

#include "gtest/gtest.h"
#include "mjpc/test/load.h"

namespace mjpc {
namespace {

static const int MAX_POINTS = 16;

static void TestHull(int num_points, mjtNum *points, int expected_num,
                     int *expected_hull) {
  int hull[MAX_POINTS];
  int num_hull = Hull2D(hull, num_points, points);

  EXPECT_EQ(num_hull, expected_num);
  for (int i = 0; i < num_hull; ++i) {
    EXPECT_EQ(hull[i], expected_hull[i]);
  }
}

static void TestNearest(int num_points, mjtNum *points, mjtNum *query,
                        mjtNum *expected_nearest) {
  int hull[MAX_POINTS];
  int num_hull = Hull2D(hull, num_points, points);
  mjtNum projected[2];

  NearestInHull(projected, query, points, hull, num_hull);
  for (int i = 0; i < 2; ++i) {
    EXPECT_EQ(projected[i], expected_nearest[i]);
  }
}

TEST(ConvexHull2d, Nearest) {
  // A point in the interior of the square is unchanged
  // clockwise points
  mjtNum points1[4][2] = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
  mjtNum query1[2] = {0.5, 0.5};
  mjtNum nearest1[2] = {0.5, 0.5};
  TestNearest(4, (mjtNum *)points1, query1, nearest1);

  // A point in the interior of the square is unchanged
  // counter-clockwise points
  mjtNum points2[4][2] = {{0, 0}, {0, 1}, {1, 1}, {1, 0}};
  mjtNum query2[2] = {0.5, 0.5};
  mjtNum nearest2[2] = {0.5, 0.5};
  TestNearest(4, (mjtNum *)points2, query2, nearest2);

  // A point in the interior of the square is unchanged
  // clockwise points, not all on hull
  mjtNum points3[5][2] = {{0, 0}, {0.5, 0.1}, {0, 1}, {1, 1}, {1, 0}};
  mjtNum query3[2] = {0.5, 0.5};
  mjtNum nearest3[2] = {0.5, 0.5};
  TestNearest(5, (mjtNum *)points3, query3, nearest3);

  // A point outside is projected into the middle of an edge
  mjtNum query4[2] = {1.5, 0.5};
  mjtNum nearest4[2] = {1.0, 0.5};
  TestNearest(5, (mjtNum *)points3, query4, nearest4);

  // A point outside is projected into the middle of an edge
  mjtNum query5[2] = {0.5, -0.5};
  mjtNum nearest5[2] = {0.5, 0.0};
  TestNearest(5, (mjtNum *)points3, query5, nearest5);
}

#define ARRAY(arr, n) (mjtNum[n][2] arr)

TEST(ConvexHull2d, PointsHullDegenerate) {
  // A triangle with an interior point
  TestHull(4,
           (mjtNum *)(mjtNum[4][2]){
               {0, 0},
               {0, 3},
               {1, 1},
               {4, 2},
           },
           3, (int[]){3, 1, 0});

  // A quadrilateral
  TestHull(4,
           (mjtNum *)(mjtNum[4][2]){
               {0, 0},
               {0, 3},
               {4, 1},
               {4, 2},
           },
           4, (int[]){3, 1, 0, 2});

  // A square and its midpoint
  TestHull(5,
           (mjtNum *)(mjtNum[5][2]){{0, 1}, {1, 1}, {1, 0}, {0, 0}, {0.5, 0.5}},
           4, (int[]){1, 0, 3, 2});

  // Three collinear points on the x-axis
  TestHull(3, (mjtNum *)(mjtNum[3][2]){{0, 0}, {1, 0}, {2, 0}}, 2,
           (int[]){2, 0});

  // Three collinear points along the y-axis
  TestHull(3, (mjtNum *)(mjtNum[3][2]){{0, 0}, {0, 1}, {0, 2}}, 2,
           (int[]){2, 0});

  // Three collinear points on a generic line
  TestHull(3,
           (mjtNum *)(mjtNum[3][2]){
               {0.30629114, 0.59596112},
               {0.7818747, 0.81709791},
               {(0.30629114 + 0.7818747) / 2, (0.59596112 + 0.81709791) / 2}},
           2, (int[]){1, 0});

  // A square plus the midpoints of the edges
  TestHull(8,
           (mjtNum *)(mjtNum[8][2]){
               {0, 1},
               {0, 2},
               {1, 2},
               {2, 2},
               {2, 1},
               {2, 0},
               {1, 0},
               {0, 0},
           },
           4, (int[]){3, 1, 7, 5});

  // A generic triangle plus the midpoints of the edges
  TestHull(6,
           (mjtNum *)(mjtNum[6][2]){
               {0.30629114, 0.59596112},
               {0.7818747, 0.81709791},
               {0.17100688, 0.32822273},
               {(0.30629114 + 0.7818747) / 2, (0.59596112 + 0.81709791) / 2},
               {(0.7818747 + 0.17100688) / 2, (0.81709791 + 0.32822273) / 2},
               {(0.30629114 + 0.17100688) / 2, (0.59596112 + 0.32822273) / 2}},
           3, (int[]){1, 0, 2});
}

const double FD_TOLERANCE = 1.0e-3;

TEST(FiniteDifference, Quadratic) {
  // quadratic
  auto quadratic = [](const double *x) {
    return 0.5 * (x[0] * x[0] + x[1] * x[1]);
  };
  const int n = 2;
  double input[n] = {1.0, 1.0};

  // gradient
  FiniteDifferenceGradient fdg(2);
  double *grad = fdg.Compute(quadratic, input, n);

  EXPECT_NEAR(grad[0], 1.0, FD_TOLERANCE);
  EXPECT_NEAR(grad[1], 1.0, FD_TOLERANCE);

  // Hessian
  FiniteDifferenceHessian fdh(2);
  double *hess = fdh.Compute(quadratic, input, n);

  // test
  EXPECT_NEAR(hess[0], 1.0, FD_TOLERANCE);
  EXPECT_NEAR(hess[1], 0.0, FD_TOLERANCE);
  EXPECT_NEAR(hess[2], 0.0, FD_TOLERANCE);
  EXPECT_NEAR(hess[3], 1.0, FD_TOLERANCE);
}

TEST(FiniteDifference, Jacobian) {
  // set up
  const int num_output = 2;
  const int num_input = 3;
  double A[num_output * num_input] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  auto f = [&A](double *output, const double *input) {
    mju_mulMatVec(output, A, input, num_output, num_input);
  };
  double input[3] = {1.0, 1.0, 1.0};

  // Jacobian
  FiniteDifferenceJacobian fdj(num_output, num_input);
  double *jac = fdj.Compute(f, input, num_output, num_input);

  // test
  EXPECT_NEAR(jac[0], A[0], FD_TOLERANCE);
  EXPECT_NEAR(jac[1], A[1], FD_TOLERANCE);
  EXPECT_NEAR(jac[2], A[2], FD_TOLERANCE);
  EXPECT_NEAR(jac[3], A[3], FD_TOLERANCE);
  EXPECT_NEAR(jac[4], A[4], FD_TOLERANCE);
  EXPECT_NEAR(jac[5], A[5], FD_TOLERANCE);
}

TEST(MatrixInMatrix, Set) {
  // set matrices within large matrix
  const int q = 8;
  std::vector<double> W(q * q);
  mju_zero(W.data(), q * q);

  std::vector<double> W1{1.0, 2.0, 3.0, 4.0};
  std::vector<double> W2{5.0, 6.0, 7.0, 8.0};
  std::vector<double> W3{9.0, 10.0, 11.0, 12.0};
  std::vector<double> W4{13.0, 14.0, 15.0, 16.0};

  // set W1
  mjpc::SetBlockInMatrix(W.data(), W1.data(), 1.0, q, q, 2, 2, 0, 0);

  // set W2
  mjpc::SetBlockInMatrix(W.data(), W2.data(), 1.0, q, q, 2, 2, 0, 4);

  // set W3
  mjpc::SetBlockInMatrix(W.data(), W3.data(), 1.0, q, q, 2, 2, 4, 0);

  // set W4
  mjpc::SetBlockInMatrix(W.data(), W4.data(), 1.0, q, q, 2, 2, 4, 4);

  std::vector<double> solution = {
      1.00000000,  2.00000000,  0.00000000,  0.00000000,  5.00000000,
      6.00000000,  0.00000000,  0.00000000,  3.00000000,  4.00000000,
      0.00000000,  0.00000000,  7.00000000,  8.00000000,  0.00000000,
      0.00000000,  0.00000000,  0.00000000,  0.00000000,  0.00000000,
      0.00000000,  0.00000000,  0.00000000,  0.00000000,  0.00000000,
      0.00000000,  0.00000000,  0.00000000,  0.00000000,  0.00000000,
      0.00000000,  0.00000000,  9.00000000,  10.00000000, 0.00000000,
      0.00000000,  13.00000000, 14.00000000, 0.00000000,  0.00000000,
      11.00000000, 12.00000000, 0.00000000,  0.00000000,  15.00000000,
      16.00000000, 0.00000000,  0.00000000,  0.00000000,  0.00000000,
      0.00000000,  0.00000000,  0.00000000,  0.00000000,  0.00000000,
      0.00000000,  0.00000000,  0.00000000,  0.00000000,  0.00000000,
      0.00000000,  0.00000000,  0.00000000,  0.00000000};

  std::vector<double> error(q * q);
  mju_sub(error.data(), solution.data(), W.data(), q * q);

  EXPECT_NEAR(mju_norm(error.data(), q * q) / error.size(), 0.0, 1.0e-5);
}

TEST(DifferentiateQuaternionTest, SubQuat) {
  // random quaternions
  double qa[4];
  double qb[4];

  for (int i = 0; i < 4; i++) {
    absl::BitGen gen_;
    qa[i] = absl::Gaussian<double>(gen_, 0.0, 1.0);
    qb[i] = absl::Gaussian<double>(gen_, 0.0, 1.0);
  }
  mju_normalize4(qa);
  mju_normalize4(qb);

  // subQuat
  double y[3];
  mju_subQuat(y, qa, qb);

  double eps = 1.0e-6;
  double Ja[9];       // quaternion difference Jacobian wrt to qa
  double Jb[9];       // quaternion difference Jacobian wrt to qb
  double JaT[9];      // quaternion difference Jacobian wrt to qa transposed
  double JbT[9];      // quaternion difference Jacobian wrt to qb transposed
  double dy[3];       // quaternion difference perturbation
  double dq[3];       // quaternion perturbation
  double qa_copy[4];  // qa copy
  double qb_copy[4];  // qb copy

  for (int i = 0; i < 3; i++) {
    // perturbation
    mju_zero3(dq);
    dq[i] = 1.0;

    // Jacobian qa
    mju_copy4(qa_copy, qa);
    mju_quatIntegrate(qa_copy, dq, eps);
    mju_subQuat(dy, qa_copy, qb);

    mju_sub3(JaT + i * 3, dy, y);
    mju_scl3(JaT + i * 3, JaT + i * 3, 1.0 / eps);

    // Jacobian qb
    mju_copy4(qb_copy, qb);
    mju_quatIntegrate(qb_copy, dq, eps);
    mju_subQuat(dy, qa, qb_copy);

    mju_sub3(JbT + i * 3, dy, y);
    mju_scl3(JbT + i * 3, JbT + i * 3, 1.0 / eps);
  }

  // transpose result
  mju_transpose(Ja, JaT, 3, 3);
  mju_transpose(Jb, JbT, 3, 3);

  // ----- utilities ----- //
  double Ga[9];  // quaternion to 3D velocity Jacobian wrt to qa
  double Gb[9];  // quaternion to 3D velocity Jacobian wrt to qa

  // compute Jacobians
  DifferentiateSubQuat(Ga, Gb, qa, qb);

  // ----- error ----- //
  double error_a[9];
  double error_b[9];
  mju_sub(error_a, Ja, Ga, 9);
  mju_sub(error_b, Jb, Gb, 9);

  // ----- test ----- //
  EXPECT_NEAR(mju_norm(error_a, 9) / 9, 0.0, 1.0e-5);
  EXPECT_NEAR(mju_norm(error_b, 9) / 9, 0.0, 1.0e-5);
}

TEST(DifferentiateQuaternionTest, DifferentiatePos) {
  // model
  mjModel *model = LoadTestModel("box3D.xml");

  // random qpos
  double qa[7];
  double qb[7];

  for (int i = 0; i < 7; i++) {
    absl::BitGen gen_;
    qa[i] = absl::Gaussian<double>(gen_, 0.0, 1.0);
    qb[i] = absl::Gaussian<double>(gen_, 0.0, 1.0);
  }
  mju_normalize4(qa + 3);
  mju_normalize4(qb + 3);

  // subQuat
  double v[6];
  mj_differentiatePos(model, v, model->opt.timestep, qa, qb);

  double eps = 1.0e-6;
  double Ja[36];      // Jacobian wrt to qa
  double Jb[36];      // Jacobian wrt to qb
  double JaT[36];     // Jacobian wrt to qa transposed
  double JbT[36];     // Jacobian wrt to qb transposed
  double dv[6];       // differentiatePos perturbation
  double dq[6];       // qpos perturbation
  double qa_copy[7];  // qa copy
  double qb_copy[7];  // qb copy

  for (int i = 0; i < 6; i++) {
    // perturbation
    mju_zero(dq, 6);
    dq[i] = 1.0;

    // Jacobian qa
    mju_copy(qa_copy, qa, model->nq);
    mj_integratePos(model, qa_copy, dq, eps);
    mj_differentiatePos(model, dv, model->opt.timestep, qa_copy, qb);

    mju_sub(JaT + i * 6, dv, v, 6);
    mju_scl(JaT + i * 6, JaT + i * 6, 1.0 / eps, 6);

    // Jacobian qb
    mju_copy(qb_copy, qb, 7);
    mj_integratePos(model, qb_copy, dq, eps);
    mj_differentiatePos(model, dv, model->opt.timestep, qa, qb_copy);

    mju_sub(JbT + i * 6, dv, v, 6);
    mju_scl(JbT + i * 6, JbT + i * 6, 1.0 / eps, 6);
  }

  // transpose result
  mju_transpose(Ja, JaT, 6, 6);
  mju_transpose(Jb, JbT, 6, 6);

  // ----- utilities ----- //
  double Ga[36];  // quaternion to 3D velocity Jacobian wrt to qa
  double Gb[36];  // quaternion to 3D velocity Jacobian wrt to qa

  // compute Jacobians
  DifferentiateDifferentiatePos(Ga, Gb, model, model->opt.timestep, qa, qb);

  // ----- error ----- //
  double error_a[36];
  double error_b[36];
  mju_sub(error_a, Ja, Ga, 36);
  mju_sub(error_b, Jb, Gb, 36);

  // ----- test ----- //
  EXPECT_NEAR(mju_norm(error_a, 36) / 36, 0.0, 1.0e-5);
  EXPECT_NEAR(mju_norm(error_b, 36) / 36, 0.0, 1.0e-5);
}

TEST(BandedMatrix, NonZeros) {
  EXPECT_EQ(BandMatrixNonZeros(2, 0), 0);
  EXPECT_EQ(BandMatrixNonZeros(2, 1), 2);
  EXPECT_EQ(BandMatrixNonZeros(3, 0), 0);
  EXPECT_EQ(BandMatrixNonZeros(3, 1), 3);
  EXPECT_EQ(BandMatrixNonZeros(3, 2), 7);
  EXPECT_EQ(BandMatrixNonZeros(3, 3), 9);
  EXPECT_EQ(BandMatrixNonZeros(4, 0), 0);
  EXPECT_EQ(BandMatrixNonZeros(4, 1), 4);
  EXPECT_EQ(BandMatrixNonZeros(4, 2), 10);
  EXPECT_EQ(BandMatrixNonZeros(4, 3), 14);
  EXPECT_EQ(BandMatrixNonZeros(4, 4), 16);
}

TEST(BlockFromMatrix, Block2x2Mat4x4) {
  // matrix
  int rm = 4;
  int cm = 4;
  double mat[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

  // block
  int rb = 2;
  int cb = 2;
  double block[4];

  // indices
  int ri, ci;

  // error
  double error[4];

  // upper left block
  ri = 0;
  ci = 0;
  BlockFromMatrix(block, mat, rb, cb, rm, cm, ri, ci);
  double ul[4] = {1, 2, 5, 6};
  mju_sub(error, block, ul, 4);
  EXPECT_NEAR(mju_norm(error, 4), 0.0, 1.0e-6);
  mju_printMat(block, 2, 2);

  // bottom right block
  ri = 2;
  ci = 2;
  BlockFromMatrix(block, mat, rb, cb, rm, cm, ri, ci);
  double br[4] = {11, 12, 15, 16};
  mju_sub(error, block, br, 4);
  EXPECT_NEAR(mju_norm(error, 4), 0.0, 1.0e-6);
  mju_printMat(block, 2, 2);

  // center block
  ri = 1;
  ci = 1;
  BlockFromMatrix(block, mat, rb, cb, rm, cm, ri, ci);
  double cc[4] = {6, 7, 10, 11};
  mju_sub(error, block, cc, 4);
  EXPECT_NEAR(mju_norm(error, 4), 0.0, 1.0e-6);
  mju_printMat(block, 2, 2);
}

}  // namespace
}  // namespace mjpc
