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

#include <vector>
#include <array>

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
  TestNearest(4, reinterpret_cast<mjtNum *>(points1), query1, nearest1);

  // A point in the interior of the square is unchanged
  // counter-clockwise points
  mjtNum points2[4][2] = {{0, 0}, {0, 1}, {1, 1}, {1, 0}};
  mjtNum query2[2] = {0.5, 0.5};
  mjtNum nearest2[2] = {0.5, 0.5};
  TestNearest(4, reinterpret_cast<mjtNum *>(points2), query2, nearest2);

  // A point in the interior of the square is unchanged
  // clockwise points, not all on hull
  mjtNum points3[5][2] = {{0, 0}, {0.5, 0.1}, {0, 1}, {1, 1}, {1, 0}};
  mjtNum query3[2] = {0.5, 0.5};
  mjtNum nearest3[2] = {0.5, 0.5};
  TestNearest(5, reinterpret_cast<mjtNum *>(points3), query3, nearest3);

  // A point outside is projected into the middle of an edge
  mjtNum query4[2] = {1.5, 0.5};
  mjtNum nearest4[2] = {1.0, 0.5};
  TestNearest(5, reinterpret_cast<mjtNum *>(points3), query4, nearest4);

  // A point outside is projected into the middle of an edge
  mjtNum query5[2] = {0.5, -0.5};
  mjtNum nearest5[2] = {0.5, 0.0};
  TestNearest(5, reinterpret_cast<mjtNum *>(points3), query5, nearest5);
}

#define ARRAY(arr, n) (mjtNum[n][2] arr)

TEST(ConvexHull2d, PointsHullDegenerate) {
  // A triangle with an interior point
  TestHull(4,
           std::array<mjtNum, 8>{{
               0, 0,
               0, 3,
               1, 1,
               4, 2,
           }}.data(),
           3, std::array<int, 3>{3, 1, 0}.data());

  // A quadrilateral
  TestHull(4,
           std::array<mjtNum, 8>{{
                                     0,
                                     0,
                                     0,
                                     3,
                                     4,
                                     1,
                                     4,
                                     2,
                                 }}
               .data(),
           4, std::array<int, 4>{3, 1, 0, 2}.data());

  // A square and its midpoint
  TestHull(5,
           std::array<mjtNum, 10>{{
               0, 1, 1, 1, 1, 0, 0, 0, 0.5, 0.5}}.data(),
           4, std::array<int, 4>{1, 0, 3, 2}.data());

  // Three collinear points on the x-axis
  TestHull(3, std::array<mjtNum, 6>{{0, 0, 1, 0, 2, 0}}.data(),
           2, std::array<int, 2>{2, 0}.data());

  // Three collinear points along the y-axis
  TestHull(3, std::array<mjtNum, 6>{{0, 0, 0, 1, 0, 2}}.data(),
           2, std::array<int, 2>{2, 0}.data());

  // Three collinear points on a generic line
  TestHull(3,
           std::array<mjtNum, 6>{{0.30629114, 0.59596112, 0.7818747, 0.81709791,
                                  (0.30629114 + 0.7818747) / 2,
                                  (0.59596112 + 0.81709791) / 2}}
               .data(),
           2, std::array<int, 2>{1, 0}.data());

  // A square plus the midpoints of the edges
  TestHull(8,
           std::array<mjtNum, 16>{{
               0, 1,
               0, 2,
               1, 2,
               2, 2,
               2, 1,
               2, 0,
               1, 0,
               0, 0,
           }}.data(),
           4, std::array<int, 4>{3, 1, 7, 5}.data());

  // A generic triangle plus the midpoints of the edges
  TestHull(6,
           std::array<mjtNum, 12>{
               {0.30629114, 0.59596112, 0.7818747, 0.81709791, 0.17100688,
                0.32822273, (0.30629114 + 0.7818747) / 2,
                (0.59596112 + 0.81709791) / 2, (0.7818747 + 0.17100688) / 2,
                (0.81709791 + 0.32822273) / 2, (0.30629114 + 0.17100688) / 2,
                (0.59596112 + 0.32822273) / 2}}
               .data(),
           3, std::array<int, 3>{1, 0, 2}.data());
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
  fdg.Compute(quadratic, input, n);

  EXPECT_NEAR(fdg.gradient[0], 1.0, FD_TOLERANCE);
  EXPECT_NEAR(fdg.gradient[1], 1.0, FD_TOLERANCE);

  // Hessian
  FiniteDifferenceHessian fdh(2);
  fdh.Compute(quadratic, input, n);

  // test
  EXPECT_NEAR(fdh.hessian[0], 1.0, FD_TOLERANCE);
  EXPECT_NEAR(fdh.hessian[1], 0.0, FD_TOLERANCE);
  EXPECT_NEAR(fdh.hessian[2], 0.0, FD_TOLERANCE);
  EXPECT_NEAR(fdh.hessian[3], 1.0, FD_TOLERANCE);
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
  fdj.Compute(f, input, num_output, num_input);

  // test
  EXPECT_NEAR(fdj.jacobian[0], A[0], FD_TOLERANCE);
  EXPECT_NEAR(fdj.jacobian[1], A[1], FD_TOLERANCE);
  EXPECT_NEAR(fdj.jacobian[2], A[2], FD_TOLERANCE);
  EXPECT_NEAR(fdj.jacobian[3], A[3], FD_TOLERANCE);
  EXPECT_NEAR(fdj.jacobian[4], A[4], FD_TOLERANCE);
  EXPECT_NEAR(fdj.jacobian[5], A[5], FD_TOLERANCE);
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

TEST(DifferentiateQuaternionTest, DifferentiatePosBox2D) {
  // model
  mjModel *model = LoadTestModel("estimator/box/task2D.xml");

  // random qpos
  double qa[3];
  double qb[3];

  for (int i = 0; i < 3; i++) {
    absl::BitGen gen_;
    qa[i] = absl::Gaussian<double>(gen_, 0.0, 1.0);
    qb[i] = absl::Gaussian<double>(gen_, 0.0, 1.0);
  }

  // subQuat
  double v[3];
  mj_differentiatePos(model, v, model->opt.timestep, qa, qb);

  double eps = 1.0e-6;
  double Ja[9];       // Jacobian wrt to qa
  double Jb[9];       // Jacobian wrt to qb
  double JaT[9];      // Jacobian wrt to qa transposed
  double JbT[9];      // Jacobian wrt to qb transposed
  double dv[3];       // differentiatePos perturbation
  double dq[3];       // qpos perturbation
  double qa_copy[3];  // qa copy
  double qb_copy[3];  // qb copy

  for (int i = 0; i < 3; i++) {
    // perturbation
    mju_zero(dq, 3);
    dq[i] = 1.0;

    // Jacobian qa
    mju_copy(qa_copy, qa, model->nq);
    mj_integratePos(model, qa_copy, dq, eps);
    mj_differentiatePos(model, dv, model->opt.timestep, qa_copy, qb);

    mju_sub(JaT + i * 3, dv, v, 3);
    mju_scl(JaT + i * 3, JaT + i * 3, 1.0 / eps, 3);

    // Jacobian qb
    mju_copy(qb_copy, qb, 3);
    mj_integratePos(model, qb_copy, dq, eps);
    mj_differentiatePos(model, dv, model->opt.timestep, qa, qb_copy);

    mju_sub(JbT + i * 3, dv, v, 3);
    mju_scl(JbT + i * 3, JbT + i * 3, 1.0 / eps, 3);
  }

  // transpose result
  mju_transpose(Ja, JaT, 3, 3);
  mju_transpose(Jb, JbT, 3, 3);

  // ----- utilities ----- //
  double Ga[9];  // quaternion to 3D velocity Jacobian wrt to qa
  double Gb[9];  // quaternion to 3D velocity Jacobian wrt to qa

  // compute Jacobians
  DifferentiateDifferentiatePos(Ga, Gb, model, model->opt.timestep, qa, qb);

  // ----- error ----- //
  double error_a[9];
  double error_b[9];
  mju_sub(error_a, Ja, Ga, 9);
  mju_sub(error_b, Jb, Gb, 9);

  // ----- test ----- //
  EXPECT_NEAR(mju_norm(error_a, 9), 0.0, 1.0e-5);
  EXPECT_NEAR(mju_norm(error_b, 9), 0.0, 1.0e-5);
  mj_deleteModel(model);
}

TEST(DifferentiateQuaternionTest, DifferentiatePos) {
  // model
  mjModel *model = LoadTestModel("box.xml");

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
  mj_deleteModel(model);
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

  // bottom right block
  ri = 2;
  ci = 2;
  BlockFromMatrix(block, mat, rb, cb, rm, cm, ri, ci);
  double br[4] = {11, 12, 15, 16};
  mju_sub(error, block, br, 4);
  EXPECT_NEAR(mju_norm(error, 4), 0.0, 1.0e-6);

  // center block
  ri = 1;
  ci = 1;
  BlockFromMatrix(block, mat, rb, cb, rm, cm, ri, ci);
  double cc[4] = {6, 7, 10, 11};
  mju_sub(error, block, cc, 4);
  EXPECT_NEAR(mju_norm(error, 4), 0.0, 1.0e-6);
}

TEST(BandMatrix, Copy) {
  // dimensions
  int dblock = 20;
  int nblock = 3;
  int num_blocks = 32;
  // int dblock = 2;
  // int nblock = 3;
  // int num_blocks = 5;
  int ntotal = dblock * num_blocks;

  // ----- create random band matrix ----- //
  std::vector<double> F(ntotal * ntotal);
  std::vector<double> A(ntotal * ntotal);

  // sample matrix square root
  absl::BitGen gen_;
  for (int i = 0; i < ntotal * ntotal; i++) {
    F[i] = absl::Gaussian<double>(gen_, 0.0, 1.0);
  }

  // A = F' F
  mju_mulMatTMat(A.data(), F.data(), F.data(), ntotal, ntotal, ntotal);

  // band(A)
  DenseToBlockBand(A.data(), ntotal, dblock, nblock);

  // band copy
  std::vector<double> Acopy(ntotal * ntotal);
  std::vector<double> scratch(2 * dblock * dblock);
  SymmetricBandMatrixCopy(Acopy.data(), A.data(), dblock, nblock, ntotal,
                          num_blocks, 0, 0, 0, 0, scratch.data());

  // error
  std::vector<double> error(ntotal * ntotal);
  mju_sub(error.data(), Acopy.data(), A.data(), ntotal * ntotal);

  // test
  EXPECT_NEAR(mju_norm(error.data(), ntotal * ntotal), 0.0, 1.0e-3);

  // ----- test upper right copy ----- //
  int num_new = 7;
  std::vector<double> B(ntotal * ntotal);
  mju_zero(B.data(), ntotal * ntotal);
  SymmetricBandMatrixCopy(B.data(), A.data(), dblock, nblock, ntotal,
                          num_blocks - num_new, 0, 0, num_new, num_new,
                          scratch.data());

  // top left block from B
  std::vector<double> tl(dblock * (num_blocks - num_new) * dblock *
                         (num_blocks - num_new));
  mju_zero(tl.data(),
           dblock * (num_blocks - num_new) * dblock * (num_blocks - num_new));
  BlockFromMatrix(tl.data(), B.data(), dblock * (num_blocks - num_new),
                  dblock * (num_blocks - num_new), ntotal, ntotal, 0, 0);

  // bottom right block from A
  std::vector<double> br(dblock * (num_blocks - num_new) * dblock *
                         (num_blocks - num_new));
  mju_zero(br.data(),
           dblock * (num_blocks - num_new) * dblock * (num_blocks - num_new));
  BlockFromMatrix(br.data(), A.data(), dblock * (num_blocks - num_new),
                  dblock * (num_blocks - num_new), ntotal, ntotal,
                  dblock * num_new, dblock * num_new);

  // error
  mju_zero(error.data(), ntotal * ntotal);
  mju_sub(error.data(), tl.data(), br.data(),
          dblock * (num_blocks - num_new) * dblock * (num_blocks - num_new));

  // test
  EXPECT_NEAR(mju_norm(error.data(), dblock * (num_blocks - num_new) * dblock *
                                         (num_blocks - num_new)),
              0.0, 1.0e-3);
}

TEST(Norm, Infinity) {
  // double
  double x[] = {1.0, 2.0, -3.0};
  double rx = InfinityNorm(x, 3);
  EXPECT_NEAR(rx, 3.0, 1.0e-5);

  // int
  int y[] = {1, 2, -3};
  int ry = InfinityNorm(y, 3);
  EXPECT_EQ(ry, 3);
}

TEST(Trace, Mat123) {
  // matrix
  std::vector<double> mat = {1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0};

  // compute trace
  double trace = Trace(mat.data(), 3);

  // test
  EXPECT_NEAR(trace, 6.0, 1.0e-5);
}

TEST(Determinant, Mat3) {
  // matrix
  double mat[9] = {1.0, 0.1, 0.01, 0.1, 1.0, 0.1, 0.01, 0.1, 1.0};

  // determinant
  double det = Determinant3(mat);

  // test
  EXPECT_NEAR(det, 0.9801, 1.0e-5);
}

TEST(Inverse, Mat3) {
  // matrix
  double mat[9] = {1.0, 0.1, 0.01, 0.1, 1.0, 0.1, 0.01, 0.1, 1.0};

  // solution
  double sol[9] = {1.0101,   -0.10101, 0.0,      -0.10101, 1.0202,
                   -0.10101, 0.0,      -0.10101, 1.0101};

  // inverse
  double res[9];
  Inverse3(res, mat);

  // test
  EXPECT_NEAR(res[0], sol[0], 1.0e-5);
  EXPECT_NEAR(res[1], sol[1], 1.0e-5);
  EXPECT_NEAR(res[2], sol[2], 1.0e-5);
  EXPECT_NEAR(res[3], sol[3], 1.0e-5);
  EXPECT_NEAR(res[4], sol[4], 1.0e-5);
  EXPECT_NEAR(res[5], sol[5], 1.0e-5);
  EXPECT_NEAR(res[6], sol[6], 1.0e-5);
  EXPECT_NEAR(res[7], sol[7], 1.0e-5);
  EXPECT_NEAR(res[8], sol[8], 1.0e-5);
}

TEST(ConditionMatrixDense, Mat3Dense) {
  // dimensions
  const int n = 3;
  const int n0 = 1;
  const int n1 = n - n0;

  // symmetric matrix
  double mat[n * n] = {1.0, 0.1, 0.01, 0.1, 1.0, 0.1, 0.01, 0.1, 1.0};

  // scratch
  double mat00[n0 * n0];
  double mat10[n1 * n0];
  double mat11[n1 * n1];
  double tmp0[n1 * n0];
  double tmp1[n1 * n1];
  double res[n1 * n1];

  // condition matrix
  ConditionMatrix(res, mat, mat00, mat10, mat11, tmp0, tmp1, n, n0, n1);

  // solution
  double solution[4] = {0.99, 0.099, 0.099, 0.9999};

  // test
  double error[n1 * n1];
  mju_sub(error, res, solution, n1 * n1);

  EXPECT_NEAR(mju_norm(error, n1 * n1), 0.0, 1.0e-4);
}

TEST(ConditionMatrixBand, Mat4Band) {
  // dimensions
  const int n = 4;
  const int n0 = 3;
  const int n1 = n - n0;
  const int nband = 2;

  // symmetric matrix
  double mat[n * n] = {1.0, 0.1, 0.0, 0.0, 0.1, 1.0, 0.1, 0.0,
                       0.0, 0.1, 1.0, 0.1, 0.0, 0.0, 0.1, 1.0};

  // scratch
  double mat00[n0 * n0];
  double mat10[n1 * n0];
  double mat11[n1 * n1];
  double tmp0[n1 * n0];
  double tmp1[n1 * n1];
  double bandfactor[n0 * n0];
  double res[n1 * n1];

  // condition matrix
  ConditionMatrix(res, mat, mat00, mat10, mat11, tmp0, tmp1, n, n0, n1,
                  bandfactor, nband);

  // solution
  double solution[n1 * n1] = {0.98989796};

  // test
  double error[n1 * n1];
  mju_sub(error, res, solution, n1 * n1);

  EXPECT_NEAR(mju_norm(error, n1 * n1), 0.0, 1.0e-4);
}

TEST(BlockInBand, Set) {
  // set up (0)
  double block0[9] = {1, 2, 3, 2, 4, 5, 3, 5, 6};
  int ntotal = 3;
  int nband = 3;
  int nblock = 3;

  // band
  double band0[9] = {0};

  // set block in band
  SetBlockInBand(band0, block0, 1.0, ntotal, nband, nblock, 0);

  // band solution
  double band0_sol[9] = {0, 0, 1, 0, 2, 4, 3, 5, 6};

  // test
  for (int i = 0; i < 9; i++) {
    EXPECT_NEAR(band0[i], band0_sol[i], 1.0e-6);
  }

  // set up (1)
  double block1a[4] = {1, 2, 2, 3};
  double block1b[4] = {6, 8, 8, 9};

  ntotal = 4;
  nband = 2;
  nblock = 2;

  double band1[8] = {0};

  // set block a in band
  SetBlockInBand(band1, block1a, 1.0, ntotal, nband, nblock, 0);

  // set block b in band
  SetBlockInBand(band1, block1b, 1.0, ntotal, nband, nblock, 2);

  // band solution
  double band1_sol[8] = {0, 1, 2, 3, 0, 6, 8, 9};

  // test
  for (int i = 0; i < 8; i++) {
    EXPECT_NEAR(band1[i], band1_sol[i], 1.0e-6);
  }

  // set up (2)
  double block2a[9] = {1, 2, 4, 2, 2, 4, 4, 4, 5};
  double block2b[9] = {1, 1, 7, 1, 1, 8, 7, 8, 9};

  ntotal = 4;
  nband = 3;
  nblock = 3;

  double band2[12] = {0};

  // set block2a in band
  SetBlockInBand(band2, block2a, 1.0, ntotal, nband, nblock, 0);

  // set block2b in band
  SetBlockInBand(band2, block2b, 1.0, ntotal, nband, nblock, 1);

  // band solution
  double band2_sol[12] = {0, 0, 1, 0, 2, 3, 4, 5, 6, 7, 8, 9};

  // test
  for (int i = 0; i < 12; i++) {
    EXPECT_NEAR(band2[i], band2_sol[i], 1.0e-6);
  }
}

}  // namespace
}  // namespace mjpc
