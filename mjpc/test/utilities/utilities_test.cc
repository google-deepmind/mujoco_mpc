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

#include "gtest/gtest.h"
#include <mujoco/mujoco.h>

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

}  // namespace
}  // namespace mjpc
