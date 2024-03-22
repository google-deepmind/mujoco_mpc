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

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <new>
#include <string>
#include <string_view>
#include <vector>

// DEEPMIND INTERNAL IMPORT
#include <absl/container/flat_hash_map.h>
#include <absl/log/check.h>
#include <absl/strings/match.h>
#include <absl/strings/str_cat.h>
#include <absl/strings/str_split.h>
#include <mujoco/mujoco.h>

#include "mjpc/array_safety.h"

#if defined(__APPLE__) || defined(_WIN32)
#include <thread>
#else
#include <sched.h>
#endif

extern "C" {
#if defined(_WIN32) || defined(__CYGWIN__)
#define NOMINMAX
#include <windows.h>
#else
#if defined(__APPLE__)
#include <mach-o/dyld.h>
#endif
#include <unistd.h>
#endif
}

namespace mjpc {
// set mjData state
void SetState(const mjModel* model, mjData* data, const double* state) {
  mju_copy(data->qpos, state, model->nq);
  mju_copy(data->qvel, state + model->nq, model->nv);
  mju_copy(data->act, state + model->nq + model->nv, model->na);
}

// get mjData state
void GetState(const mjModel* model, const mjData* data, double* state) {
  mju_copy(state, data->qpos, model->nq);
  mju_copy(state + model->nq, data->qvel, model->nv);
  mju_copy(state + model->nq + model->nv, data->act, model->na);
}

// get numerical data from custom using string
double* GetCustomNumericData(const mjModel* m, std::string_view name) {
  for (int i = 0; i < m->nnumeric; i++) {
    if (std::string_view(m->names + m->name_numericadr[i]) == name) {
      return m->numeric_data + m->numeric_adr[i];
    }
  }
  return nullptr;
}

// get text data from custom using string
char* GetCustomTextData(const mjModel* m, std::string_view name) {
  for (int i = 0; i < m->ntext; i++) {
    if (std::string_view(m->names + m->name_textadr[i]) == name) {
      return m->text_data + m->text_adr[i];
    }
  }
  return nullptr;
}

// Clamp x between bounds, e.g., bounds[0] <= x[i] <= bounds[1]
void Clamp(double* x, const double* bounds, int n) {
  for (int i = 0; i < n; i++) {
    x[i] = mju_clip(x[i], bounds[2 * i], bounds[2 * i + 1]);
  }
}

int ReinterpretAsInt(double value) {
  return *std::launder(reinterpret_cast<const int*>(&value));
}

double ReinterpretAsDouble(int64_t value) {
  return *std::launder(reinterpret_cast<const double*>(&value));
}

absl::flat_hash_map<std::string, std::vector<std::string>>
ResidualSelectionLists(const mjModel* m) {
  absl::flat_hash_map<std::string, std::vector<std::string>> result;
  for (int i = 0; i < m->ntext; i++) {
    if (!absl::StartsWith(std::string_view(m->names + m->name_textadr[i]),
                          "residual_list_")) {
      continue;
    }
    std::string name = &m->names[m->name_textadr[i]];
    std::string_view options(m->text_data + m->text_adr[i]);
    result[absl::StripPrefix(name, "residual_list_")] =
        absl::StrSplit(options, '|');
  }
  return result;
}

std::string ResidualSelection(const mjModel* m, std::string_view name,
                              double residual_parameter) {
  std::string list_name = absl::StrCat("residual_list_", name);

  // we're using a double field to store an integer - reinterpret as an int
  int list_index = ReinterpretAsInt(residual_parameter);

  for (int i = 0; i < m->ntext; i++) {
    if (list_name == &m->names[m->name_textadr[i]]) {
      // get the nth element in the list of options (without constructing a
      // vector<string>)
      std::string_view options(m->text_data + m->text_adr[i]);
      for (std::string_view value : absl::StrSplit(options, '|')) {
        if (list_index == 0) return std::string(value);
        list_index--;
      }
    }
  }
  return "";
}

double ResidualParameterFromSelection(const mjModel* m, std::string_view name,
                                      const std::string_view value) {
  std::string list_name = absl::StrCat("residual_list_", name);
  for (int i = 0; i < m->ntext; i++) {
    if (list_name == &m->names[m->name_textadr[i]]) {
      int64_t list_index = 0;
      std::string_view options(m->text_data + m->text_adr[i],
                               m->text_size[i] - 1);
      std::vector<std::string> values = absl::StrSplit(options, '|');
      for (std::string_view v : absl::StrSplit(options, '|')) {
        if (v == value) {
          return ReinterpretAsDouble(list_index);
        }
        list_index++;
      }
    }
  }
  return 0;
}

// get sensor data using string
double* SensorByName(const mjModel* m, const mjData* d,
                     const std::string& name) {
  int id = mj_name2id(m, mjOBJ_SENSOR, name.c_str());
  if (id == -1) {
    std::cerr << "sensor \"" << name << "\" not found.\n";
    return nullptr;
  } else {
    return d->sensordata + m->sensor_adr[id];
  }
}

// get default residual parameter data using string
double DefaultParameterValue(const mjModel* model, std::string_view name) {
  int id =
      mj_name2id(model, mjOBJ_NUMERIC, absl::StrCat("residual_", name).c_str());
  if (id == -1) {
    mju_error_s("Parameter '%s' not found", std::string(name).c_str());
    return 0;
  }
  return model->numeric_data[model->numeric_adr[id]];
}

// get index to residual parameter data using string
int ParameterIndex(const mjModel* model, std::string_view name) {
  int id =
      mj_name2id(model, mjOBJ_NUMERIC, absl::StrCat("residual_", name).c_str());

  if (id == -1) {
    mju_error_s("Parameter '%s' not found", std::string(name).c_str());
  }

  int i;
  for (i = 0; i < model->nnumeric; i++) {
    const char* first_residual = mj_id2name(model, mjOBJ_NUMERIC, i);
    if (absl::StartsWith(first_residual, "residual_")) {
      break;
    }
  }
  return id - i;
}

double DefaultResidualSelection(const mjModel* m, int numeric_index) {
  // list selections are stored as ints, but numeric values are doubles.
  int64_t value = m->numeric_data[m->numeric_adr[numeric_index]];
  return *std::launder(reinterpret_cast<const double*>(&value));
}

int CostTermByName(const mjModel* m, const std::string& name) {
  int id = mj_name2id(m, mjOBJ_SENSOR, name.c_str());
  if (id == -1 || m->sensor_type[id] != mjSENS_USER) {
    std::cerr << "cost term \"" << name << "\" not found.\n";
    return -1;
  } else {
    return id;
  }
}

void CheckSensorDim(const mjModel* model, int residual_size) {
  int user_sensor_dim = 0;
  bool encountered_nonuser_sensor = false;
  for (int i = 0; i < model->nsensor; i++) {
    if (model->sensor_type[i] == mjSENS_USER) {
      user_sensor_dim += model->sensor_dim[i];
      if (encountered_nonuser_sensor) {
        mju_error("user type sensors must come before other sensor types");
      }
    } else {
      encountered_nonuser_sensor = true;
    }
  }
  if (user_sensor_dim != residual_size) {
    mju_error(
        "mismatch between total user-sensor dimension %d "
        "and residual size %d",
        user_sensor_dim, residual_size);
  }
}

// get traces from sensors
void GetTraces(double* traces, const mjModel* m, const mjData* d,
               int num_trace) {
  if (num_trace > kMaxTraces) {
    mju_error("Number of traces should be less than 100\n");
  }
  if (num_trace == 0) {
    return;
  }

  // allocate string
  char str[7];
  for (int i = 0; i < num_trace; i++) {
    // set trace id
    mujoco::util_mjpc::sprintf_arr(str, "trace%i", i);
    double* trace = SensorByName(m, d, str);
    if (trace) mju_copy(traces + 3 * i, trace, 3);
  }
}

// get keyframe `qpos` data using string
double* KeyQPosByName(const mjModel* m, const mjData* d,
                      const std::string& name) {
  int id = mj_name2id(m, mjOBJ_KEY, name.c_str());
  if (id == -1) {
    return nullptr;
  }
  return m->key_qpos + m->nq * id;
}

void LinearRange(double* t, double t_step, double t0, int N) {
  for (int i = 0; i < N; i++) {
    t[i] = t0 + i * t_step;
  }
}

// zero-order interpolation
void ZeroInterpolation(double* output, double x, const std::vector<double>& xs,
                       const double* ys, int dim, int length) {
  // bounds
  int bounds[2];
  FindInterval(bounds, xs, x, length);

  // zero-order hold
  mju_copy(output, ys + dim * bounds[0], dim);
}

// linear interpolation
void LinearInterpolation(double* output, double x,
                         const std::vector<double>& xs, const double* ys,
                         int dim, int length) {
  // bounds
  int bounds[2];
  FindInterval(bounds, xs, x, length);

  // bound
  if (bounds[0] == bounds[1]) {
    mju_copy(output, ys + dim * bounds[0], dim);
    return;
  }

  // time normalization
  double t = (x - xs[bounds[0]]) / (xs[bounds[1]] - xs[bounds[0]]);

  // interpolation
  mju_scl(output, ys + dim * bounds[0], 1.0 - t, dim);
  mju_addScl(output, output, ys + dim * bounds[1], t, dim);
}

// coefficients for cubic interpolation
void CubicCoefficients(double* coefficients, double x,
                       const std::vector<double>& xs, int T) {
  // find interval
  int bounds[2];
  FindInterval(bounds, xs, x, T);

  // boundary
  if (bounds[0] == bounds[1]) {
    coefficients[0] = 1.0;
    coefficients[1] = 0.0;
    coefficients[2] = 0.0;
    coefficients[3] = 0.0;
    return;
  }
  // scaled interpolation point
  double t = (x - xs[bounds[0]]) / (xs[bounds[1]] - xs[bounds[0]]);

  // coefficients
  coefficients[0] = 2.0 * t * t * t - 3.0 * t * t + 1.0;
  coefficients[1] =
      (t * t * t - 2.0 * t * t + t) * (xs[bounds[1]] - xs[bounds[0]]);
  coefficients[2] = -2.0 * t * t * t + 3 * t * t;
  coefficients[3] = (t * t * t - t * t) * (xs[bounds[1]] - xs[bounds[0]]);
}

// finite-difference vector
double FiniteDifferenceSlope(double x, const std::vector<double>& xs,
                             const double* ys, int dim, int length, int i) {
  // find interval
  int bounds[2];
  FindInterval(bounds, xs, x, length);
  // lower out of bounds
  if (bounds[0] == 0 && bounds[1] == 0) {
    if (length > 2) {
      return (ys[dim * (bounds[1] + 1) + i] - ys[dim * bounds[1] + i]) /
             (xs[bounds[1] + 1] - xs[bounds[1]]);
    } else {
      return 0.0;
    }
    // upper out of bounds
  } else if (bounds[0] == length - 1 && bounds[1] == length - 1) {
    if (length > 2) {
      return (ys[dim * bounds[0] + i] - ys[dim * (bounds[0] - 1) + i]) /
             (xs[bounds[0]] - xs[bounds[0] - 1]);
    } else {
      return 0.0;
    }
    // bounds
  } else if (bounds[0] == 0) {
    return (ys[dim * bounds[1] + i] - ys[dim * bounds[0] + i]) /
           (xs[bounds[1]] - xs[bounds[0]]);
    // interval
  } else {
    return 0.5 * (ys[dim * bounds[1] + i] - ys[dim * bounds[0] + i]) /
               (xs[bounds[1]] - xs[bounds[0]]) +
           0.5 * (ys[dim * bounds[0] + i] - ys[dim * (bounds[0] - 1) + i]) /
               (xs[bounds[0]] - xs[bounds[0] - 1]);
  }
}

// cubic polynominal interpolation
void CubicInterpolation(double* output, double x, const std::vector<double>& xs,
                        const double* ys, int dim, int length) {
  // find interval
  int bounds[2];
  FindInterval(bounds, xs, x, length);
  // bound
  if (bounds[0] == bounds[1]) {
    mju_copy(output, ys + dim * bounds[0], dim);
    return;
  }
  // coefficients
  double coefficients[4];
  CubicCoefficients(coefficients, x, xs, length);
  // interval
  for (int i = 0; i < dim; i++) {
    // points and slopes
    double p0 = ys[bounds[0] * dim + i];
    double m0 = FiniteDifferenceSlope(xs[bounds[0]], xs, ys, dim, length, i);
    double m1 = FiniteDifferenceSlope(xs[bounds[1]], xs, ys, dim, length, i);
    double p1 = ys[bounds[1] * dim + i];
    // polynominal
    output[i] = coefficients[0] * p0 + coefficients[1] * m0 +
                coefficients[2] * p1 + coefficients[3] * m1;
  }
}

// returns the path to the directory containing the current executable
std::string GetExecutableDir() {
#if defined(_WIN32) || defined(__CYGWIN__)
  constexpr char kPathSep = '\\';
  std::string realpath = [&]() -> std::string {
    std::unique_ptr<char[]> realpath(nullptr);
    DWORD buf_size = 128;
    bool success = false;
    while (!success) {
      realpath.reset(new (std::nothrow) char[buf_size]);
      if (!realpath) {
        std::cerr << "cannot allocate memory to store executable path\n";
        return "";
      }

      DWORD written = GetModuleFileNameA(nullptr, realpath.get(), buf_size);
      if (written < buf_size) {
        success = true;
      } else if (written == buf_size) {
        // realpath is too small, grow and retry
        buf_size *= 2;
      } else {
        std::cerr << "failed to retrieve executable path: " << GetLastError()
                  << "\n";
        return "";
      }
    }
    return realpath.get();
  }();
#else
  constexpr char kPathSep = '/';
#if defined(__APPLE__)
  std::unique_ptr<char[]> buf(nullptr);
  {
    std::uint32_t buf_size = 0;
    _NSGetExecutablePath(nullptr, &buf_size);
    buf.reset(new char[buf_size]);
    if (!buf) {
      std::cerr << "cannot allocate memory to store executable path\n";
      return "";
    }
    if (_NSGetExecutablePath(buf.get(), &buf_size)) {
      std::cerr << "unexpected error from _NSGetExecutablePath\n";
    }
  }
  const char* path = buf.get();
#else
  const char* path = "/proc/self/exe";
#endif
  std::string realpath = [&]() -> std::string {
    std::unique_ptr<char[]> realpath(nullptr);
    std::uint32_t buf_size = 128;
    bool success = false;
    while (!success) {
      realpath.reset(new (std::nothrow) char[buf_size]);
      if (!realpath) {
        std::cerr << "cannot allocate memory to store executable path\n";
        return "";
      }

      std::size_t written = readlink(path, realpath.get(), buf_size);
      if (written < buf_size) {
        realpath.get()[written] = '\0';
        success = true;
      } else if (written == -1) {
        if (errno == EINVAL) {
          // path is already not a symlink, just use it
          return path;
        }

        std::cerr << "error while resolving executable path: "
                  << std::strerror(errno) << '\n';
        return "";
      } else {
        // realpath is too small, grow and retry
        buf_size *= 2;
      }
    }
    return realpath.get();
  }();
#endif

  if (realpath.empty()) {
    return "";
  }

  for (std::size_t i = realpath.size() - 1; i > 0; --i) {
    if (realpath.c_str()[i] == kPathSep) {
      return realpath.substr(0, i);
    }
  }

  // don't scan through the entire file system's root
  return "";
}

// Returns the directory where tasks are stored
static std::string GetTasksDir() {
  const char* tasks_dir = std::getenv("MJPC_TASKS_DIR");
  if (tasks_dir) {
    return tasks_dir;
  }
  return absl::StrCat(GetExecutableDir(), "/../mjpc/tasks");
}

// convenience function for paths
std::string GetModelPath(std::string_view path) {
  return absl::StrCat(GetTasksDir(), "/", path);
}

// dx = (x2 - x1) / h
void Diff(mjtNum* dx, const mjtNum* x1, const mjtNum* x2, mjtNum h, int n) {
  mjtNum inv_h = 1 / h;
  for (int i = 0; i < n; i++) {
    dx[i] = inv_h * (x2[i] - x1[i]);
  }
}

// finite-difference two state vectors ds = (s2 - s1) / h
void StateDiff(const mjModel* m, mjtNum* ds, const mjtNum* s1, const mjtNum* s2,
               mjtNum h) {
  int nq = m->nq, nv = m->nv, na = m->na;

  if (nq == nv) {
    Diff(ds, s1, s2, h, nq + nv + na);
  } else {
    mj_differentiatePos(m, ds, h, s1, s2);
    Diff(ds + nv, s1 + nq, s2 + nq, h, nv + na);
  }
}

// return global height of nearest group 0 geom under given position
mjtNum Ground(const mjModel* model, const mjData* data, const mjtNum pos[3],
              const mjtByte* geomgroup) {
  mjtNum down[3] = {0, 0, -1};                      // aim ray straight down
  const mjtNum height_offset = .5;  // add some height in case of penetration
  const mjtByte flg_static = 1;     // include static geoms
  const int bodyexclude = -1;       // don't exclude any bodies
  int geomid;                       // id of intersecting geom
  mjtNum query[3] = {pos[0], pos[1], pos[2] + height_offset};
  const mjtByte default_geomgroup[6] = {1, 0, 0, 0, 0, 0};
  const mjtByte* query_geomgroup = geomgroup ? geomgroup : default_geomgroup;
  mjtNum dist = mj_ray(model, data, query, down, query_geomgroup, flg_static,
                       bodyexclude, &geomid);

  if (dist < 0) {  // SHOULD NOT OCCUR
    mju_error("no group 0 geom detected by raycast");
  }

  return pos[2] + height_offset - dist;
}

// find frame that best matches 4 feet, z points to body
void FootFrame(double feet_pos[3], double feet_mat[9], double feet_quat[4],
               const double body[3], const double foot0[3],
               const double foot1[3], const double foot2[3],
               const double foot3[3]) {
  // average foot pos
  double pos[3];
  for (int i = 0; i < 3; i++) {
    pos[i] = 0.25 * (foot0[i] + foot1[i] + foot2[i] + foot3[i]);
  }

  // compute feet covariance
  double cov[9] = {0};
  for (const double* foot : {foot0, foot1, foot2, foot3}) {
    double dif[3], difTdif[9];
    mju_sub3(dif, foot, pos);
    mju_sqrMatTD(difTdif, dif, nullptr, 1, 3);
    mju_addTo(cov, difTdif, 9);
  }

  // eigendecompose
  double eigval[3], quat[4], mat[9];
  mju_eig3(eigval, mat, quat, cov);

  // make sure foot-plane normal (z axis) points to body
  double zaxis[3] = {mat[2], mat[5], mat[8]};
  double to_body[3];
  mju_sub3(to_body, body, pos);
  if (mju_dot3(zaxis, to_body) < 0) {
    // flip both z and y (rotate around x), to maintain frame handedness
    for (const int i : {1, 2, 4, 5, 7, 8}) mat[i] *= -1;
  }

  // copy outputs
  if (feet_pos) mju_copy3(feet_pos, pos);
  if (feet_mat) mju_copy(feet_mat, mat, 9);
  if (feet_quat) mju_mat2Quat(feet_quat, mat);
}

// set x to be the point on the segment [p0 p1] that is nearest to x
void ProjectToSegment(double x[3], const double p0[3], const double p1[3]) {
  double axis[3];
  mju_sub3(axis, p1, p0);

  double half_length = mju_normalize3(axis) / 2;

  double center[3];
  mju_add3(center, p0, p1);
  mju_scl3(center, center, 0.5);

  double center_to_x[3];
  mju_sub3(center_to_x, x, center);

  // project and clamp
  double t = mju_dot3(center_to_x, axis);
  t = mju_max(-half_length, mju_min(half_length, t));

  // find point
  double offset[3];
  mju_scl3(offset, axis, t);
  mju_add3(x, center, offset);
}

// default cost colors
const float CostColors[20][3]{
    {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.3, 0.3, 1.0}, {1.0, 1.0, 0.0},
    {0.0, 1.0, 1.0}, {1.0, 0.5, 0.5}, {0.5, 1.0, 0.5}, {0.5, 0.5, 1.0},
    {1.0, 1.0, 0.5}, {0.5, 1.0, 1.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0},
    {0.3, 0.3, 1.0}, {1.0, 1.0, 0.0}, {0.0, 1.0, 1.0}, {1.0, 0.5, 0.5},
    {0.5, 1.0, 0.5}, {0.5, 0.5, 1.0}, {1.0, 1.0, 0.5}, {0.5, 1.0, 1.0},
};

// plots - vertical line
void PlotVertical(mjvFigure* fig, double time, double min_value,
                  double max_value, int N, int index) {
  for (int i = 0; i < N; i++) {
    fig->linedata[index][2 * i] = time;
    fig->linedata[index][2 * i + 1] =
        min_value + i / (N - 1) * (max_value - min_value);
  }
  fig->linepnt[index] = N;
}

// plots - update data
void PlotUpdateData(mjvFigure* fig, double* bounds, double x, double y,
                    int length, int index, int x_update, int y_update,
                    double x_bound_lower) {
  int pnt = mjMIN(length, fig->linepnt[index] + 1);

  // shift previous data
  for (int i = pnt - 1; i > 0; i--) {
    if (x_update) {
      fig->linedata[index][2 * i] = fig->linedata[index][2 * i - 2];
    }
    if (y_update) {
      fig->linedata[index][2 * i + 1] = fig->linedata[index][2 * i - 1];
    }

    // bounds
    if (fig->linedata[index][2 * i] > x_bound_lower) {
      if (fig->linedata[index][2 * i + 1] < bounds[0]) {
        bounds[0] = fig->linedata[index][2 * i + 1];
      }
      if (fig->linedata[index][2 * i + 1] > bounds[1]) {
        bounds[1] = fig->linedata[index][2 * i + 1];
      }
    }
  }

  // current data
  fig->linedata[index][0] = x;
  fig->linedata[index][1] = y;
  fig->linepnt[index] = pnt;
}

// reset plot data to zeros
void PlotResetData(mjvFigure* fig, int length, int index) {
  if (index >= mjMAXLINE) {
    std::cerr << "Too many plots requested: " << index << '\n';
    return;
  }
  int pnt = mjMIN(length, fig->linepnt[index] + 1);

  // shift previous data
  for (int i = pnt - 1; i > 0; i--) {
    fig->linedata[index][2 * i] = 0.0;
    fig->linedata[index][2 * i + 1] = 0.0;
  }

  // current data
  fig->linedata[index][0] = 0.0;
  fig->linedata[index][1] = 0.0;
  fig->linepnt[index] = pnt;
}

// plots - horizontal line
void PlotHorizontal(mjvFigure* fig, const double* xs, double y, int length,
                    int index) {
  for (int i = 0; i < length; i++) {
    fig->linedata[index][2 * i] = xs[i];
    fig->linedata[index][2 * i + 1] = y;
  }
  fig->linepnt[index] = length;
}

// plots - set data
void PlotData(mjvFigure* fig, double* bounds, const double* xs,
              const double* ys, int dim, int dim_limit, int length,
              int start_index, double x_bound_lower) {
  for (int j = 0; j < dim_limit; j++) {
    for (int t = 0; t < length; t++) {
      // set data
      fig->linedata[start_index + j][2 * t] = xs[t];
      fig->linedata[start_index + j][2 * t + 1] = ys[t * dim + j];

      // check bounds if x > x_bound_lower
      if (fig->linedata[start_index + j][2 * t] > x_bound_lower) {
        if (fig->linedata[start_index + j][2 * t + 1] < bounds[0]) {
          bounds[0] = fig->linedata[start_index + j][2 * t + 1];
        }
        if (fig->linedata[start_index + j][2 * t + 1] > bounds[1]) {
          bounds[1] = fig->linedata[start_index + j][2 * t + 1];
        }
      }
    }
    fig->linepnt[start_index + j] = length;
  }
}

// add geom to scene
void AddGeom(mjvScene* scene, mjtGeom type, const mjtNum size[3],
             const mjtNum pos[3], const mjtNum mat[9], const float rgba[4]) {
  // if no available geoms, return
  if (scene->ngeom >= scene->maxgeom) return;

  // add geom
  mjtNum mat_world[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  mjv_initGeom(&scene->geoms[scene->ngeom], type, size, pos,
               mat ? mat : mat_world, rgba);
  scene->geoms[scene->ngeom].category = mjCAT_DECOR;

  // increment ngeom
  scene->ngeom += 1;
}

// add connector geom to scene
void AddConnector(mjvScene* scene, mjtGeom type, mjtNum width,
                  const mjtNum from[3], const mjtNum to[3],
                  const float rgba[4]) {
  // if no available geoms, return
  if (scene->ngeom >= scene->maxgeom) return;

  // make connector geom
  mjv_initGeom(&scene->geoms[scene->ngeom], type,
               /*size=*/nullptr, /*pos=*/nullptr, /*mat=*/nullptr, rgba);
  scene->geoms[scene->ngeom].category = mjCAT_DECOR;
  mjv_makeConnector(&scene->geoms[scene->ngeom], type, width, from[0], from[1],
                    from[2], to[0], to[1], to[2]);

  // increment ngeom
  scene->ngeom += 1;
}

// number of available hardware threads
#if defined(__APPLE__) || defined(_WIN32)
int NumAvailableHardwareThreads(void) {
  return std::thread::hardware_concurrency();
}
#else
int NumAvailableHardwareThreads(void) {
  // start by assuming a maximum of 128 hardware threads and keep growing until
  // the cpu_set_t is big enough to hold the mask for the entire machine
  for (int max_count = 128; true; max_count *= 2) {
    std::unique_ptr<cpu_set_t, void (*)(cpu_set_t*)> set(
        CPU_ALLOC(max_count), +[](cpu_set_t* set) { CPU_FREE(set); });
    size_t setsize = CPU_ALLOC_SIZE(max_count);
    int result = sched_getaffinity(getpid(), setsize, set.get());
    if (result == 0) {
      // success
      return CPU_COUNT_S(setsize, set.get());
    } else if (errno != EINVAL) {
      // failure other than max_count being too small, just return 1
      return 1;
    }
  }
}
#endif

// check mjData for warnings, return true if any warnings
bool CheckWarnings(mjData* data) {
  bool warnings_found = false;
  for (int i = 0; i < mjNWARNING; i++) {
    if (data->warning[i].number > 0) {
      // reset
      data->warning[i].number = 0;

      // return failure
      warnings_found = true;
    }
  }
  return warnings_found;
}

// compute vector with log-based scaling between min and max values
void LogScale(double* values, double max_value, double min_value, int steps) {
  double step =
      (std::log(max_value) - std::log(min_value)) / std::max((steps - 1), 1);
  for (int i = 0; i < steps; i++) {
    values[i] = std::exp(std::log(min_value) + i * step);
  }
}

// ============== 2d convex hull ==============

// note: written in MuJoCo-style C for possible future inclusion
namespace {  // private functions in an anonymous namespace

// 2d vector dot-product
mjtNum mju_dot2(const mjtNum vec1[2], const mjtNum vec2[2]) {
  return vec1[0] * vec2[0] + vec1[1] * vec2[1];
}

// 2d vector squared distance
mjtNum mju_sqrdist2(const mjtNum vec1[2], const mjtNum vec2[2]) {
  const mjtNum diff[2] = {vec1[0] - vec2[0], vec1[1] - vec2[1]};
  return mju_dot2(diff, diff);
}

// returns true if edge to candidate is to the right of edge to next
bool IsEdgeOutside(const mjtNum current[2], const mjtNum next[2],
                   const mjtNum candidate[2]) {
  mjtNum current_edge[2] = {next[0] - current[0], next[1] - current[1]};
  mjtNum candidate_edge[2] = {candidate[0] - current[0],
                              candidate[1] - current[1]};
  mjtNum rotated_edge[2] = {current_edge[1], -current_edge[0]};
  mjtNum projection = mju_dot2(candidate_edge, rotated_edge);

  // check if candidate edge is to the right
  if (projection > mjMINVAL) {
    // actually to the right: accept
    return true;
  } else if (abs(projection) < mjMINVAL) {
    // numerically equivalent: accept if longer
    mjtNum current_length2 = mju_dot2(current_edge, current_edge);
    mjtNum candidate_length2 = mju_dot2(candidate_edge, candidate_edge);
    return (candidate_length2 > current_length2);
  }
  // not to the right
  return false;
}

// returns 2D point on line segment from v0 to v1 that is nearest to query point
void ProjectToSegment2D(mjtNum res[2], const mjtNum query[2],
                        const mjtNum v0[2], const mjtNum v1[2]) {
  mjtNum axis[2] = {v1[0] - v0[0], v1[1] - v0[1]};
  mjtNum length = mju_sqrt(mju_dot2(axis, axis));
  axis[0] /= length;
  axis[1] /= length;
  mjtNum center[2] = {0.5 * (v1[0] + v0[0]), 0.5 * (v1[1] + v0[1])};
  mjtNum t = mju_dot2(query, axis) - mju_dot2(center, axis);
  t = mju_clip(t, -length / 2, length / 2);
  res[0] = center[0] + t * axis[0];
  res[1] = center[1] + t * axis[1];
}

}  // namespace

// returns point in 2D convex hull that is nearest to query
void NearestInHull(mjtNum res[2], const mjtNum query[2], const mjtNum* points,
                   const int* hull, int num_hull) {
  int outside = 0;      // assume query point is inside the hull
  mjtNum best_sqrdist;  // smallest squared distance so far
  for (int i = 0; i < num_hull; i++) {
    const mjtNum* v0 = points + 2 * hull[i];
    const mjtNum* v1 = points + 2 * hull[(i + 1) % num_hull];

    // edge from v0 to v1
    mjtNum e01[2] = {v1[0] - v0[0], v1[1] - v0[1]};

    // normal to the edge, pointing *into* the convex hull
    mjtNum n01[2] = {-e01[1], e01[0]};

    // if constraint is active, project to the edge, compare to best so far
    mjtNum v0_to_query[2] = {query[0] - v0[0], query[1] - v0[1]};
    if (mju_dot(v0_to_query, n01, 2) < 0) {
      mjtNum projection[2];
      ProjectToSegment2D(projection, query, v0, v1);
      mjtNum sqrdist = mju_sqrdist2(projection, query);
      if (!outside || (outside && sqrdist < best_sqrdist)) {
        // first or closer candidate, copy to res
        res[0] = projection[0];
        res[1] = projection[1];
        best_sqrdist = sqrdist;
      }
      outside = 1;
    }
  }

  // not outside any edge, return the query point
  if (!outside) {
    res[0] = query[0];
    res[1] = query[1];
  }
}

// find the convex hull of a small set of 2D points
int Hull2D(int* hull, int num_points, const mjtNum* points) {
  // handle small number of points
  if (num_points < 1) return 0;
  hull[0] = 0;
  if (num_points == 1) return 1;
  if (num_points == 2) {
    hull[1] = 1;
    return 2;
  }

  // find the point with largest x value - must lie on hull
  mjtNum best_x = points[0];
  mjtNum best_y = points[1];
  for (int i = 1; i < num_points; i++) {
    mjtNum x = points[2 * i];
    mjtNum y = points[2 * i + 1];

    // accept if larger, use y value to tie-break exact equality
    if (x > best_x || (x == best_x && y > best_y)) {
      best_x = x;
      best_y = y;
      hull[0] = i;
    }
  }

  //  Gift-wrapping algorithm takes time O(nh)
  // TODO(benmoran) Investigate faster convex hull methods.
  int num_hull = 1;
  for (int i = 0; i < num_points; i++) {
    // loop over all points, find point that is furthest outside
    int next = -1;
    for (int candidate = 0; candidate < num_points; candidate++) {
      if ((next == -1) ||
          IsEdgeOutside(points + 2 * hull[num_hull - 1], points + 2 * next,
                        points + 2 * candidate)) {
        next = candidate;
      }
    }

    // termination condition
    if ((num_hull > 1) && (next == hull[0])) {
      break;
    }

    // add new point
    hull[num_hull++] = next;
  }

  return num_hull;
}

// finite-difference gradient constructor
FiniteDifferenceGradient::FiniteDifferenceGradient(int dim) { Resize(dim); }

// finite-difference gradient resize memory
void FiniteDifferenceGradient::Resize(int dim) {
  // allocate memory
  if (dim != gradient.size()) gradient.resize(dim);
  if (dim != workspace_.size()) workspace_.resize(dim);
}

// compute finite-difference gradient
void FiniteDifferenceGradient::Compute(
    std::function<double(const double* x)> func, const double* input, int dim) {
  // resize
  Resize(dim);

  // ----- compute ----- //
  // set workspace
  mju_copy(workspace_.data(), input, dim);

  // finite difference
  for (int i = 0; i < dim; i++) {
    // positive perturbation
    workspace_[i] += 0.5 * epsilon;
    double fp = func(workspace_.data());

    // negative
    workspace_[i] -= 1.0 * epsilon;
    double fn = func(workspace_.data());

    // gradient
    gradient[i] = (fp - fn) / epsilon;

    // reset
    workspace_[i] = input[i];
  }
}

// finite-difference Jacobian constructor
FiniteDifferenceJacobian::FiniteDifferenceJacobian(int num_output,
                                                   int num_input) {
  // resize
  Resize(num_output, num_input);
}

// finite-difference Jacobian memory resize
void FiniteDifferenceJacobian::Resize(int num_output, int num_input) {
  // resize
  if (jacobian.size() != num_output * num_input)
    jacobian.resize(num_output * num_input);
  if (jacobian_transpose.size() != num_output * num_input)
    jacobian_transpose.resize(num_output * num_input);
  if (output.size() != num_output) output.resize(num_output);
  if (output_nominal.size() != num_output) output_nominal.resize(num_output);
  if (workspace_.size() != num_input) workspace_.resize(num_input);
}

// compute Jacobian
void FiniteDifferenceJacobian::Compute(
    std::function<void(double* output, const double* input)> func,
    const double* input, int num_output, int num_input) {
  // resize
  Resize(num_output, num_input);

  // copy workspace
  mju_copy(workspace_.data(), input, num_input);

  // nominal evaluation
  mju_zero(output_nominal.data(), num_output);
  func(output_nominal.data(), workspace_.data());

  for (int i = 0; i < num_input; i++) {
    // perturb input
    workspace_[i] += epsilon;

    // evaluate
    mju_zero(output.data(), num_output);
    func(output.data(), workspace_.data());

    // Jacobian
    double* JT = jacobian_transpose.data() + i * num_output;
    mju_sub(JT, output.data(), output_nominal.data(), num_output);
    mju_scl(JT, JT, 1.0 / epsilon, num_output);

    // reset workspace
    workspace_[i] = input[i];
  }

  // transpose
  mju_transpose(jacobian.data(), jacobian_transpose.data(), num_input,
                num_output);
}

// finite-difference Hessian constructor
FiniteDifferenceHessian::FiniteDifferenceHessian(int dim) {
  // resize memory
  Resize(dim);
}

// finite-difference Hessian memory resize
void FiniteDifferenceHessian::Resize(int dim) {
  // resize
  if (dim * dim != hessian.size()) hessian.resize(dim * dim);
  if (dim != workspace1_.size()) workspace1_.resize(dim);
  if (dim != workspace2_.size()) workspace2_.resize(dim);
  if (dim != workspace3_.size()) workspace3_.resize(dim);
}

// compute finite-difference Hessian
void FiniteDifferenceHessian::Compute(
    std::function<double(const double* x)> func, const double* input, int dim) {
  // resize
  Resize(dim);

  // set workspace
  mju_copy(workspace1_.data(), input, dim);
  mju_copy(workspace2_.data(), input, dim);
  mju_copy(workspace3_.data(), input, dim);

  // evaluate at candidate
  double f = func(input);

  // centered finite-difference
  for (int i = 0; i < dim; i++) {
    for (int j = i; j < dim; j++) {  // skip bottom triangle
      // workspace 1
      workspace1_[i] += epsilon;
      workspace1_[j] += epsilon;

      double fij = func(workspace1_.data());

      // workspace 2
      workspace2_[i] += epsilon;
      double fi = func(workspace2_.data());

      // workspace 3
      workspace3_[j] += epsilon;
      double fj = func(workspace3_.data());

      // Hessian value
      double H = (fij - fi - fj + f) / (epsilon * epsilon);
      hessian[i * dim + j] = H;
      hessian[j * dim + i] = H;

      // reset workspace 1
      workspace1_[i] = input[i];
      workspace1_[j] = input[j];

      // reset workspace 2
      workspace2_[i] = input[i];

      // reset workspace 3
      workspace3_[j] = input[j];
    }
  }
}

// set scaled block (size: rb x cb) in mat (size: rm x cm) given mat upper row
// and left column indices (ri, ci)
void SetBlockInMatrix(double* mat, const double* block, double scale, int rm,
                      int cm, int rb, int cb, int ri, int ci) {
  // loop over block rows
  for (int i = 0; i < rb; i++) {
    // loop over block columns
    for (int j = 0; j < cb; j++) {
      mat[(ri + i) * cm + ci + j] = scale * block[i * cb + j];
    }
  }
}

// set scaled block (size: rb x cb) in mat (size: rm x cm) given mat upper row
// and left column indices (ri, ci)
void AddBlockInMatrix(double* mat, const double* block, double scale, int rm,
                      int cm, int rb, int cb, int ri, int ci) {
  // loop over block rows
  for (int i = 0; i < rb; i++) {
    // loop over block columns
    for (int j = 0; j < cb; j++) {
      mat[(ri + i) * cm + ci + j] += scale * block[i * cb + j];
    }
  }
}

// get block (size: rb x cb) from mat (size: rm x cm) given mat upper row
// and left column indices (ri, ci)
void BlockFromMatrix(double* block, const double* mat, int rb, int cb, int rm,
                     int cm, int ri, int ci) {
  // loop over block rows
  for (int i = 0; i < rb; i++) {
    double* block_cols = block + i * cb;
    const double* mat_cols = mat + (ri + i) * cm + ci;
    mju_copy(block_cols, mat_cols, cb);
  }
}

// differentiate mju_subQuat wrt qa, qb
void DifferentiateSubQuat(double jaca[9], double jacb[9], const double qa[4],
                          const double qb[4]) {
  // compute 3D velocity
  double axis[3];
  mju_subQuat(axis, qa, qb);

  // angle + normalize axis
  double angle = mju_normalize3(axis);
  double th2 = 0.5 * angle;

  // coefficients
  double c0 = th2;
  double c1 = 1.0 - (mju_abs(th2) < 6e-8 ? 1.0 : th2 / mju_tan(th2));

  // compute Jacobian
  double jac[9];
  jac[0] = 1.0 + c1 * (-axis[1] * axis[1] - axis[2] * axis[2]);
  jac[1] = c1 * axis[0] * axis[1] - c0 * axis[2];
  jac[2] = c0 * axis[1] + c1 * axis[0] * axis[2];
  jac[3] = c0 * axis[2] + c1 * axis[0] * axis[1];
  jac[4] = 1.0 + c1 * (-axis[0] * axis[0] - axis[2] * axis[2]);
  jac[5] = c1 * axis[1] * axis[2] - c0 * axis[0];
  jac[6] = c1 * axis[0] * axis[2] - c0 * axis[1];
  jac[7] = c0 * axis[0] + c1 * axis[1] * axis[2];
  jac[8] = 1.0 + c1 * (-axis[0] * axis[0] - axis[1] * axis[1]);

  // Jacobian wrt qa
  if (jaca) {
    mju_copy(jaca, jac, 9);
  }

  // Jacobian wrt qb
  if (jacb) {
    // jacb = -jaca^T
    mju_transpose(jacb, jac, 3, 3);
    mju_scl(jacb, jacb, -1.0, 9);
  }
}

// differentiate velocity by finite-differencing two positions wrt to qpos1,
// qpos2
void DifferentiateDifferentiatePos(double* jac1, double* jac2,
                                   const mjModel* model, double dt,
                                   const double* qpos1, const double* qpos2) {
  // mjtNum neg[4], dif[4];

  // zero Jacobians
  if (jac1) mju_zero(jac1, model->nv * model->nv);
  if (jac2) mju_zero(jac2, model->nv * model->nv);

  // loop over joints
  for (int j = 0; j < model->njnt; j++) {
    // get addresses in qpos and qvel
    int padr = model->jnt_qposadr[j];
    int vadr = model->jnt_dofadr[j];

    switch (model->jnt_type[j]) {
      case mjJNT_FREE:
        for (int i = 0; i < 3; i++) {
          // qvel[vadr + i] = (qpos2[padr + i] - qpos1[padr + i]) / dt;
          if (jac1) jac1[(vadr + i) * model->nv + vadr + i] = -1.0 / dt;
          if (jac2) jac2[(vadr + i) * model->nv + vadr + i] = 1.0 / dt;
        }
        vadr += 3;
        padr += 3;

        // continute with rotations
        [[fallthrough]];

      case mjJNT_BALL:
        // mju_negQuat(neg, qpos1 + padr);  // solve:  qpos1 * dif = qpos2
        // mju_mulQuat(dif, neg, qpos2 + padr);
        // mju_quat2Vel(qvel + vadr, dif, dt);

        // NOTE: order swap for qpos1, qpos2
        if (jac1) {
          // compute subQuat Jacobian block
          double jac1_blk[9];
          DifferentiateSubQuat(NULL, jac1_blk, qpos2 + padr, qpos1 + padr);

          // set block in Jacobian
          SetBlockInMatrix(jac1, jac1_blk, 1.0 / dt, model->nv, model->nv, 3, 3,
                           vadr, vadr);
        }

        if (jac2) {
          // compute subQuat Jacobian block
          double jac2_blk[9];
          DifferentiateSubQuat(jac2_blk, NULL, qpos2 + padr, qpos1 + padr);

          // set block in Jacobian
          SetBlockInMatrix(jac2, jac2_blk, 1.0 / dt, model->nv, model->nv, 3, 3,
                           vadr, vadr);
        }

        break;

      case mjJNT_HINGE:
      case mjJNT_SLIDE:
        // qvel[vadr] = (qpos2[padr] - qpos1[padr]) / dt;
        if (jac1) jac1[vadr * model->nv + vadr] = -1.0 / dt;
        if (jac2) jac2[vadr * model->nv + vadr] = 1.0 / dt;
    }
  }
}

// compute number of nonzeros in band matrix
int BandMatrixNonZeros(int ntotal, int nband) {
  // no band
  if (nband == 0) return 0;

  // diagonal matrix
  if (nband == 1) return ntotal;

  // initialize number of nonzeros
  int nnz = 0;

  // diagonal
  nnz += ntotal;

  // off diagonals
  for (int k = 1; k < nband; k++) {
    nnz += 2 * (ntotal - k);
  }

  // total non-zeros
  return nnz;
}

// get duration since time point
double GetDuration(std::chrono::steady_clock::time_point time) {
  return std::chrono::duration_cast<std::chrono::microseconds>(
             std::chrono::steady_clock::now() - time)
      .count();
}

// copy symmetric band matrix block by block
void SymmetricBandMatrixCopy(double* res, const double* mat, int dblock,
                             int nblock, int ntotal, int num_blocks,
                             int res_start_row, int res_start_col,
                             int mat_start_row, int mat_start_col,
                             double* scratch) {
  // check for no blocks to copy
  if (num_blocks == 0) return;

  // tmp: block from mat
  double* tmp1 = scratch;
  double* tmp2 = scratch + dblock * dblock;

  // loop over upper band
  for (int i = 0; i < num_blocks; i++) {
    // number of columns to loop over for row
    int num_cols = mju_min(nblock, num_blocks - i);

    for (int j = i; j < i + num_cols; j++) {
      // get block from A
      BlockFromMatrix(tmp1, mat, dblock, dblock, ntotal, ntotal,
                      (i + mat_start_row) * dblock,
                      (j + mat_start_col) * dblock);

      // set block in matrix
      AddBlockInMatrix(res, tmp1, 1.0, ntotal, ntotal, dblock, dblock,
                       (i + res_start_row) * dblock,
                       (j + res_start_col) * dblock);

      if (j > i) {
        mju_transpose(tmp2, tmp1, dblock, dblock);
        AddBlockInMatrix(res, tmp2, 1.0, ntotal, ntotal, dblock, dblock,
                         (j + res_start_col) * dblock,
                         (i + res_start_row) * dblock);
      }
    }
  }
}

// zero block (size: rb x cb) in mat (size: rm x cm) given mat upper row
// and left column indices (ri, ci)
void ZeroBlockInMatrix(double* mat, int rm, int cm, int rb, int cb, int ri,
                       int ci) {
  // loop over block rows
  for (int i = 0; i < rb; i++) {
    // loop over block columns
    for (int j = 0; j < cb; j++) {
      mat[(ri + i) * cm + ci + j] = 0.0;
    }
  }
}

// square dense to block band matrix
void DenseToBlockBand(double* res, int dim, int dblock, int nblock) {
  // number of block rows / columns
  int num_blocks = dim / dblock;

  // zero off-band blocks
  for (int i = 0; i < num_blocks; i++) {
    for (int j = i + nblock; j < num_blocks; j++) {
      ZeroBlockInMatrix(res, dim, dim, dblock, dblock, i * dblock, j * dblock);
      if (j > i)
        ZeroBlockInMatrix(res, dim, dim, dblock, dblock, j * dblock,
                          i * dblock);
    }
  }
}

// trace of square matrix
double Trace(const double* mat, int n) {
  // initialize
  double trace = 0.0;

  // sum diagonal terms
  for (int i = 0; i < n; i++) {
    trace += mat[n * i + i];
  }

  return trace;
}

// determinant of 3x3 matrix
double Determinant3(const double* mat) {
  // unpack
  double a = mat[0];
  double b = mat[1];
  double c = mat[2];
  double d = mat[3];
  double e = mat[4];
  double f = mat[5];
  double g = mat[6];
  double h = mat[7];
  double i = mat[8];

  // determinant
  return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
}

// inverse of 3x3 matrix
void Inverse3(double* res, const double* mat) {
  // unpack
  double a = mat[0];
  double b = mat[1];
  double c = mat[2];
  double d = mat[3];
  double e = mat[4];
  double f = mat[5];
  double g = mat[6];
  double h = mat[7];
  double i = mat[8];

  // determinant
  double det = Determinant3(mat);

  // inverse
  res[0] = e * i - f * h;
  res[1] = -(b * i - c * h);
  res[2] = b * f - c * e;
  res[3] = -(d * i - f * g);
  res[4] = a * i - c * g;
  res[5] = -(a * f - c * d);
  res[6] = d * h - e * g;
  res[7] = -(a * h - b * g);
  res[8] = a * e - b * d;

  // scale
  mju_scl(res, res, 1.0 / det, 9);
}

// condition matrix: res = mat11 - mat10 * mat00 \ mat10^T; return rank of
// mat00
// TODO(taylor): thread
void ConditionMatrix(double* res, const double* mat, double* mat00,
                     double* mat10, double* mat11, double* tmp0, double* tmp1,
                     int n, int n0, int n1, double* bandfactor, int nband) {
  // unpack mat
  BlockFromMatrix(mat00, mat, n0, n0, n, n, 0, 0);
  BlockFromMatrix(mat10, mat, n1, n0, n, n, n0, 0);
  BlockFromMatrix(mat11, mat, n1, n1, n, n, n0, n0);

  // factorize mat00, solve mat00 \ mat10^T
  if (nband > 0 && bandfactor) {
    mju_dense2Band(bandfactor, mat00, n0, nband, 0);

    // factorize mat00
    mju_cholFactorBand(bandfactor, n0, nband, 0, 0.0, 0.0);

    // tmp0 = mat00 \ mat01 = (mat00^-1 mat01)^T
    for (int i = 0; i < n1; i++) {
      mju_cholSolveBand(tmp0 + n0 * i, bandfactor, mat10 + n0 * i, n0, nband,
                        0);
    }
  } else {
    // factorize mat00
    mju_cholFactor(mat00, n0, 0.0);

    // tmp0 = mat00 \ mat01 = (mat00^-1 mat01)^T
    for (int i = 0; i < n1; i++) {
      mju_cholSolve(tmp0 + n0 * i, mat00, mat10 + n0 * i, n0);
    }
  }

  // tmp1 = mat10 * (mat00 \ mat01)
  mju_mulMatMatT(tmp1, tmp0, mat10, n1, n0, n1);

  // res = mat11 - mat10 * (mat00 \ mat01)
  mju_sub(res, mat11, tmp1, n1 * n1);
}

// principal eigenvector of 4x4 matrix
// QUEST algorithm from "Three-Axis Attitude Determination from Vector
// Observations"
void PrincipalEigenVector4(double* res, const double* mat,
                           double eigenvalue_init) {
  // Z = mat[0:3, 3]
  double Z[3] = {mat[3], mat[7], mat[11]};

  // S = mat[0:3, 0:3] + mat[3, 3] * I
  double S[9] = {mat[0] + mat[15], mat[1],           mat[2],
                 mat[4],           mat[5] + mat[15], mat[6],
                 mat[8],           mat[9],           mat[10] + mat[15]};

  // delta = det(S)
  double delta = Determinant3(S);

  // kappa = trace(delta * S^-1)
  double tmp0[9];
  Inverse3(tmp0, S);
  mju_scl(tmp0, tmp0, delta, 9);
  double kappa = Trace(tmp0, 3);

  // sigma = 0.5 * trace(S)
  double sigma = 0.5 * Trace(S, 3);
  double sigma2 = sigma * sigma;

  // S * Z
  double SZ[3];
  mju_mulMatVec(SZ, S, Z, 3, 3);

  // d = Z' * S * S * Z
  double d = mju_dot(SZ, SZ, 3);

  // c = delta + Z' * S * Z
  double c = delta + mju_dot(Z, SZ, 3);

  // b = sigma * sigma + Z' * Z
  double b = sigma2 + mju_dot(Z, Z, 3);

  // a = sigma * sigma - kappa
  double a = sigma2 - kappa;

  // -- find largest eigenvalue -- //

  // initialize
  double x = eigenvalue_init;

  // coefficients
  double ab = a + b;
  double bias = a * b + c * sigma - d;

  // iterate
  for (int i = 0; i < 10; i++) {
    // eigenvalue powers
    double x2 = x * x;
    double x3 = x2 * x;
    double x4 = x3 * x;

    // root
    double num = x4 - ab * x2 - c * x + bias;

    // root derivative
    double den = 4.0 * x3 - 2.0 * ab * x - c;

    // update
    x -= num / den;
  }

  // -- principal eigenvector -- //
  double alpha = x * x - sigma2 + kappa;
  double beta = x - sigma;
  double gamma = (x + sigma) * alpha - delta;

  // X = alpha * Z + beta * S * Z + S * S * Z
  double X[3];
  mju_mulMatVec(X, S, SZ, 3, 3);
  mju_addToScl(X, SZ, beta, 3);
  mju_addToScl(X, Z, alpha, 3);

  // scale
  double scl = 1.0 / mju_sqrt(gamma * gamma + mju_dot(X, X, 3));

  // eigenvector
  res[0] = scl * X[0];
  res[1] = scl * X[1];
  res[2] = scl * X[2];
  res[3] = scl * gamma;
}

// set scaled symmetric block matrix in band matrix
void SetBlockInBand(double* band, const double* block, double scale, int ntotal,
                    int nband, int nblock, int shift, int row_skip, bool add) {
  // loop over block rows
  for (int i = row_skip; i < nblock; i++) {
    // width of block lower triangle row
    int width = i + 1;

    // number of leading zeros in band row
    int column_shift = nband - width;

    // row segments
    double* band_row = band + (shift + row_skip + i) * nband + column_shift;
    const double* block_row = block + (row_skip + i) * nblock;

    // copy block row segment into band row
    if (add) {
      mju_addToScl(band_row, block_row, scale, width);
    } else {
      mju_copy(band_row, block_row, width);
      mju_scl(band_row, band_row, scale, width);
    }
  }
}

}  // namespace mjpc
