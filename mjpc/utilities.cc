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
#include <unordered_map>
#include <vector>

// DEEPMIND INTERNAL IMPORT
#include <absl/container/flat_hash_map.h>
#include <absl/strings/match.h>
#include <absl/strings/str_cat.h>
#include <absl/strings/str_split.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mujoco.h>
#include "mjpc/array_safety.h"

#if defined(__APPLE__) || defined(_WIN32)
#include <thread>
#else
#include <sched.h>
#endif

extern "C" {
#if defined(_WIN32) || defined(__CYGWIN__)
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
  for (int i = 0; i < m->ntextdata; i++) {
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
  return *reinterpret_cast<const int*>(&value);
}

double ReinterpretAsDouble(int64_t value) {
  return *reinterpret_cast<const double*>(&value);
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
  int id = mj_name2id(model, mjOBJ_NUMERIC,
                      absl::StrCat("residual_", name).c_str());
  if (id == -1) {
    mju_error_s("Parameter '%s' not found", std::string(name).c_str());
    return 0;
  }
  return model->numeric_data[model->numeric_adr[id]];
}

// get index to residual parameter data using string
int ParameterIndex(const mjModel* model, std::string_view name) {
  int id = mj_name2id(model, mjOBJ_NUMERIC,
                      absl::StrCat("residual_", name).c_str());

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
  return *reinterpret_cast<const double*>(&value);
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
  for (int i=0; i < model->nsensor; i++) {
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
    mju_error("mismatch between total user-sensor dimension %d "
              "and residual size %d", user_sensor_dim, residual_size);
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
  } else {
    return m->key_qpos + m->nq * id;
  }
}

// get keyframe `qvel` data using string
double* KeyQVelByName(const mjModel* m, const mjData* d,
                      const std::string& name) {
  int id = mj_name2id(m, mjOBJ_KEY, name.c_str());
  if (id == -1) {
    return nullptr;
  } else {
    return m->key_qvel + m->nv * id;
  }
}

// return a power transformed sequence
void PowerSequence(double* t, double t_step, double t1, double t2, double p,
                   double N) {
  // y = a * t^p + b
  double den = (mju_pow(t1, p) - mju_pow(t2, p));
  double a = (t1 - t2) / den;
  double b = (-t1 * mju_pow(t2, p) + t2 * mju_pow(t1, p)) / den;

  double t_running = t1;
  for (int i = 0; i < N; i++) {
    t[i] = a * mju_pow(t_running, p) + b;
    t_running += t_step;
  }
}

// find interval in monotonic sequence containing value
void FindInterval(int* bounds, const std::vector<double>& sequence,
                  double value, int length) {
  // get bounds
  auto it =
      std::upper_bound(sequence.begin(), sequence.begin() + length, value);
  int upper_bound = it - sequence.begin();
  int lower_bound = upper_bound - 1;

  // set bounds
  if (lower_bound < 0) {
    bounds[0] = 0;
    bounds[1] = 0;
  } else if (lower_bound > length - 1) {
    bounds[0] = length - 1;
    bounds[1] = length - 1;
  } else {
    bounds[0] = mju_max(lower_bound, 0);
    bounds[1] = mju_min(upper_bound, length - 1);
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
  return absl::StrCat(GetExecutableDir(), "/../tasks");
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
mjtNum Ground(const mjModel* model, const mjData* data, const mjtNum pos[3]) {
  const mjtByte geomgroup[6] = {1, 0, 0, 0, 0, 0};  // only detect group 0
  mjtNum down[3] = {0, 0, -1};      // aim ray straight down
  const mjtNum height_offset = .5;  // add some height in case of penetration
  const mjtByte flg_static = 1;     // include static geoms
  const int bodyexclude = -1;       // don't exclude any bodies
  int geomid;                       // id of intersecting geom
  mjtNum query[3] = {pos[0], pos[1], pos[2] + height_offset};
  mjtNum dist = mj_ray(model, data, query, down, geomgroup, flg_static,
                       bodyexclude, &geomid);

  if (dist < 0) {  // SHOULD NOT OCCUR
    mju_error("no group 0 geom detected by raycast");
  }

  return pos[2] + height_offset - dist;
}


// find frame that best matches 4 feet, z points to body
void FootFrame(double feet_pos[3], double feet_mat[9], double feet_quat[4],
               const double body[3],
               const double foot0[3], const double foot1[3],
               const double foot2[3], const double foot3[3]) {
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
    {1.0, 1.0, 0.5}, {0.5, 1.0, 1.0},
    {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.3, 0.3, 1.0}, {1.0, 1.0, 0.0},
    {0.0, 1.0, 1.0}, {1.0, 0.5, 0.5}, {0.5, 1.0, 0.5}, {0.5, 0.5, 1.0},
    {1.0, 1.0, 0.5}, {0.5, 1.0, 1.0},
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
  mjtNum mat_world[9] = {1, 0, 0,  0, 1, 0,  0, 0, 1};
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
  mjv_makeConnector(&scene->geoms[scene->ngeom], type, width,
                    from[0], from[1], from[2], to[0], to[1], to[2]);

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
      mju_log(max_value) - mju_log(min_value) / mju_max((steps - 1), 1);
  for (int i = 0; i < steps; i++) {
    values[i] = mju_exp(mju_log(min_value) + i * step);
  }
}

// ============== 2d convex hull ==============

// note: written in MuJoCo-style C for possible future inclusion
namespace {  // private functions in an anonymous namespace

// 2d vector dot-product
mjtNum mju_dot2(const mjtNum vec1[2], const mjtNum vec2[2]) {
  return vec1[0]*vec2[0] + vec1[1]*vec2[1];
}

// 2d vector squared distance
mjtNum mju_sqrdist2(const mjtNum vec1[2], const mjtNum vec2[2]) {
  const mjtNum diff[2] = {vec1[0]-vec2[0], vec1[1]-vec2[1]};
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
  mjtNum center[2] = {0.5*(v1[0] + v0[0]), 0.5*(v1[1] + v0[1])};
  mjtNum t = mju_dot2(query, axis) - mju_dot2(center, axis);
  t = mju_clip(t, -length/2, length/2);
  res[0] = center[0] + t*axis[0];
  res[1] = center[1] + t*axis[1];
}

}  // namespace

// returns point in 2D convex hull that is nearest to query
void NearestInHull(mjtNum res[2], const mjtNum query[2],
                   const mjtNum* points, const int* hull, int num_hull) {
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
    mjtNum x = points[2*i];
    mjtNum y = points[2*i + 1];

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
          IsEdgeOutside(points + 2*hull[num_hull - 1],
                        points + 2*next,
                        points + 2*candidate)) {
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

}  // namespace mjpc
