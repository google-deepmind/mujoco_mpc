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

#include "utilities.h"

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

// DEEPMIND INTERNAL IMPORT
#include <absl/strings/str_cat.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mujoco.h>
#include "array_safety.h"

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

// Clamp x between bounds, e.g., bounds[0] <= x[i] <= bounds[1]
void Clamp(double* x, const double* bounds, int n) {
  for (int i = 0; i < n; i++) {
    x[i] = mju_clip(x[i], bounds[2 * i], bounds[2 * i + 1]);
  }
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

int CostTermByName(const mjModel* m, const mjData* d, const std::string& name) {
  int id = mj_name2id(m, mjOBJ_SENSOR, name.c_str());
  if (id == -1 || m->sensor_type[id] != mjSENS_USER) {
    std::cerr << "cost term \"" << name << "\" not found.\n";
    return -1;
  } else {
    return id;
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

// get keyframe data using string
double* KeyFrameByName(const mjModel* m, const mjData* d,
                       const std::string& name) {
  int id = mj_name2id(m, mjOBJ_KEY, name.c_str());
  if (id == -1) {
    return nullptr;
  } else {
    return m->key_qpos + m->nq * id;
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
void FindInterval(int* bounds, const double* sequence, double value,
                  int length) {
  // final index
  int T = length - 1;

  // set bounds
  bounds[0] = 0;
  bounds[1] = T;

  // index evaluation
  int middle;

  if (sequence[0] <= value) {
    if (sequence[T] > value) {
      while (bounds[0] < bounds[1] - 1) {
        middle = std::ceil((bounds[0] + bounds[1]) / 2.0 - 1e-10);
        if (sequence[middle] <= value) {
          bounds[0] = middle;
        } else {
          bounds[1] = middle;
        }
      }
    } else {
      bounds[0] = bounds[1];
    }
  } else {
    bounds[1] = bounds[0];
  }
}

// zero-order interpolation
void ZeroInterpolation(double* output, double x, const double* xs,
                       const double* ys, int dim, int length) {
  // bounds
  int bounds[2];
  FindInterval(bounds, xs, x, length);

  // zero-order hold
  mju_copy(output, ys + dim * bounds[0], dim);
}

// linear interpolation
void LinearInterpolation(double* output, double x, const double* xs,
                         const double* ys, int dim, int length) {
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
void CubicCoefficients(double* coefficients, double x, const double* xs,
                       int T) {
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
double FiniteDifferenceSlope(double x, const double* xs, const double* ys,
                             int dim, int length, int i) {
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
void CubicInterpolation(double* output, double x, const double* xs,
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
const float CostColors[10][3]{
    {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}, {1.0, 1.0, 0.0},
    {0.0, 1.0, 1.0}, {0.5, 0.0, 0.0}, {0.0, 0.5, 0.0}, {0.0, 0.0, 0.5},
    {0.5, 0.5, 0.0}, {0.0, 0.5, 0.5},
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

}  // namespace mjpc
