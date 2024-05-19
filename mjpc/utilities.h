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

#ifndef MJPC_UTILITIES_H_
#define MJPC_UTILITIES_H_

#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include <absl/container/flat_hash_map.h>
#include <mujoco/mujoco.h>

namespace mjpc {

// maximum number of traces that are visualized
inline constexpr int kMaxTraces = 99;

// set mjData state
void SetState(const mjModel* model, mjData* data, const double* state);

// get mjData state
void GetState(const mjModel* model, const mjData* data, double* state);

// get numerical data from a custom element in mjModel with the given name
double* GetCustomNumericData(const mjModel* m, std::string_view name);

// get text data from a custom element in mjModel with the given name
char* GetCustomTextData(const mjModel* m, std::string_view name);

// get a scalar value from a custom element in mjModel with the given name
template <typename T>
std::optional<T> GetNumber(const mjModel* m, std::string_view name) {
  double* data = GetCustomNumericData(m, name);
  if (data) {
    return static_cast<T>(data[0]);
  } else {
    return std::nullopt;
  }
}

// get a single numerical value from a custom element in mjModel, or return the
// default value if a custom element with the specified name does not exist
template <typename T>
T GetNumberOrDefault(T default_value, const mjModel* m, std::string_view name) {
  return GetNumber<T>(m, name).value_or(default_value);
}

// reinterpret double as int
int ReinterpretAsInt(double value);

// reinterpret int64_t as double
double ReinterpretAsDouble(int64_t value);

// returns a map from custom field name to the list of valid values for that
// field
absl::flat_hash_map<std::string, std::vector<std::string>>
ResidualSelectionLists(const mjModel* m);

// get the string selected in a drop down with the given name, given the value
// in the residual parameters vector
std::string ResidualSelection(const mjModel* m, std::string_view name,
                              double residual_parameter);
// returns a value for residual parameters that fits the given text value
// in the given list
double ResidualParameterFromSelection(const mjModel* m, std::string_view name,
                                      std::string_view value);

// returns a default value to put in residual parameters, given the index of a
// custom numeric attribute in the model
double DefaultResidualSelection(const mjModel* m, int numeric_index);

// Clamp x between bounds, e.g., bounds[0] <= x[i] <= bounds[1]
void Clamp(double* x, const double* bounds, int n);

// get sensor data using string
double* SensorByName(const mjModel* m, const mjData* d,
                     const std::string& name);

double DefaultParameterValue(const mjModel* model, std::string_view name);

int ParameterIndex(const mjModel* model, std::string_view name);

int CostTermByName(const mjModel* m, const std::string& name);

// return total size of sensors of type user
int ResidualSize(const mjModel* model);

// sanity check that residual size equals total user-sensor dimension
void CheckSensorDim(const mjModel* model, int residual_size);

// get traces from sensors
void GetTraces(double* traces, const mjModel* m, const mjData* d,
               int num_trace);

// get keyframe `qpos` data using string
double* KeyQPosByName(const mjModel* m, const mjData* d,
                      const std::string& name);

// fills t with N numbers, starting from t0 and incrementing by t_step
void LinearRange(double* t, double t_step, double t0, int N);

// find interval in monotonic sequence containing value
template <typename T>
void FindInterval(int* bounds, const std::vector<T>& sequence, double value,
                  int length) {
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
                       const double* ys, int dim, int length);

// linear interpolation
void LinearInterpolation(double* output, double x,
                         const std::vector<double>& xs, const double* ys,
                         int dim, int length);

// coefficients for cubic interpolation
void CubicCoefficients(double* coefficients, double x,
                       const std::vector<double>& xs, int T);

// finite-difference vector
double FiniteDifferenceSlope(double x, const std::vector<double>& xs,
                             const double* ys, int dim, int length, int i);

// cubic polynomial interpolation
void CubicInterpolation(double* output, double x, const std::vector<double>& xs,
                        const double* ys, int dim, int length);

// returns the path to the directory containing the current executable
std::string GetExecutableDir();

// returns path to a model XML file given path relative to models dir
std::string GetModelPath(std::string_view path);

// dx = (x2 - x1) / h
void Diff(mjtNum* dx, const mjtNum* x1, const mjtNum* x2, mjtNum h, int n);

// finite-difference two state vectors ds = (s2 - s1) / h
void StateDiff(const mjModel* m, mjtNum* ds, const mjtNum* s1, const mjtNum* s2,
               mjtNum h);

// return global height of nearest geom in geomgroup under given position
mjtNum Ground(const mjModel* model, const mjData* data, const mjtNum pos[3],
             const mjtByte* geomgroup = nullptr);

// set x to be the point on the segment [p0 p1] that is nearest to x
void ProjectToSegment(double x[3], const double p0[3], const double p1[3]);

// find frame that best matches 4 feet, z points to body
void FootFrame(double feet_pos[3], double feet_mat[9], double feet_quat[4],
               const double body[3], const double foot0[3],
               const double foot1[3], const double foot2[3],
               const double foot3[3]);

// default cost colors
extern const float CostColors[20][3];
constexpr int kNCostColors = sizeof(CostColors) / (sizeof(float) * 3);

// plots - vertical line
void PlotVertical(mjvFigure* fig, double time, double min_value,
                  double max_value, int N, int index);

// plots - update data
void PlotUpdateData(mjvFigure* fig, double* bounds, double x, double y,
                    int length, int index, int x_update, int y_update,
                    double x_bound_lower);

// plots - reset
void PlotResetData(mjvFigure* fig, int length, int index);

// plots - horizontal line
void PlotHorizontal(mjvFigure* fig, const double* xs, double y, int length,
                    int index);

// plots - set data
void PlotData(mjvFigure* fig, double* bounds, const double* xs,
              const double* ys, int dim, int dim_limit, int length,
              int start_index, double x_bound_lower);

// add geom to scene
void AddGeom(mjvScene* scene, mjtGeom type, const mjtNum size[3],
             const mjtNum pos[3], const mjtNum mat[9], const float rgba[4]);

// add connector geom to scene
void AddConnector(mjvScene* scene, mjtGeom type, mjtNum width,
                  const mjtNum from[3], const mjtNum to[3],
                  const float rgba[4]);

// number of available hardware threads
int NumAvailableHardwareThreads();

// check mjData for warnings, return true if any warnings
bool CheckWarnings(mjData* data);

// compute vector with log-based scaling between min and max values
void LogScale(double* values, double max_value, double min_value, int steps);

// get a pointer to a specific element of a vector, or nullptr if out of bounds
template <typename T>
inline T* DataAt(std::vector<T>& vec, typename std::vector<T>::size_type elem) {
  if (elem < vec.size()) {
    return &vec[elem];
  } else {
    return nullptr;
  }
}

// increases the value of an atomic variable.
// in C++20 atomic::operator+= is built-in for floating point numbers, but this
// function works in C++11
inline void IncrementAtomic(std::atomic<double>& v, double a) {
  for (double t = v.load(); !v.compare_exchange_weak(t, t + a);) {
  }
}

// get a pointer to a specific element of a vector, or nullptr if out of bounds
template <typename T>
inline const T* DataAt(const std::vector<T>& vec,
                       typename std::vector<T>::size_type elem) {
  return DataAt(const_cast<std::vector<T>&>(vec), elem);
}

using UniqueMjData = std::unique_ptr<mjData, void (*)(mjData*)>;

inline UniqueMjData MakeUniqueMjData(mjData* d) {
  return UniqueMjData(d, mj_deleteData);
}

using UniqueMjModel = std::unique_ptr<mjModel, void (*)(mjModel*)>;

inline UniqueMjModel MakeUniqueMjModel(mjModel* d) {
  return UniqueMjModel(d, mj_deleteModel);
}

// returns point in 2D convex hull that is nearest to query
void NearestInHull(mjtNum res[2], const mjtNum query[2], const mjtNum* points,
                   const int* hull, int num_hull);

// find the convex hull of a set of 2D points
int Hull2D(int* hull, int num_points, const mjtNum* points);

// TODO(etom): move findiff-related functions to a different library.

// finite-difference gradient
class FiniteDifferenceGradient {
 public:
  // constructor
  explicit FiniteDifferenceGradient(int dim);

  // resize memory
  void Resize(int dim);

  // compute gradient
  void Compute(std::function<double(const double* x)> func,
                  const double* input, int dim);

  // members
  std::vector<double> gradient;
  double epsilon = 1.0e-5;
 private:
  std::vector<double> workspace_;
};

// finite-difference Jacobian
class FiniteDifferenceJacobian {
 public:
  // constructor
  FiniteDifferenceJacobian(int num_output, int num_input);

  // resize memory
  void Resize(int num_output, int num_input);

  // compute Jacobian
  void Compute(std::function<void(double* output, const double* input)> func,
                  const double* input, int num_output, int num_input);

  // members
  std::vector<double> jacobian;
  std::vector<double> jacobian_transpose;
  std::vector<double> output;
  std::vector<double> output_nominal;
  double epsilon = 1.0e-5;
 private:
  std::vector<double> workspace_;
};

// finite-difference Hessian
class FiniteDifferenceHessian {
 public:
  // constructor
  explicit FiniteDifferenceHessian(int dim);

  // resize memory
  void Resize(int dim);

  // compute
  void Compute(std::function<double(const double* x)> func,
                  const double* input, int dim);

  // members
  std::vector<double> hessian;
  double epsilon = 1.0e-5;
 private:
  std::vector<double> workspace1_;
  std::vector<double> workspace2_;
  std::vector<double> workspace3_;
};

// set scaled block (size: rb x cb) in mat (size: rm x cm) given mat upper row
// and left column indices (ri, ci)
void SetBlockInMatrix(double* mat, const double* block, double scale, int rm,
                      int cm, int rb, int cb, int ri, int ci);

// set scaled block (size: rb x cb) in mat (size: rm x cm) given mat upper row
// and left column indices (ri, ci)
void AddBlockInMatrix(double* mat, const double* block, double scale, int rm,
                      int cm, int rb, int cb, int ri, int ci);

// get block (size: rb x cb) from mat (size: rm x cm) given mat upper row
// and left column indices (ri, ci)
void BlockFromMatrix(double* block, const double* mat, int rb, int cb, int rm,
                     int cm, int ri, int ci);

// differentiate mju_subQuat wrt qa, qb
void DifferentiateSubQuat(double jaca[9], double jacb[9], const double qa[4],
                          const double qb[4]);

// differentiate velocity by finite-differencing two positions wrt to qpos1,
// qpos2
void DifferentiateDifferentiatePos(double* jac1, double* jac2,
                                   const mjModel* model, double dt,
                                   const double* qpos1, const double* qpos2);

// compute number of nonzeros in band matrix
int BandMatrixNonZeros(int ntotal, int nband);

// TODO(etom): rename (SecondsSince?)
double GetDuration(std::chrono::steady_clock::time_point time);

// copy symmetric band matrix block by block
void SymmetricBandMatrixCopy(double* res, const double* mat, int dblock,
                             int nblock, int ntotal, int num_blocks,
                             int res_start_row, int res_start_col,
                             int mat_start_row, int mat_start_col,
                             double* scratch);

// zero block (size: rb x cb) in mat (size: rm x cm) given mat upper row
// and left column indices (ri, ci)
void ZeroBlockInMatrix(double* mat, int rm, int cm, int rb, int cb, int ri,
                       int ci);

// square dense to block band matrix
void DenseToBlockBand(double* res, int dim, int dblock, int nblock);

// infinity norm
template <typename T>
T InfinityNorm(T* x, int n) {
  return std::abs(*std::max_element(x, x + n, [](T a, T b) -> bool {
    return (std::abs(a) < std::abs(b));
  }));
}

// trace of square matrix
double Trace(const double* mat, int n);

// determinant of 3x3 matrix
double Determinant3(const double* mat);

// inverse of 3x3 matrix
void Inverse3(double* res, const double* mat);

// condition matrix: res = mat11 - mat10 * mat00 \ mat10^T; return rank of mat00
// TODO(taylor): thread
void ConditionMatrix(double* res, const double* mat, double* mat00,
                     double* mat10, double* mat11, double* tmp0, double* tmp1,
                     int n, int n0, int n1, double* bandfactor = NULL,
                     int nband = 0);

// principal eigenvector of 4x4 matrix
// QUEST algorithm from "Three-Axis Attitude Determination from Vector
// Observations"
void PrincipalEigenVector4(double* res, const double* mat,
                           double eigenvalue_init = 12.0);

// set scaled symmetric block matrix in band matrix
void SetBlockInBand(double* band, const double* block, double scale, int ntotal,
                    int nband, int nblock, int shift, int row_skip = 0,
                    bool add = true);

}  // namespace mjpc

#endif  // MJPC_UTILITIES_H_
