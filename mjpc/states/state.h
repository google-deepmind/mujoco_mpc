#ifndef MJPC_STATES_STATE_H_
#define MJPC_STATES_STATE_H_

#include <shared_mutex>
#include <vector>

#include <mujoco/mujoco.h>

namespace mjpc {

// data and methods for state
class State {
 public:
  friend class StateTest;

  // constructor
  State() = default;

  // destructor
  ~State() = default;

  // ----- methods ----- //

  // initialize settings
  void Initialize(const mjModel* model) {}

  // allocate memory
  void Allocate(const mjModel* model);

  // reset memory to zeros
  void Reset();

  // set state from data
  void Set(const mjModel* model, const mjData* data);

  // set qpos 
  void SetPosition(const mjModel* model, const double* qpos);

  // set qvel 
  void SetVelocity(const mjModel* model, const double* qvel);

  // set act 
  void SetAct(const mjModel* model, const double* act);

  // set mocap
  void SetMocap(const mjModel* model, const double* mocap_pos, const double* mocap_quat);

  // set user data
  void SetUserData(const mjModel* model, const double* userdata);

  // set time 
  void SetTime(const mjModel* model, double time);

  // copy into destination
  void CopyTo(double* dst_state, double* dst_mocap, double* dst_userdata, double* time) const;
  void CopyTo(const mjModel* model, mjData* data) const;

  const std::vector<double>& state() const { return state_; }
  const std::vector<double>& mocap() const { return mocap_; }
  const std::vector<double>& userdata() const { return userdata_; }
  double time() const { return time_; }

 private:
  std::vector<double> state_;  // (state dimension x 1)
  std::vector<double> mocap_;  // (mocap dimension x 1)
  std::vector<double> userdata_; // (nuserdata x 1)
  double time_;
  mutable std::shared_mutex mtx_;
};

}  // namespace mjpc

#endif  // MJPC_STATES_STATE_H_