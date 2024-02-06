# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Tuple

import mujoco
import numpy as np
from numpy import typing as npt


# %%
def diff_differentiatePos(
    model: mujoco.MjModel, dt: float, qpos1: npt.ArrayLike, qpos2: npt.ArrayLike
) -> Tuple[npt.ArrayLike, npt.ArrayLike]:
  """Differentiate mj_differentiatePos wrt qpos1, qpos2.

  Args:
      model (mujoco.MjModel): MuJoCo model
      dt (float): timestep
      qpos1 (npt.ArrayLike): previous configuration
      qpos2 (npt.ArrayLike): next configuration

  Returns:
      Tuple[npt.ArrayLike, npt.ArrayLike]: Jacobian wrt to qpos1, qpos2
  """
  # initialize Jacobians
  jac1 = np.zeros((model.nv, model.nv))
  jac2 = np.zeros((model.nv, model.nv))

  # loop over joints
  for j in range(model.njnt):
    # get address in qpos, qvel
    padr = model.jnt_qposadr[j]
    vadr = model.jnt_dofadr[j]

    # joint type cases
    match model.jnt_type[j]:
      case mujoco.mjtJoint.mjJNT_FREE:
        for i in range(3):
          jac1[vadr + i, vadr + i] = -1.0 / dt
          jac2[vadr + i, vadr + i] = 1.0 / dt
        vadr += 3
        padr += 3
        blk1 = np.zeros((3, 3))
        blk2 = np.zeros((3, 3))
        mujoco.mjd_subQuat(
            qpos2[padr : (padr + 4)], qpos1[padr : (padr + 4)], blk2, blk1
        )
        idx = slice(vadr, vadr + 3)
        jac1[idx, idx] = blk1
        jac2[idx, idx] = blk2
      case mujoco.mjtJoint.mjJNT_BALL:
        blk1 = np.zeros((3, 3))
        blk2 = np.zeros((3, 3))
        idxq = slice(padr, padr + 4)
        mujoco.mjd_subQuat(qpos2[idxq], qpos1[idxq], blk2, blk1)
        idxv = slice(vadr, vadr + 3)
        jac1[idxv, idxv] = blk1
        jac2[idxv, idxv] = blk2
      case mujoco.mjtJoint.mjJNT_HINGE:
        jac1[vadr, vadr] = -1.0 / dt
        jac2[vadr, vadr] = 1.0 / dt
      case mujoco.mjtJoint.mjJNT_SLIDE:
        jac1[vadr, vadr] = -1.0 / dt
        jac2[vadr, vadr] = 1.0 / dt
      case _:
        raise NotImplementedError("Invalid joint")

  return jac1, jac2


# %%
def qpos_to_qvel_qacc(
    model: mujoco.MjModel,
    qpos: npt.ArrayLike,
    horizon: int,
) -> Tuple[npt.ArrayLike, npt.ArrayLike]:
  """Compute velocity and acceleration using configurations.

  v1 = (q1 - q0) / h
  a1 = (v2 - v1) / h = (q2 - 2q1 + q0) / h^2

  Args:
      model (mujoco.MjModel): MuJoCo model
      qpos (npt.ArrayLike): trajectory of configurations
      horizon (int): number of timesteps

  Returns:
      Tuple[npt.ArrayLike, npt.ArrayLike]: velocity and accelerations
      trajectories
  """
  qvel = np.zeros((model.nv, horizon))
  qacc = np.zeros((model.nv, horizon))

  # loop over configurations
  for t in range(1, horizon):
    # previous and current configurations
    q0 = qpos[:, t - 1]
    q1 = qpos[:, t]

    # compute velocity
    v1 = np.zeros(model.nv)
    mujoco.mj_differentiatePos(model, v1, model.opt.timestep, q0, q1)
    qvel[:, t] = v1

    # compute acceleration
    if t > 1:
      # previous velocity
      v0 = qvel[:, t - 1]

      # previous acceleration
      qacc[:, t - 1] = (v1 - v0) / model.opt.timestep
  return qvel, qacc


# %%
def diff_qpos_to_qvel_qacc(
    model: mujoco.MjModel,
    qpos: npt.ArrayLike,
    horizon: int,
) -> Tuple[
    npt.ArrayLike, npt.ArrayLike, npt.ArrayLike, npt.ArrayLike, npt.ArrayLike
]:
  """Velocity and acceleration from mujoco_mpc.demos.configurations (derivatives wrt configurations.

  Args:
      model (mujoco.MjModel): MuJoCo model
      qpos (npt.ArrayLike): trajectory of configurations
      horizon (int): number of timesteps

  Returns:
      Tuple[ npt.ArrayLike, npt.ArrayLike, npt.ArrayLike, npt.ArrayLike, npt.ArrayLike ]: velocity and acceleration derivatives wrt to configurations
  """
  dvdq0 = [np.zeros((model.nv, model.nv)) for t in range(horizon)]
  dvdq1 = [np.zeros((model.nv, model.nv)) for t in range(horizon)]
  dadq0 = [np.zeros((model.nv, model.nv)) for t in range(horizon)]
  dadq1 = [np.zeros((model.nv, model.nv)) for t in range(horizon)]
  dadq2 = [np.zeros((model.nv, model.nv)) for t in range(horizon)]

  # loop over configurations
  for t in range(1, horizon):
    # previous and current configurations
    q1 = qpos[:, t - 1]
    q2 = qpos[:, t]

    # velocity Jacobians
    D0, D1 = diff_differentiatePos(model, model.opt.timestep, q1, q2)
    dvdq0[t] = D0
    dvdq1[t] = D1

    # acceleration Jacobians
    if t > 1:
      # da1dq0 = -dv1dq0 / h
      dadq0[t - 1] = -1.0 * dvdq0[t - 1] / model.opt.timestep

      # da1dq1 = (dv2dq1 - dv1dq1) / h
      dadq1[t - 1] = (dvdq0[t] - dvdq1[t - 1]) / model.opt.timestep

      # da1dq2 = dv2dq2 / h
      dadq2[t - 1] = dvdq1[t] / model.opt.timestep

  return dvdq0, dvdq1, dadq0, dadq1, dadq2


# %%
def inverse_dynamics(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos: npt.ArrayLike,
    qvel: npt.ArrayLike,
    qacc: npt.ArrayLike,
    horizon: int,
) -> Tuple[npt.ArrayLike, npt.ArrayLike]:
  """Inverse dynamics.

  (s, f) <- id(q, v, a) = id(q1, v(q0, q1), a(q0, q1, q2))

  Args:
      model (mujoco.MjModel): MuJoCo model
      data (mujoco.MjData): MuJoCo data
      qpos (npt.ArrayLike): trajectory of configurations
      qvel (npt.ArrayLike): trajectory of velocities
      qacc (npt.ArrayLike): trajectory of accelerations
      horizon (int): number of timesteps

  Returns:
      Tuple[npt.ArrayLike, npt.ArrayLike]: sensor and force trajectories
  """
  sensor = np.zeros((model.nsensordata, horizon))
  force = np.zeros((model.nv, horizon))

  # loop over horizon
  for t in range(0, horizon):
    # first step
    if t == 0:
      # set data
      data.qpos = qpos[:, t]
      data.qvel = np.zeros(model.nv)
      data.qacc = np.zeros(model.nv)

      # evaluate position sensors
      mujoco.mj_fwdPosition(model, data)
      mujoco.mj_sensorPos(model, data)

      # zero memory
      sensor[:, t] = 0.0

      # set position sensors
      for i in range(model.nsensor):
        if model.sensor_needstage[i] == mujoco.mjtStage.mjSTAGE_POS:
          adr = model.sensor_adr[i]
          dim = model.sensor_dim[i]
          idx = slice(adr, adr + dim)
          sensor[idx, t] = data.sensordata[idx]

    # last step
    elif t == horizon - 1:
      # set data
      data.qpos = qpos[:, t]
      data.qvel = qvel[:, t]
      data.qacc = np.zeros(model.nv)

      # evaluate position and velocity sensors
      mujoco.mj_fwdPosition(model, data)
      mujoco.mj_sensorPos(model, data)
      mujoco.mj_fwdVelocity(model, data)
      mujoco.mj_sensorVel(model, data)

      # zero memory
      sensor[:, t] = 0.0

      # only set position and velocity sensors
      for i in range(model.nsensor):
        needstage = model.sensor_needstage[i]
        if (
            needstage == mujoco.mjtStage.mjSTAGE_POS
            or needstage == mujoco.mjtStage.mjSTAGE_VEL
        ):
          adr = model.sensor_adr[i]
          dim = model.sensor_dim[i]
          idx = slice(adr, adr + dim)
          sensor[idx, t] = data.sensordata[idx]
    else:
      # set data
      data.qpos = qpos[:, t]
      data.qvel = qvel[:, t]
      data.qacc = qacc[:, t]

      # inverse dynamics
      mujoco.mj_inverse(model, data)

      # copy sensor and force
      sensor[:, t] = data.sensordata
      force[:, t] = data.qfrc_inverse

  return sensor, force


# %%
def diff_inverse_dynamics(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos: npt.ArrayLike,
    qvel: npt.ArrayLike,
    qacc: npt.ArrayLike,
    horizon: int,
    eps: float = 1.0e-8,
    flg_actuation: bool = True,
) -> Tuple[
    npt.ArrayLike,
    npt.ArrayLike,
    npt.ArrayLike,
    npt.ArrayLike,
    npt.ArrayLike,
    npt.ArrayLike,
]:
  """Inverse dynamics (derivatives wrt qpos, qvel, qacc).

  dfdq, dfdv, dfda, dsdq, dsdv, dsda

  Args:
      model (mujoco.MjModel): MuJoCo model
      data (mujoco.MjData): MuJoCo data
      qpos (npt.ArrayLike): trajectory of configurations
      qvel (npt.ArrayLike): trajectory of velocities
      qacc (npt.ArrayLike): trajectory of accelerations
      horizon (int): number of timesteps
      eps (float, optional): finite-difference perturbation. Defaults to 1.0e-8.
      flg_actuation (bool, optional): Flag to include qfrc_actuator in inverse dynamics. Defaults to True.

  Returns:
      Tuple[ npt.ArrayLike, npt.ArrayLike, npt.ArrayLike, npt.ArrayLike, npt.ArrayLike, npt.ArrayLike, ]: sensor and force derivatives wrt to configurations, velocities, and accelerations
  """
  # Jacobians
  dfdq = [np.zeros((model.nv, model.nv)) for _ in range(horizon)]
  dfdv = [np.zeros((model.nv, model.nv)) for _ in range(horizon)]
  dfda = [np.zeros((model.nv, model.nv)) for _ in range(horizon)]
  dsdq = [np.zeros((model.nsensordata, model.nv)) for _ in range(horizon)]
  dsdv = [np.zeros((model.nsensordata, model.nv)) for _ in range(horizon)]
  dsda = [np.zeros((model.nsensordata, model.nv)) for _ in range(horizon)]

  # transposed Jacobians
  dqds = np.zeros((model.nv, model.nsensordata))
  dvds = np.zeros((model.nv, model.nsensordata))
  dads = np.zeros((model.nv, model.nsensordata))
  dqdf = np.zeros((model.nv, model.nv))
  dvdf = np.zeros((model.nv, model.nv))
  dadf = np.zeros((model.nv, model.nv))

  # loop over horizon
  for t in range(0, horizon):
    # first step
    if t == 0:
      # set data
      data.qpos = qpos[:, t]
      data.qvel = np.zeros(model.nv)
      data.qacc = np.zeros(model.nv)

      # Jacobian
      mujoco.mjd_inverseFD(
          model,
          data,
          eps,
          flg_actuation,
          None,
          None,
          None,
          dqds,
          None,
          None,
          None,
      )

      # transpose
      dsdq[t] = np.transpose(dqds)

      # zero velocity and acceleration sensor derivatives
      for i in range(model.nsensor):
        if model.sensor_needstage[i] != mujoco.mjtStage.mjSTAGE_POS:
          adr = model.sensor_adr[i]
          dim = model.sensor_dim[i]
          dsdq[t][adr : (adr + dim), 0 : model.nv] = 0.0

    # last step
    elif t == horizon - 1:
      # set data
      data.qpos = qpos[:, t]
      data.qvel = qvel[:, t]
      data.qacc = np.zeros(model.nv)

      # Jacobian
      mujoco.mjd_inverseFD(
          model,
          data,
          eps,
          flg_actuation,
          None,
          None,
          None,
          dqds,
          dvds,
          None,
          None,
      )

      # transpose
      dsdq[t] = np.transpose(dqds)
      dsdv[t] = np.transpose(dvds)

      # zero acceleration sensor derivatives
      for i in range(model.nsensor):
        if model.sensor_needstage[i] == mujoco.mjtStage.mjSTAGE_ACC:
          adr = model.sensor_adr[i]
          dim = model.sensor_dim[i]
          idx = slice(adr, adr + dim)
          dsdq[t][idx, 0 : model.nv] = 0.0
          dsdv[t][idx, 0 : model.nv] = 0.0
    else:
      # set data
      data.qpos = qpos[:, t]
      data.qvel = qvel[:, t]
      data.qacc = qacc[:, t]

      # Jacobian
      mujoco.mjd_inverseFD(
          model,
          data,
          eps,
          flg_actuation,
          dqdf,
          dvdf,
          dadf,
          dqds,
          dvds,
          dads,
          None,
      )

      # transpose
      dsdq[t] = np.transpose(dqds)
      dsdv[t] = np.transpose(dvds)
      dsda[t] = np.transpose(dads)
      dfdq[t] = np.transpose(dqdf)
      dfdv[t] = np.transpose(dvdf)
      dfda[t] = np.transpose(dadf)

  return dfdq, dfdv, dfda, dsdq, dsdv, dsda


# %%
def diff_sensor(
    model: mujoco.MjModel,
    dsdq: npt.ArrayLike,
    dsdv: npt.ArrayLike,
    dsda: npt.ArrayLike,
    dvdq0: npt.ArrayLike,
    dvdq1: npt.ArrayLike,
    dadq0: npt.ArrayLike,
    dadq1: npt.ArrayLike,
    dadq2: npt.ArrayLike,
    horizon: int,
) -> npt.ArrayLike:
  """Sensor derivative wrt configurations.

  ds / dq012

  Args:
      model (mujoco.MjModel): MuJoCo model
      dsdq (npt.ArrayLike): trajectory of sensor derivatives wrt configurations
      dsdv (npt.ArrayLike): trajectory of sensor derivatives wrt velocities
      dsda (npt.ArrayLike): trajectory of sensor derivatives wrt accelerations
      dvdq0 (npt.ArrayLike): trajectory of velocity derivatives wrt previous configuration
      dvdq1 (npt.ArrayLike): trajectory of velocity derivatives wrt current configuration
      dadq0 (npt.ArrayLike): trajectory of acceleration derivatives wrt previous configuration
      dadq1 (npt.ArrayLike): trajectory of acceleration derivatives wrt current configuration
      dadq2 (npt.ArrayLike): trajectory of acceleration derivatives wrt next configuration
      horizon (int): number of timesteps

  Returns:
      npt.ArrayLike: trajectory of sensor derivatives wrt previous, current, and next configurations
  """
  dsdq012 = [
      np.zeros((model.nsensordata, 3 * model.nv)) for _ in range(horizon)
  ]
  # loop over horizon
  for t in range(0, horizon):
    # first step
    if t == 0:
      dsdq012[t][:, 0 : model.nv] = 0.0
      dsdq012[t][:, model.nv : (2 * model.nv)] = dsdq[t]
      dsdq012[t][:, (2 * model.nv) : (3 * model.nv)] = 0.0
    # last step
    elif t == horizon - 1:
      dsdq012[t][:, 0 : model.nv] = dsdv[t] @ dvdq0[t]
      dsdq012[t][:, model.nv : (2 * model.nv)] = dsdq[t] + dsdv[t] @ dvdq1[t]
      dsdq012[t][:, (2 * model.nv) : (3 * model.nv)] = 0.0
    else:
      dsdq012[t][:, 0 : model.nv] = dsdv[t] @ dvdq0[t] + dsda[t] @ dadq0[t]
      dsdq012[t][:, model.nv : (2 * model.nv)] = (
          dsdq[t] + dsdv[t] @ dvdq1[t] + dsda[t] @ dadq1[t]
      )
      dsdq012[t][:, (2 * model.nv) : (3 * model.nv)] = dsda[t] @ dadq2[t]

  return dsdq012


# %%
def diff_force(
    model: mujoco.MjModel,
    dfdq: npt.ArrayLike,
    dfdv: npt.ArrayLike,
    dfda: npt.ArrayLike,
    dvdq0: npt.ArrayLike,
    dvdq1: npt.ArrayLike,
    dadq0: npt.ArrayLike,
    dadq1: npt.ArrayLike,
    dadq2: npt.ArrayLike,
    horizon: int,
) -> npt.ArrayLike:
  """Force derivative wrt configurations.

  df / dq012

  Args:
      model (mujoco.MjModel): MuJoCo model
      dfdq (npt.ArrayLike): trajectory of force derivatives wrt configurations
      dfdv (npt.ArrayLike): trajectory of force derivatives wrt velocities
      dfda (npt.ArrayLike): trajectory of force derivatives wrt accelerations
      dvdq0 (npt.ArrayLike): trajectory of velocity derivatives wrt previous configuration
      dvdq1 (npt.ArrayLike): trajectory of velocity derivatives wrt current configuration
      dadq0 (npt.ArrayLike): trajectory of acceleration derivatives wrt previous configuration
      dadq1 (npt.ArrayLike): trajectory of acceleration derivatives wrt current configuration
      dadq2 (npt.ArrayLike): trajectory of acceleration derivatives wrt next configuration
      horizon (int): number of timesteps

  Returns:
      npt.ArrayLike: trajectory of force derivatives wrt to previous, current, and next configurations
  """
  dfdq012 = [np.zeros((model.nv, 3 * model.nv)) for _ in range(horizon)]

  # loop over horizon
  for t in range(horizon):
    # first step
    if t == 0:
      dfdq012[t][:, :] = 0.0
    # last step
    elif t == horizon - 1:
      dfdq012[t][:, :] = 0.0
    else:
      dfdq012[t][:, 0 : model.nv] = dfdv[t] @ dvdq0[t] + dfda[t] @ dadq0[t]
      dfdq012[t][:, model.nv : (2 * model.nv)] = (
          dfdq[t] + dfdv[t] @ dvdq1[t] + dfda[t] @ dadq1[t]
      )
      dfdq012[t][:, (2 * model.nv) : (3 * model.nv)] = dfda[t] @ dadq2[t]

  return dfdq012


# %%
def cost_force(
    model: mujoco.MjModel,
    force: npt.ArrayLike,
    target: npt.ArrayLike,
    weights: npt.ArrayLike,
    horizon: int,
) -> float:
  """Force cost

  cf = sum(w_t * n_t(f_t(qt-1,qt,qt+1) - target_t)) for t = 1,...,T-1)

  Args:
      model (mujoco.MjModel): MuJoCo model
      force (npt.ArrayLike): trajectory of forces
      target (npt.ArrayLike): trajectory of target forces
      weights (npt.ArrayLike): trajectory of weights
      horizon (int): number of timesteps

  Returns:
      float: total force cost across timesteps
  """
  # initialize cost
  cost = 0.0

  # loop over horizon
  for t in range(1, horizon - 1):
    # residual
    res = force[:, t] - target[:, t]

    # quadratic cost
    quad_cost = 0.5 * res.T @ np.diag(weights[:, t]) @ res

    # scale
    scale = (model.opt.timestep**4) / (model.nv * (horizon - 2))
    cost += scale * quad_cost

  return cost


# %%
def diff_cost_force(
    model: mujoco.MjModel,
    force: npt.ArrayLike,
    target: npt.ArrayLike,
    weights: npt.ArrayLike,
    dfdq012: npt.ArrayLike,
    horizon: int,
) -> Tuple[npt.ArrayLike, npt.ArrayLike]:
  """Force cost (derivatives wrt configurations).

  d cf / d q0,...,qT

  Args:
      model (mujoco.MjModel): MuJoCo model
      force (npt.ArrayLike): trajectory of forces
      target (npt.ArrayLike): trajectory of force targets
      weights (npt.ArrayLike): trajectory of weights
      dfdq012 (npt.ArrayLike): trajectory of force derivatives wrt previous, current, and next configurations
      horizon (int): number of timesteps

  Returns:
      Tuple[npt.ArrayLike, npt.ArrayLike]: gradient and Hessian of force cost wrt to configurations
  """
  # dimensions
  ntotal = model.nv * horizon
  nband = 3 * model.nv

  # derivatives
  grad = np.zeros(ntotal)
  hess = np.zeros((ntotal, nband))

  # loop over horizon
  for t in range(1, horizon - 1):
    # residual
    res = force[:, t] - target[:, t]

    # scale
    scale = (model.opt.timestep**4) / (model.nv * (horizon - 2))

    # quadratic norm gradient
    norm_grad = scale * np.diag(weights[:, t]) @ res

    # quadratic norm Hessian
    norm_hess = np.diag(scale * weights[:, t])

    # indices
    idx = slice((t - 1) * model.nv, (t + 2) * model.nv)

    # gradient
    grad[idx] += dfdq012[t].T @ norm_grad

    # Hessian
    blk = dfdq012[t].T @ norm_hess @ dfdq012[t]
    hess = add_block_in_band(
        hess, blk, 1.0, ntotal, nband, 3 * model.nv, (t - 1) * model.nv
    )

  return grad, hess


# %%
def cost_sensor(
    model: mujoco.MjModel,
    sensor: npt.ArrayLike,
    target: npt.ArrayLike,
    weights: npt.ArrayLike,
    horizon: int,
) -> float:
  """Sensor cost.

  cs = sum(w_t * n_t(s_t(qt-1,qt,qt+1) - target_t)) for t = 0,...,T)

  Args:
      model (mujoco.MjModel): MuJoCo model
      sensor (npt.ArrayLike): trajectory of sensors
      target (npt.ArrayLike): trajectory of sensor targets
      weights (npt.ArrayLike): trajectory of weights
      horizon (int): number of timesteps

  Returns:
      float: total sensor cost across timesteps
  """
  # initialize cost
  cost = 0.0

  # loop over horizon
  for t in range(horizon):
    # residual
    res = sensor[:, t] - target[:, t]

    # loop over sensors
    for i in range(model.nsensor):
      # skip
      needstage = model.sensor_needstage[i]
      if t == 0 and needstage != mujoco.mjtStage.mjSTAGE_POS:
        continue
      if t == horizon - 1 and needstage == mujoco.mjtStage.mjSTAGE_ACC:
        continue

      # sensor i
      adr = model.sensor_adr[i]
      dim = model.sensor_dim[i]

      # indices
      idx = slice(adr, adr + dim)

      # quadratic cost
      quad_cost = 0.5 * weights[i, t] * np.dot(res[idx], res[idx])

      # scale
      time_scale = 1.0
      if needstage == mujoco.mjtStage.mjSTAGE_VEL:
        time_scale = model.opt.timestep**2
      elif needstage == mujoco.mjtStage.mjSTAGE_ACC:
        time_scale = model.opt.timestep**4
      scale = time_scale / (dim * horizon)
      cost += scale * quad_cost

  return cost


# %%
def diff_cost_sensor(
    model: mujoco.MjModel,
    sensor: npt.ArrayLike,
    target: npt.ArrayLike,
    weights: npt.ArrayLike,
    dsdq012: npt.ArrayLike,
    horizon: int,
) -> Tuple[npt.ArrayLike, npt.ArrayLike]:
  """Sensor cost (derivatives wrt configurations).

  d cs / d q0,...,qT

  Args:
      model (mujoco.MjModel): MuJoCo model
      sensor (npt.ArrayLike): trajectory of sensors
      target (npt.ArrayLike): trajectory of sensor targets
      weights (npt.ArrayLike): trajectory of weights
      dsdq012 (npt.ArrayLike): trajectory of sensor derivatives wrt previous, current, and next configurations
      horizon (int): number of timesteps

  Returns:
      Tuple[npt.ArrayLike, npt.ArrayLike]: gradient and Hessian of total sensor cost wrt configuration
  """
  # dimensions
  ntotal = model.nv * horizon
  nband = 3 * model.nv

  # derivatives
  grad = np.zeros(ntotal)
  hess = np.zeros((ntotal, nband))

  # loop over horizon
  for t in range(horizon):
    # residual
    res = sensor[:, t] - target[:, t]

    # loop over sensors
    for i in range(model.nsensor):
      # skip
      needstage = model.sensor_needstage[i]
      if t == 0 and needstage != mujoco.mjtStage.mjSTAGE_POS:
        continue
      if t == horizon - 1 and needstage == mujoco.mjtStage.mjSTAGE_ACC:
        continue

      # adr
      adr = model.sensor_adr[i]
      dim = model.sensor_dim[i]

      # indices
      idx = slice(adr, adr + dim)

      # scale
      time_scale = 1.0
      if needstage == mujoco.mjtStage.mjSTAGE_VEL:
        time_scale = model.opt.timestep**2
      elif needstage == mujoco.mjtStage.mjSTAGE_ACC:
        time_scale = model.opt.timestep**4
      scale = time_scale / (dim * horizon)

      # quadratic norm gradient
      normi_grad = (scale * weights[i, t] * res[idx]).reshape((dim, 1))

      # quadratic norm Hessian
      normi_hess = (scale * weights[i, t] * np.eye(dim)).reshape((dim, dim))

      # first step
      if t == 0:
        # indices
        idxt = slice(0, model.nv)

        # subblock
        dsidq1 = dsdq012[t][idx, model.nv : (2 * model.nv)].reshape(
            (dim, model.nv)
        )

        # gradient
        grad[idxt] += (dsidq1.T @ normi_grad).ravel()

        # Hessian
        blk = dsidq1.T @ normi_hess @ dsidq1
        hess = add_block_in_band(hess, blk, 1.0, ntotal, nband, model.nv, 0)
      # last step
      elif t == horizon - 1:
        # indices
        idxt = slice((t - 1) * model.nv, (t + 1) * model.nv)

        # subblock
        dsidq01 = dsdq012[t][idx, 0 : (2 * model.nv)].reshape(
            (dim, 2 * model.nv)
        )

        # gradient
        grad[idxt] += (dsidq01.T @ normi_grad).ravel()

        # Hessian
        blk = dsidq01.T @ normi_hess @ dsidq01
        hess = add_block_in_band(
            hess, blk, 1.0, ntotal, nband, 2 * model.nv, (t - 1) * model.nv
        )
      else:
        # indices
        idxt = slice((t - 1) * model.nv, (t + 2) * model.nv)

        # subblock
        dsidq012 = dsdq012[t][idx, :].reshape((dim, 3 * model.nv))

        # gradient
        grad[idxt] += (dsidq012.T @ normi_grad).ravel()

        # Hessian
        blk = dsidq012.T @ normi_hess @ dsidq012
        hess = add_block_in_band(
            hess, blk, 1.0, ntotal, nband, 3 * model.nv, (t - 1) * model.nv
        )

  return grad, hess


# %%
def configuration_update(
    model: mujoco.MjModel,
    qpos: npt.ArrayLike,
    update: npt.ArrayLike,
    step: float,
    horizon: int,
) -> npt.ArrayLike:
  """Update configuration.

  q <- q + step * dq

  Args:
      model (mujoco.MjModel): MuJoCo model
      qpos (npt.ArrayLike): trajectory of configurations
      update (npt.ArrayLike): search direction
      step (float): size of update to configurations
      horizon (int): number of timesteps

  Returns:
      npt.ArrayLike: updated configuration trajectory
  """
  qpos_new = np.zeros((model.nq, horizon))

  # loop over configurations
  for t in range(horizon):
    q = np.array(qpos[:, t])
    dq = update[(t * model.nv) : ((t + 1) * model.nv)]
    mujoco.mj_integratePos(model, q, dq, step)
    qpos_new[:, t] = q

  return qpos_new


# %%
def add_block_in_band(
    band: npt.ArrayLike,
    block: npt.ArrayLike,
    scale: float,
    ntotal: int,
    nband: int,
    nblock: int,
    shift: int,
) -> npt.ArrayLike:
  """Set symmetric block matrix in band matrix.

  Args:
      band (npt.ArrayLike): band matrix
      block (npt.ArrayLike): block matrix to be added to band matrix
      scale (float): scaling for block matrix
      ntotal (int): number of band matrix rows
      nband (int): dimension of block
      nblock (int): number of blocks in band
      shift (int): number of rows to shift before adding block

  Returns:
      npt.ArrayLike: _description_
  """
  band_update = np.copy(band)
  # loop over block rows
  for i in range(nblock):
    # width of block lower triangle row
    width = i + 1

    # number of leading zeros in band row
    column_shift = nband - width

    # add block row into band row
    band_update[shift + i, column_shift : (column_shift + width)] += (
        scale * block[i, :width]
    )

  return band_update


# %%
class DirectOptimizer:
  """`DirectOptimizer` for planning state trajectories using MuJoCo inverse dynamics.

  Attributes:
    model: MuJoCo mjModel.
    data: MuJoCo mjData.
    horizon: planning horizon.
    qpos: trajectory of configurations (nq x horizon).
    qvel: trajectory of velocities (nv x horizon).
    qacc: trajectory of accelerations (nv x horizon).
    sensor: trajectory of sensor values (nsensordata x horizon).
    force: trajectory of qfrc_inverse values (nv x horizon).
    sensor_target: target sensor values (nsensordata x horizon).
    force_target: target qfrc_actuator values (nv x horizon).
    weights_sensor: weights for sensor norms (nsensor x horizon).
    weights_force: weights for force norms (nv x horizon).
    _dvdq0: Jacobian of velocity wrt previous configuration ((nv x nv) x horizon).
    _dvdq1: Jacobian of velocity wrt current configuration ((nv x nv) x horizon).
    _dadq0: Jacobian of acceleration wrt previous configuration ((nv x nv) x horizon).
    _dadq1: Jacobian of acceleration wrt current configuration ((nv x nv) x horizon).
    _dadq2: Jacobian of acceleration wrt next configuration ((nv x nv) x horizon).
    _dsdq: Jacobian of sensor wrt configuration ((nsensordata x nv) x horizon).
    _dsdv: Jacobian of sensor wrt velocity ((nsensordata x nv) x horizon).
    _dsda: Jacobian of sensor wrt acceleration ((nsensordata x nv) x horizon).
    _dfdq: Jacobian of force wrt configuration ((nv x nv) x horizon).
    _dfdv: Jacobian of force wrt velocity ((nv x nv) x horizon).
    _dfda: Jacobian of force wrt acceleration ((nv x nv) x horizon).
    _dsdq012: Jacobian of sensor wrt previous, current, and next configurations ((nv x 3 nv) x horizon).
    _dfdq012: Jacobian of force wrt previous, current, and next configurations ((nv x 3 nv) x horizon).
    cost_total: combined force and sensor costs.
    cost_force: sum of weighted force norms.
    cost_sensor: sum of weighted sensor norms.
    cost_initial: initial total cost.
    _ntotal: number of decision variables (nv * horizon).
    _nband: cost Hessian band dimension (3 * nv).
    _gradient: gradient of cost wrt to decision variables (nv * horizon).
    _hessian: band representation of cost Hessian wrt decision variables (nv * horizon x 3 * nv).
    _hessian_factor: factorization of band represented cost Hessian
    _search_direction: Gauss-Newton search direction (nv * horizon).
    _qpos_candidate: candidate search point for configuration trajectory (nq x horizon).
    _regularization: current value for cost Hessian regularization.
    _gradient_norm: normalized L2-norm of cost gradient.
    _iterations_step: number of step iterations performed.
    _iterations_search: number of curve search iterations performed.
    _status_msg: status message.
    gradient_tolerance: setting for solve termination based on cost gradient.
    direction_tolerance: setting for solver termination based on search direction infinity norm.
    max_iterations: setting for maximum number of step iterations performed.
    max_search_iterations: setting for maximum number of search iterations performed at each step iteration.
    regularization_min: minimum regularization value.
    regularization_max: maximum regularization value.
    regularization_scale: value for increasing/decreasing regularization.
  """

  def __init__(self, model: mujoco.MjModel, horizon: int):
    """Construct direct optimizer.

    Args:
        model (mujoco.MjModel): MuJoCo model
        horizon (int): number of timesteps
    """
    # model + data
    self.model = model

    # set discrete inverse dynamics
    self.model.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_INVDISCRETE
    self.data = mujoco.MjData(model)

    # horizon
    self.horizon = horizon

    # trajectories
    self.qpos = np.zeros((model.nq, horizon))
    self.qvel = np.zeros((model.nv, horizon))
    self.qacc = np.zeros((model.nv, horizon))
    self.sensor = np.zeros((model.nsensordata, horizon))
    self.force = np.zeros((model.nv, horizon))

    # targets
    self.sensor_target = np.zeros((model.nsensordata, horizon))
    self.force_target = np.zeros((model.nv, horizon))

    # weights
    self.weights_sensor = np.zeros((model.nsensor, horizon))
    self.weights_force = np.zeros((model.nv, horizon))

    # finite-difference velocity and acceleration Jacobians (wrt configurations)
    self._dvdq0 = [np.zeros((model.nv, model.nv)) for t in range(horizon)]
    self._dvdq1 = [np.zeros((model.nv, model.nv)) for t in range(horizon)]
    self._dadq0 = [np.zeros((model.nv, model.nv)) for t in range(horizon)]
    self._dadq1 = [np.zeros((model.nv, model.nv)) for t in range(horizon)]
    self._dadq2 = [np.zeros((model.nv, model.nv)) for t in range(horizon)]

    # inverse dynamics Jacobians (wrt configuration, velocity, acceleration)
    self._dsdq = [
        np.zeros((model.nsensordata, model.nv)) for t in range(horizon)
    ]
    self._dsdv = [
        np.zeros((model.nsensordata, model.nv)) for t in range(horizon)
    ]
    self._dsda = [
        np.zeros((model.nsensordata, model.nv)) for t in range(horizon)
    ]
    self._dfdq = [np.zeros((model.nv, model.nv)) for t in range(horizon)]
    self._dfdv = [np.zeros((model.nv, model.nv)) for t in range(horizon)]
    self._dfda = [np.zeros((model.nv, model.nv)) for t in range(horizon)]

    # sensor and force Jacobians (wrt configurations)
    self._dsdq012 = [
        np.zeros((model.nsensordata, 3 * model.nv)) for t in range(horizon)
    ]
    self._dfdq012 = [np.zeros((model.nv, 3 * model.nv)) for t in range(horizon)]

    # cost terms
    self.cost_total = 0.0
    self.cost_force = 0.0
    self.cost_sensor = 0.0
    self.cost_initial = 0.0

    # cost derivatives
    self._ntotal = model.nv * horizon
    self._nband = 3 * model.nv
    self._gradient = np.zeros(self._ntotal)
    self._hessian = np.zeros((self._ntotal, self._nband))

    # cost Hessian factor
    self._hessian_factor = np.zeros((self._ntotal, self._nband))

    # search direction
    self._search_direction = np.zeros(self._ntotal)

    # candidate qpos
    self._qpos_candidate = np.zeros((model.nq, horizon))

    # regularization
    self._regularization = 1.0e-12

    # status
    self._gradient_norm = 0.0
    self._iterations_step = 0
    self._iterations_search = 0
    self._status_msg = ""
    self._improvement = 0.0

    # settings
    self.gradient_tolerance = 1.0e-6
    self.cost_difference_tolerance = 1.0e-6
    self.direction_tolerance = 1.0e-6
    self.max_iterations = 1000
    self.max_search_iterations = 1000
    self.regularization_min = 1.0e-8
    self.regularization_max = 1.0e12
    self.regularization_scale = np.sqrt(10.0)

  def cost(self, qpos: npt.ArrayLike) -> float:
    """Return total cost (force + sensor)

    Args:
        qpos (npt.ArrayLike): trajectory of configurations

    Returns:
        float: total cost (sensor + force)
    """
    # compute finite-difference velocity and acceleration
    self.qvel, self.qacc = qpos_to_qvel_qacc(self.model, qpos, self.horizon)

    # evaluate inverse dynamics
    self.sensor, self.force = inverse_dynamics(
        self.model,
        self.data,
        qpos,
        self.qvel,
        self.qacc,
        self.horizon,
    )

    # force cost
    self.cost_force = cost_force(
        self.model,
        self.force,
        self.force_target,
        self.weights_force,
        self.horizon,
    )

    # sensor cost
    self.cost_sensor = cost_sensor(
        self.model,
        self.sensor,
        self.sensor_target,
        self.weights_sensor,
        self.horizon,
    )

    # total cost
    self.cost_total = self.cost_force + self.cost_sensor

    return self.cost_total

  def _cost_derivatives(
      self,
      qpos: npt.ArrayLike,
  ):
    """Compute total cost derivatives.

    Args:
        qpos (npt.ArrayLike): trajectory of configurations
    """
    # evaluate cost to compute intermediate values
    self.cost(qpos)

    # finite-difference Jacobians
    (
        self._dvdq0,
        self._dvdq1,
        self._dadq0,
        self._dadq1,
        self._dadq2,
    ) = diff_qpos_to_qvel_qacc(
        self.model,
        qpos,
        self.horizon,
    )

    # inverse dynamics Jacobians
    (
        self._dfdq,
        self._dfdv,
        self._dfda,
        self._dsdq,
        self._dsdv,
        self._dsda,
    ) = diff_inverse_dynamics(
        self.model,
        self.data,
        qpos,
        self.qvel,
        self.qacc,
        self.horizon,
    )

    # force derivatives
    self._dfdq012 = diff_force(
        self.model,
        self._dfdq,
        self._dfdv,
        self._dfda,
        self._dvdq0,
        self._dvdq1,
        self._dadq0,
        self._dadq1,
        self._dadq2,
        self.horizon,
    )

    # sensor derivatives
    self._dsdq012 = diff_sensor(
        self.model,
        self._dsdq,
        self._dsdv,
        self._dsda,
        self._dvdq0,
        self._dvdq1,
        self._dadq0,
        self._dadq1,
        self._dadq2,
        self.horizon,
    )

    # force cost derivatives
    force_gradient, force_hessian = diff_cost_force(
        self.model,
        self.force,
        self.force_target,
        self.weights_force,
        self._dfdq012,
        self.horizon,
    )

    # sensor cost derivatives
    sensor_gradient, sensor_hessian = diff_cost_sensor(
        self.model,
        self.sensor,
        self.sensor_target,
        self.weights_sensor,
        self._dsdq012,
        self.horizon,
    )

    self._gradient = force_gradient + sensor_gradient
    self._hessian = force_hessian + sensor_hessian

  def _eval_search_direction(self) -> bool:
    """Compute search direction.

    Returns:
        bool: Flag indicating search direction computation success.
    """
    # factorize cost Hessian
    self._hessian_factor = np.array(self._hessian.ravel())
    min_diag = mujoco.mju_cholFactorBand(
        self._hessian_factor,
        self._ntotal,
        self._nband,
        0,
        self._regularization,
        0.0,
    )

    # check rank deficient
    if min_diag < 1.0e-16:
      self._status_msg = "rank deficient cost Hessian"
      return False

    # compute search direction
    mujoco.mju_cholSolveBand(
        self._search_direction,
        self._hessian_factor,
        self._gradient,
        self._ntotal,
        self._nband,
        0,
    )

    # check small direction
    if (
        np.linalg.norm(self._search_direction, np.inf)
        < self.direction_tolerance
    ):
      self._status_msg = "small search direction"
      return False

    return True

  def _update_regularization(self) -> bool:
    """Update regularization.

    Returns:
        bool: Flag indicating success of regularization update
    """
    # compute expected = g' d + 0.5 d' H d
    expected = np.dot(self._gradient, self._search_direction)
    tmp = np.zeros(self._ntotal)
    mujoco.mju_bandMulMatVec(
        tmp,
        self._hessian,
        self._search_direction,
        self._ntotal,
        self._nband,
        0,
        1,
        1,
    )
    expected += 0.5 * np.dot(self._search_direction, tmp)

    # check for no expected decrease
    if expected <= 0.0:
      self._status_msg = "no expected decrease"
      return False

    # reduction ratio
    reduction_ratio = self._improvement / expected

    # update regularization
    if reduction_ratio > 0.75:
      # decrease
      self._regularization = np.maximum(
          self.regularization_min,
          self._regularization / self.regularization_scale,
      )
    elif reduction_ratio < 0.25:
      # increase
      self._regularization = np.minimum(
          self.regularization_max,
          self._regularization * self.regularization_scale,
      )

    return True

  def optimize(self):
    """Optimize configuration trajectories."""
    # reset status
    self._gradient_norm = 0.0
    self._iterations_step = 0
    self._iterations_search = 0

    # initial cost
    self.cost_initial = self.cost(self.qpos)
    current_cost = self.cost_initial

    # reset regularization
    self._regularization = self.regularization_min

    # steps
    for i in range(self.max_iterations):
      # set iteration count
      self._iterations_step = i

      # compute search direction
      self._cost_derivatives(self.qpos)

      # check gradient tolerance
      self._gradient_norm = np.linalg.norm(self._gradient) / self._ntotal
      if self._gradient_norm < self.gradient_tolerance:
        self._status_msg = "gradient tolerance achieved"
        return

      # search iterations
      candidate_cost = current_cost
      self._improvement = 0.0
      self._regularization = self.regularization_min
      for j in range(self.max_search_iterations):
        # set iteration count
        self._iterations_search = j

        # max search iterations
        if j == self.max_search_iterations - 1:
          self._status_msg = "max search iterations"
          return

        # compute search direction
        if not self._eval_search_direction():
          return

        # compute candidate
        self._qpos_candidate = configuration_update(
            self.model,
            self.qpos,
            self._search_direction,
            -1.0,
            self.horizon,
        )

        # candidate cost
        candidate_cost = self.cost(self._qpos_candidate)
        self._improvement = current_cost - candidate_cost

        # check improvement
        if candidate_cost < current_cost:
          # update cost
          current_cost = candidate_cost

          # update configurations
          self.qpos = np.array(self._qpos_candidate)
          break
        else:
          # increase regularization
          self._regularization = np.minimum(
              self.regularization_scale * self._regularization,
              self.regularization_max,
          )

      # max search iterations
      if i == self.max_iterations - 1:
        self._status_msg = "max step iterations"
        return

      # update regularization
      if not self._update_regularization():
        return

  # print optimizer status
  def status(self):
    """Print status information."""
    print("Direct Optimizer Status")
    print(" cost")
    print(" total          :", self.cost_total)
    print(" sensor         :", self.cost_sensor)
    print(" force          :", self.cost_force)
    print(" initial        :", self.cost_initial)
    print("\n")
    print(" iterations")
    print(" step           :", self._iterations_step)
    print(" search         :", self._iterations_search)
    print("\n")
    print(
        " gradient norm  :",
        np.linalg.norm(self._gradient) / self._ntotal,
    )
    print(" regularization :", self._regularization)
    print("\n")
    print(" info           : ", self._status_msg)
