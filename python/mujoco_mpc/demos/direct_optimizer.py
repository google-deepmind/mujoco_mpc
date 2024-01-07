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

import matplotlib.pyplot as plt
import mujoco
import numpy as np
from numpy import typing as npt
from typing import Tuple


# %%
# differentiate mj_differentiatePos
def diff_differentiatePos(
    model: mujoco.MjModel, dt: float, qpos1: npt.ArrayLike, qpos2: npt.ArrayLike
) -> Tuple[npt.ArrayLike, npt.ArrayLike]:
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

  return jac1, jac2


# %%
# velocity and acceleration from configuration
# v1 = (q1 - q0) / h
# a1 = (v2 - v1) / h = (q2 - 2q1 + q0) / h^2
def qpos_to_qvel_qacc(
    model: mujoco.MjModel,
    qpos: npt.ArrayLike,
    horizon: int,
) -> Tuple[npt.ArrayLike, npt.ArrayLike]:
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
# velocity and acceleration from configurations (derivatives wrt configurations)
def diff_qpos_to_qvel_qacc(
    model: mujoco.MjModel,
    qpos: npt.ArrayLike,
    horizon: int,
) -> Tuple[
    npt.ArrayLike, npt.ArrayLike, npt.ArrayLike, npt.ArrayLike, npt.ArrayLike
]:
  dvdq0 = [np.zeros((model.nv, model.nv)) for t in range(T)]
  dvdq1 = [np.zeros((model.nv, model.nv)) for t in range(T)]
  dadq0 = [np.zeros((model.nv, model.nv)) for t in range(T)]
  dadq1 = [np.zeros((model.nv, model.nv)) for t in range(T)]
  dadq2 = [np.zeros((model.nv, model.nv)) for t in range(T)]

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
# inverse dynamics
# (f, s) <- id(q, v, a, u) = id(q1, v(q0, q1), a(q0, q1, q2), u)
def inverse_dynamics(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos: npt.ArrayLike,
    qvel: npt.ArrayLike,
    qacc: npt.ArrayLike,
    ctrl: npt.ArrayLike,
    horizon: int,
) -> Tuple[npt.ArrayLike, npt.ArrayLike]:
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
      data.ctrl = np.zeros(model.nu)

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
      data.ctrl = np.zeros(model.nu)

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
      data.ctrl = ctrl[:, t]

      # inverse dynamics
      mujoco.mj_inverse(model, data)

      # copy sensor and force
      sensor[:, t] = data.sensordata
      force[:, t] = data.qfrc_inverse

  return sensor, force


# %%
# inverse dynamics (derivatives wrt qpos, qvel, qacc)
# dfdq, dfdv, dfda, dsdq, dsdv, dsda
def diff_inverse_dynamics(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos: npt.ArrayLike,
    qvel: npt.ArrayLike,
    qacc: npt.ArrayLike,
    ctrl: npt.ArrayLike,
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
      data.ctrl = np.zeros(model.nu)

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
      data.ctrl = np.zeros(model.nu)

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
      data.ctrl = ctrl[:, t]

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
# sensor derivative wrt configurations
# ds / dq012
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
# force derivative wrt configurations
# df / dq012
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
# force cost
# cf = sum(w_t * n_t(f_t(qt-1,qt,qt+1) - target_t)) for t = 1,...,T-1)
def cost_force(
    model: mujoco.MjModel,
    force: npt.ArrayLike,
    target: npt.ArrayLike,
    weights: npt.ArrayLike,
    horizon: int,
) -> float:
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
# force cost (derivatives wrt configurations)
# d cf / d q0,...,qT
def diff_cost_force(
    model: mujoco.MjModel,
    force: npt.ArrayLike,
    target: npt.ArrayLike,
    weights: npt.ArrayLike,
    dfdq012: npt.ArrayLike,
    horizon: int,
) -> Tuple[npt.ArrayLike, npt.ArrayLike]:
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
# sensor cost
# cs = sum(w_t * n_t(s_t(qt-1,qt,qt+1) - target_t)) for t = 0,...,T)
def cost_sensor(
    model: mujoco.MjModel,
    sensor: npt.ArrayLike,
    target: npt.ArrayLike,
    weights: npt.ArrayLike,
    horizon: int,
) -> float:
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
# sensor cost (derivatives wrt configurations)
# d cs / d q0,...,qT
def diff_cost_sensor(
    model: mujoco.MjModel,
    sensor: npt.ArrayLike,
    target: npt.ArrayLike,
    weights: npt.ArrayLike,
    dsdq012: npt.ArrayLike,
    horizon: int,
) -> Tuple[npt.ArrayLike, npt.ArrayLike]:
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
# update configuration
# q <- q + step * dq
def configuration_update(
    model: mujoco.MjModel,
    qpos: npt.ArrayLike,
    update: npt.ArrayLike,
    step: float,
    horizon: int,
) -> npt.ArrayLike:
  qpos_new = np.zeros((model.nq, horizon))

  # loop over configurations
  for t in range(horizon):
    q = np.array(qpos[:, t])
    dq = update[(t * model.nv) : ((t + 1) * model.nv)]
    mujoco.mj_integratePos(model, q, dq, step)
    qpos_new[:, t] = q

  return qpos_new


# %%
# set symmetric block matrix in band matrix
def add_block_in_band(
    band: npt.ArrayLike,
    block: npt.ArrayLike,
    scale: float,
    ntotal: int,
    nband: int,
    nblock: int,
    shift: int,
) -> npt.ArrayLike:
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
    ctrl: trajectory of actions/controls (nu x horizon).
    sensor: trajectory of sensor values (nsensordata x horizon).
    force: trajectory of qfrc_actuator values (nv x horizon).
    sensor_target: target sensor values (nsensordata x horizon).
    force_target: target qfrc_inverse values (nv x horizon).
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
    # model + data
    self.model = model
    # TODO(taylor): set discrete inverse dynamics!!
    self.model.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_INVDISCRETE
    self.data = mujoco.MjData(model)

    # horizon
    self.horizon = horizon

    # trajectories
    self.qpos = np.zeros((model.nq, horizon))
    self.qvel = np.zeros((model.nv, horizon))
    self.qacc = np.zeros((model.nv, horizon))
    self.ctrl = np.zeros((model.nu, horizon))
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
    # compute finite-difference velocity and acceleration
    self.qvel, self.qacc = qpos_to_qvel_qacc(self.model, qpos, self.horizon)

    # evaluate inverse dynamics
    self.sensor, self.force = inverse_dynamics(
        self.model,
        self.data,
        qpos,
        self.qvel,
        self.qacc,
        self.ctrl,
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
        self.ctrl,
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
    print("Direct Optimizer Status")
    print(" cost")
    print(" total          :", optimizer.cost_total)
    print(" sensor         :", optimizer.cost_sensor)
    print(" force          :", optimizer.cost_force)
    print(" initial        :", optimizer.cost_initial)
    print("\n")
    print(" iterations")
    print(" step           :", self._iterations_step)
    print(" search         :", self._iterations_search)
    print("\n")
    print(
        " gradient norm  :",
        np.linalg.norm(optimizer._gradient) / optimizer._ntotal,
    )
    print(" regularization :", optimizer._regularization)
    print("\n")
    print(" info           : ", self._status_msg)


# %%
## Example

# 2D Particle Model
xml = """
<mujoco model="Particle">
  <visual>
    <headlight ambient=".4 .4 .4" diffuse=".8 .8 .8" specular="0.1 0.1 0.1"/>
    <map znear=".01"/>
    <quality shadowsize="2048"/>
    <global elevation="-15"/>
  </visual>

  <asset>
    <texture name="blue_grid" type="2d" builtin="checker" rgb1=".02 .14 .44" rgb2=".27 .55 1" width="300" height="300" mark="edge" markrgb="1 1 1"/>
    <material name="blue_grid" texture="blue_grid" texrepeat="1 1" texuniform="true" reflectance=".2"/>
    <texture name="skybox" type="skybox" builtin="gradient" rgb1=".66 .79 1" rgb2=".9 .91 .93" width="800" height="800"/>
    <material name="self" rgba=".7 .5 .3 1"/>
    <material name="decoration" rgba=".2 .6 .3 1"/>
  </asset>

  <option timestep="0.01"></option>

  <worldbody>
    <light name="light" pos="0 0 1"/>
    <camera name="fixed" pos="0 0 .75" quat="1 0 0 0"/>
    <geom name="ground" type="plane" pos="0 0 0" size=".3 .3 .1" material="blue_grid"/>
    <geom name="wall_x" type="plane" pos="-.3 0 .02" zaxis="1 0 0"  size=".02 .3 .02" material="decoration"/>
    <geom name="wall_y" type="plane" pos="0 -.3 .02" zaxis="0 1 0"  size=".3 .02 .02" material="decoration"/>
    <geom name="wall_neg_x" type="plane" pos=".3 0 .02" zaxis="-1 0 0"  size=".02 .3 .02" material="decoration"/>
    <geom name="wall_neg_y" type="plane" pos="0 .3 .02" zaxis="0 -1 0"  size=".3 .02 .02" material="decoration"/>

    <body name="pointmass" pos="0 0 .01">
      <camera name="cam0" pos="0 -0.3 0.3" xyaxes="1 0 0 0 0.7 0.7"/>
      <joint name="root_x" type="slide"  pos="0 0 0" axis="1 0 0" />
      <joint name="root_y" type="slide"  pos="0 0 0" axis="0 1 0" />
      <geom name="pointmass" type="sphere" size=".01" material="self" mass=".3"/>
      <site name="tip" pos="0 0 0" size="0.01"/>
    </body>
  </worldbody>

  <actuator>
    <motor name="x_motor" joint="root_x" gear="1" ctrllimited="true" ctrlrange="-1 1"/>
    <motor name="y_motor" joint="root_y" gear="1" ctrllimited="true" ctrlrange="-1 1"/>
  </actuator>

  <sensor>
    <jointpos name="x" joint="root_x" />
    <jointpos name="y" joint="root_y" />
  </sensor>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

# %%
## rollout
np.random.seed(0)

# simulation horizon
T = 3

# trajectories
qpos = np.zeros((model.nq, T))
qvel = np.zeros((model.nv, T))
qacc = np.zeros((model.nv, T))
ctrl = np.zeros((model.nu, T))
qfrc = np.zeros((model.nv, T))
sensor = np.zeros((model.nsensordata, T))
noisy_sensor = np.zeros((model.nsensordata, T))
time = np.zeros(T)

# set initial state
mujoco.mj_resetData(model, data)

# frames
frames = []
FPS = 1.0 / model.opt.timestep

# simulate
for t in range(T):
  # forward dynamics
  mujoco.mj_forward(model, data)

  # cache
  qpos[:, t] = data.qpos
  qvel[:, t] = data.qvel
  qacc[:, t] = data.qacc
  ctrl[:, t] = data.ctrl
  qfrc[:, t] = data.qfrc_actuator
  sensor[:, t] = data.sensordata
  time[t] = data.time

  # noisy sensors
  noisy_sensor[:, t] = sensor[:, t]

  # intergrate with Euler
  mujoco.mj_Euler(model, data)

# %%
# create optimizer
optimizer = DirectOptimizer(model, T)

# initialize
optimizer.qpos = 0.0 * np.ones((model.nq, T))

# set data
optimizer.sensor_target = sensor
optimizer.force_target = qfrc
optimizer.ctrl = ctrl

# set weights
optimizer.weights_force[:, :] = 1.0
optimizer.weights_sensor[:, :] = 1.0

# settings
optimizer.max_iterations = 10
optimizer.max_search_iterations = 10

# optimize
optimizer.optimize()

# status
optimizer.status()


# %%
def test_gradient(
    optimizer: DirectOptimizer,
    qpos: npt.ArrayLike,
    eps: float = 1.0e-10,
    verbose: bool = False,
):
  # evaluate nominal cost
  c0 = optimizer.cost(qpos)

  # evaluate optimizer gradient
  optimizer._cost_derivatives(qpos)
  g0 = np.array(optimizer._gradient)

  # finite difference gradient
  g = np.zeros(optimizer._ntotal)

  # horizon
  T = optimizer.horizon

  # loop over inputs
  for i in range(optimizer._ntotal):
    # nudge
    nudge = np.zeros(optimizer._ntotal)
    nudge[i] += eps

    # qpos nudge
    qnudge = configuration_update(optimizer.model, qpos, nudge, 1.0, T)

    # evaluate
    c = optimizer.cost(qnudge)

    # derivative
    g[i] = (c - c0) / eps

  if verbose:
    print("gradient optimizer: \n", g0)
    print("gradient finite difference: \n", g)

  # return max difference
  return np.linalg.norm(g - g0, np.Inf)


def test_hessian(
    optimizer: DirectOptimizer,
    qpos: npt.ArrayLike,
    eps: float = 1.0e-5,
    verbose: bool = False,
):
  # evaluate nominal cost
  c0 = optimizer.cost(qpos)

  # evaluate optimizer Hessian
  optimizer._cost_derivatives(qpos)
  h0 = np.zeros((optimizer._ntotal, optimizer._ntotal))
  mujoco.mju_band2Dense(
      h0, optimizer._hessian.ravel(), optimizer._ntotal, optimizer._nband, 0, 1
  )

  # finite difference gradient
  h = np.zeros((optimizer._ntotal, optimizer._ntotal))

  # horizon
  T = optimizer.horizon

  for i in range(optimizer._ntotal):
    for j in range(i, optimizer._ntotal):
      # workspace
      w1 = np.zeros(optimizer._ntotal)
      w2 = np.zeros(optimizer._ntotal)
      w3 = np.zeros(optimizer._ntotal)

      # workspace 1
      w1[i] += eps
      w1[j] += eps

      # qpos nudge 1
      qnudge1 = configuration_update(optimizer.model, qpos, w1, 1.0, T)

      cij = optimizer.cost(qnudge1)

      # workspace 2
      w2[i] += eps

      # qpos nudge 2
      qnudge2 = configuration_update(optimizer.model, qpos, w2, 1.0, T)

      ci = optimizer.cost(qnudge2)

      # workspace 3
      w3[j] += eps

      # qpos nudge 3
      qnudge3 = configuration_update(optimizer.model, qpos, w3, 1.0, T)

      cj = optimizer.cost(qnudge3)

      # Hessian value
      h[i, j] = (cij - ci - cj + c0) / (eps * eps)
      h[j, i] = (cij - ci - cj + c0) / (eps * eps)

  if verbose:
    print("Hessian optimizer: \n", h0)
    print("Hessian finite difference: \n", h)

  # return maximum difference
  return np.linalg.norm((h - h0).ravel(), np.Inf)


# %%
test_gradient(optimizer, np.ones((model.nq, T)))

# %%
test_hessian(optimizer, np.zeros((model.nq, T)))

# %%
# initialization
T = 100
q0 = np.array([-0.25, -0.25])
qM = np.array([-0.25, 0.25])
qN = np.array([0.25, -0.25])
qT = np.array([0.25, 0.25])

# compute linear interpolation
qinterp = np.zeros((model.nq, T))
for t in range(T):
  # slope
  slope = (qT - q0) / T

  # interpolation
  qinterp[:, t] = q0 + t * slope

# time
time = [t * model.opt.timestep for t in range(T)]

# %%
# plot position
fig = plt.figure()

# arm position
plt.plot(qinterp[0, :], qinterp[1, :], label="interpolation", color="black")
plt.plot(q0[0], q0[1], color="magenta", label="waypoint", marker="o")
plt.plot(qM[0], qM[1], color="magenta", marker="o")
plt.plot(qN[0], qN[1], color="magenta", marker="o")
plt.plot(qT[0], qT[1], color="magenta", marker="o")

plt.legend()
plt.xlabel("X")
plt.ylabel("Y")

# %%
# create optimizer
optimizer = DirectOptimizer(model, T)

# settings
optimizer.max_iterations = 10
optimizer.max_search_iterations = 10

# set data
for t in range(T):
  # set initial state
  if t == 0 or t == 1:
    optimizer.qpos[:, t] = q0
    optimizer.sensor_target[: model.nq, t] = q0
    optimizer.weights_force[:, t] = 1.0
    optimizer.weights_sensor[:, t] = 1.0

  # set goal
  elif t >= T - 2:
    optimizer.qpos[:, t] = qT
    optimizer.sensor_target[: model.nq, t] = qT
    optimizer.weights_force[:, t] = 1.0
    optimizer.weights_sensor[:, t] = 1.0

  # set waypoint
  elif t == 25:
    optimizer.qpos[:, t] = qM
    optimizer.sensor_target[: model.nq, t] = qM
    optimizer.weights_force[:, t] = 1.0
    optimizer.weights_sensor[:, t] = 1.0

  # set waypoint
  elif t == 75:
    optimizer.qpos[:, t] = qN
    optimizer.sensor_target[: model.nq, t] = qN
    optimizer.weights_force[:, t] = 1.0
    optimizer.weights_sensor[:, t] = 1.0

  # initialize qpos
  else:
    optimizer.qpos[:, t] = qinterp[:, t]
    optimizer.weights_force[:, t] = 1.0
    optimizer.weights_sensor[:, t] = 0.0

# optimize
optimizer.optimize()

# status
optimizer.status()

# %%
# plot position
fig = plt.figure()

plt.plot(qinterp[0, :], qinterp[1, :], label="interpolation", color="black")
plt.plot(
    optimizer.qpos[0, :],
    optimizer.qpos[1, :],
    label="direct trajopt",
    color="orange",
)
plt.plot(q0[0], q0[1], color="magenta", label="waypoint", marker="o")
plt.plot(qM[0], qM[1], color="magenta", marker="o")
plt.plot(qN[0], qN[1], color="magenta", marker="o")
plt.plot(qT[0], qT[1], color="magenta", marker="o")

plt.legend()
plt.xlabel("X")
plt.ylabel("Y")
