# Direct optimizer

**Table of Contents**

- [Direct optimizer](#direct-optimizer)
  - [Cost Function](#cost-function)
  - [Cost Derivatives](#cost-derivatives)
  - [Practical workflow](#practical-workflow)
  - [Reference](#reference)
  - [API](#api)

## Cost Function
```math
\begin{aligned}
    \underset{q_{0:T}}{\text{minimize }} & \quad \sum_{t = 1}^{T - 1} \Big(\sum_{i = 1}^{S} w_s^{(i)} \textbf{n}^{(i)}(s^{(i)}(q_t, v_t, a_t, u_t) - y_t^{(i)}) + \frac{1}{2} \lVert g(q_t, v_t, a_t, u_t) - \tau_t \rVert_{\textbf{diag}(w_{g})}^2 \Big) \\
    \text{subject to} & \quad v_t = (q_t - q_{t - 1}) / h\\
    & \quad a_t = (q_{t + 1} - 2 q_t + q_{t - 1}) / h^2 = (v_{t+1} - v_t) / h,
\end{aligned}
```

The constraints are handled implicitly. Velocities and accelerations are computed using finite-difference approximations from the configuration decision variables.

**Variables**
- $q \in \mathbf{R}^{n_q}$: configuration ```[qpos]```
- $v \in \mathbf{R}^{n_v}$: velocity ```[qvel]```
- $a \in \mathbf{R}^{n_v}$: accelerations ```[qacc]```
- $u \in \mathbf{R}^{n_u}$: action ```[ctrl]```
- $y \in \mathbf{R}^{n_s}$: sensor measurement ```[sensordata]```
- $\tau \in \mathbf{R}^{n_v}$: inverse dynamics force ```[qfrc_actuator]```
- $h \in \mathbf{R}_{+}\,$: time step ```[timestep]```
- $T$: estimation horizon
- $t$: discrete time step

**Models**
- $s : \mathbf{R}^{n_q} \times \mathbf{R}^{n_v} \times \mathbf{R}^{n_v} \times \mathbf{R}^{n_u} \rightarrow \mathbf{R}^{n_s}$: sensor model
- $g : \mathbf{R}^{n_q} \times \mathbf{R}^{n_v} \times \mathbf{R}^{n_v} \times \mathbf{R}^{n_u} \rightarrow \mathbf{R}^{n_v}$: inverse dynamics model
- $\textbf{n}: \mathbf{R}^{n} \rightarrow \mathbf{R}_{+}$: user-specified [convex norm](../mjpc/norm.h)

**Weights**

Sensors:
- $w_s \in \mathbf{R}_{+}^{S}$:

Forces:
- $w_g \in \mathbf{R}_{+}^{n_v}$:

Computed with user inputs:
- $\sigma_s \in \mathbf{R}_{+}^S$: sensor noise
- $\sigma_g \in \mathbf{R}_{+}^{n_v}$: process noise

Rescaled:
- $w_s^{(i)} = p / (\sigma_s^{(i)} \cdot n_s^{(i)} \cdot (T - 1))$
```math
p = \begin{cases} h^2 & {\texttt{settings.time\_scaling \& acceleration sensor}} \\ h & {\texttt{settings.time\_scaling \& velocity sensor}} \\ 1 & {\texttt{else}}\end{cases}
```
- $w_g^{(i)} = p / (\sigma_g^{(i)} \cdot n_v \cdot (T - 1))$
```math
p = \begin{cases} h^2 & {\texttt{settings.time\_scaling}} \\ 1 & {\texttt{else}} \end{cases}
```

## Cost Derivatives

**Residuals**

Sensor residual

$r_s = \Big(s^{(0)}(q_1, v_1, a_1, u_1) - y_1^{(0)}, \dots, s^{(i)}(q_t, v_t, a_t, u_t) - y_t^{(i)}, \dots, s^{(S)}(q_{T - 1}, v_{T - 1}, a_{T - 1}, u_{T - 1}) - y_{T -1 }^{(S)}\Big) \in \mathbf{R}^{n_s (T - 2)}$

Force residual

$r_g = \Big(g(q_1, v_1, a_1, u_1) - \tau_1, \dots, g(q_t, v_t, a_t, u_t) - \tau_t, \dots, g(q_{T - 1}, v_{T - 1}, a_{T - 1}, u_{T - 1}) - \tau_{T - 1}\Big) \in \mathbf{R}^{n_v (T - 2)}$

**Gradient**

The gradient of the cost $c$ with respect to the configuration trajectory $q_{0:T}$:

$d c/ d q_{0:T} = J_s^T N_s + J_g^T N_g$

where

- $J_s = dr_s / d q_{0:T} \in \mathbf{R}^{n_s (T - 2) \times n_v T}$: Jacobian of sensor residual with respect to configuration trajectory
- $J_g = dr_g / d q_{0:T} \in \mathbf{R}^{n_v (T - 2) \times n_v T}$: Jacobian of force residual with respect to configuration trajectory
- $N_s = \Big(w_s^{(0)} d \textbf{n}^{(0)} / d r_{s_1}^{(0)}, \dots, w_s^{(i)} d \textbf{n}^{(i)} / d r_{s_t}^{(i)}, \dots, w_s^{(S)}  d \textbf{n}^{(S)} / d r_{s_{T-1}}^{(S)} \Big)$
- $N_g = \Big(\textbf{diag}(w_g) r_{g_{1}}, \dots, \textbf{diag}(w_g) r_{g_{t}} \dots, \textbf{diag}(w_g) r_{g_{T-1}} \Big)$

**Hessian**

The [Gauss-Newton](https://en.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm) approximation of the cost Hessian:

$d^2 c / d q_{0:T}^2 \approx J_s^T N_{ss} J_s + J_g^T N_{gg} J_g$

where

- $N_{ss} = \textbf{diag}\Big(w_s^{(0)} d^2 \textbf{n}^{(0)} / d (r_{s_1}^{(0)})^2, \dots, w_s^{(i)} d^2 \textbf{n}^{(i)} / d (r_{s_t}^{(i)})^2, \dots, w_s^{(S)} d^2 \textbf{n}^{(S)} / d (r_{s_{T-1}}^{(S)})^2\Big)$
- $N_{gg} = \textbf{diag}\Big(\textbf{diag}(w_g) r_{g_1}, \dots, \textbf{diag}(w_g) r_{g_t} \dots, \textbf{diag}(w_g) r_{g_{T-1}} \Big)$

This approximation: 1) is computationally less expensive compared to computing the exact Hessian 2) ensures the matrix is [positive semidefinite](https://en.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm).

## Practical workflow

`Direct` estimates a configuration trajectory from sensor and force measurements. The configuration trajectory is the
optimizer's decision variable; the sensor trajectory is obtained by evaluating the MuJoCo sensor model at those
configurations. As a result, `sensor_measurement` and `configuration` are not interchangeable arrays, even when a model
contains joint-position sensors.

A minimal workflow is demonstrated by
[`DirectOptimize.Particle2D`](../mjpc/test/direct/direct_optimize_test.cc). The test:

1. Rolls out a model to obtain a reference configuration, actuator-force and sensor trajectory.
2. Copies the reference configuration into `configuration` and `configuration_previous` as the initial estimate.
3. Copies the rollout's `sensordata` into `sensor_measurement` and `qfrc_actuator` into `force_measurement`.
4. Perturbs `configuration`, then calls `Optimize()`.
5. Checks that the cost and gradient decrease and that the recovered configuration is close to the reference rollout.

The essential setup is:

```cpp
Direct optimizer(model, T);
mju_copy(optimizer.configuration.Data(), sim.qpos.Data(), nq * T);
mju_copy(optimizer.configuration_previous.Data(), sim.qpos.Data(), nq * T);
mju_copy(optimizer.force_measurement.Data(), sim.qfrc_actuator.Data(), nv * T);
mju_copy(optimizer.sensor_measurement.Data(), sim.sensor.Data(), ns * T);

// Perturb optimizer.configuration here to create the initial estimation error.
optimizer.Optimize();
```

After optimization, compare like with like:

- compare `configuration` with the expected `qpos` trajectory;
- compare `sensor_prediction` with `sensor_measurement` to inspect sensor residuals;
- compare `force_prediction` with `force_measurement` to inspect inverse-dynamics residuals.

A match between `sensor_prediction` and `sensor_measurement` only means that the estimated state reproduces the measured
sensor values. It does not in general imply element-wise equality between `configuration` (`qpos`) and
`sensor_measurement` (`sensordata`): sensors may have different dimensions, represent derived quantities, or observe
only part of the configuration.

There are also boundary conditions to keep in mind. At the first timestep only position-stage sensor residuals are used.
At the last timestep, position and velocity sensor residuals are disabled by default; they can be enabled with
`settings.last_step_position_sensors` and `settings.last_step_velocity_sensors`. Acceleration-stage sensor residuals are
not used at either boundary because finite-difference acceleration requires neighbouring configurations.

For a trajectory-tracking application, desired joint configurations should therefore not be treated as if they were
raw sensor measurements without first checking that the model's sensor definitions and the enabled residual terms encode
that objective. `Direct` is solving the sensor/force estimation problem defined in the [cost function](#cost-function),
including its process and sensor weights.

## Reference
[Physically-Consistent Sensor Fusion in Contact-Rich Behaviors](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=31c0842aa1d4808541a64014c24928123e1d949e).
Kendall Lowrey, Svetoslav Kolev, Yuval Tassa, Tom Erez, Emo Todorov. 2014.

## API
[Direct API](../mjpc/direct/direct.h)
