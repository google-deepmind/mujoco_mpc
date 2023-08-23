# Estimators

**Table of Contents**

- [Batch Estimator](#batch-estimator)
  - [Cost Function](#cost-function)
  - [Cost Derivatives](#cost-derivatives)
  - [Filter](#batch-filter)
  - [Settings](#batch-settings)
  - [API](#batch-api)
  - [Reference](#batch-reference)
- [Extended Kalman Filter](#kalman-filter)
  - [Algorithm](#kalman-algorithm)
  - [API](#kalman-api)
  - [Reference](#kalman-reference)
- [Unscented Kalman Filter](#unscented-filter)
  - [Algorithm](#unscented-algorithm)
  - [API](#unscented-api)
  - [Reference](#unscented-reference)

# Batch Estimator

## Cost Function
$$
\begin{align*}
    \underset{q_{0:T}}{\text{minimize }} & \quad \sum_{t = 1}^{T - 1} \Big(\sum_{i = 1}^{S} w_s^{(i)} \| s^{(i)}(q_t, v_t, a_t, u_t) - y_t^{(i)} \| + \frac{1}{2} \| g(q_t, v_t, a_t, u_t) - \tau_t \|_{\textbf{diag}(w_{g})}^2 \Big) \\
    \text{subject to} & \quad v_t = (q_t - q_{t - 1}) / h\\
    & \quad a_t = (q_{t + 1} - 2 q_t + q_{t - 1}) / h^2 = (v_{t+1} - v_t) / h,
\end{align*}
$$

The constraints are handled implicitly. Velocities and accelerations are computed using finite-difference approximations from the configuration decision variables.

**Variables**
- $q \in \mathbf{R}^{n_q}$: configuration
- $v \in \mathbf{R}^{n_v}$: velocity
- $a \in \mathbf{R}^{n_v}$: accelerations
- $u \in \mathbf{R}^{n_u}$: action
- $y \in \mathbf{R}^{n_s}$: sensor measurement
- $\tau \in \mathbf{R}^{n_v}$: inverse dynamics force
- $h \in \mathbf{R}_{+}\,$: time step
- $T$: estimation horizon
- $t$: discrete time step

**Models**
- $s : \mathbf{R}^{n_q} \times \mathbf{R}^{n_v} \times \mathbf{R}^{n_v} \times \mathbf{R}^{n_u} \rightarrow \mathbf{R}^{n_s}$: sensor model
- $g : \mathbf{R}^{n_q} \times \mathbf{R}^{n_v} \times \mathbf{R}^{n_v} \times \mathbf{R}^{n_u} \rightarrow \mathbf{R}^{n_v}$: inverse dynamics model

**Weights**

Sensors:
- $w_s \in \mathbf{R}_{+}^{S}$: 

Forces:
- $w_g \in \mathbf{R}_{+}^{n_v}$:

Computed with user inputs:
- $\sigma_s \in \mathbf{R}_{+}^S$: sensor noise
- $\sigma_g \in \mathbf{R}_{+}^{n_v}$: process noise

Rescaled:
- $ w_s^{(i)} = p / (\sigma_s^{(i)} \cdot n_s^{(i)} \cdot (T - 1))$
  - $p = \begin{cases} h^2 & {\texttt{settings.time\_scaling \& acceleration sensor}} \\ h & {\texttt{settings.time\_scaling \& velocity sensor}} \\ 1 & {\texttt{else}} \end{cases}$
- $ w_f^{(i)} = p / (\sigma_g^{(i)} \cdot n_v \cdot (T - 1))$
  - $p = \begin{cases} h^2 & {\texttt{settings.time\_scaling}} \\ 1 & {\texttt{else}} \end{cases}$

## Cost Derivatives

## Filter

An additional *prior* cost

$$
\frac{1}{2} (q_{0:T} - \bar{q}_{0:T})^T P (q_{0:T} - \bar{q}_{0:T})
$$

with $P \in \mathbf{S}_{++}^{n_v \times T}$ and overbar ($\bar{\,\,\,}$) denoting a reference configuration is added to the cost when filtering.

The prior weights
$$
P_{0:T} = \begin{bmatrix} P_{[0:t, 0:t]} & P_{[0:t, t+1:T]} \\ P_{[t+1:T, 0:t]} & P_{[t+1:T, t+1:T]} \end{bmatrix}
$$

are recursively updated by conditioning on a subset of the weights
$$
P_{t+1:T | 0:t} = P_{[t+1:T, t+1:T]} - P_{[t+1:T, 0:t]} P_{[0:t, 0:t]}^{-1} P_{[0:t, t+1:T]}
$$

## Settings

## Reference
[Physically-Consistent Sensor Fusion in Contact-Rich Behaviors](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=31c0842aa1d4808541a64014c24928123e1d949e).
Kendall Lowrey, Svetoslav Kolev, Yuval Tassa, Tom Erez, Emo Todorov. 2014.

# Extended Kalman Filter

## Algorithm

### Prediction Update

$$
\begin{aligned}
x_{t+1} &= f(x_t, u_t)\\
P_{t+1} &= A_t^T P_t A_t + Q
\end{aligned}
$$

### Measurement Update

$$
\begin{aligned}
x_t &\mathrel{+}= P_t C_t^T (C_t P_t C_t^T + R)^{-1} (y_t - s(x_t, u_t))\\
P_t &\mathrel{+}= P_t C_t^T (C_t P_t C_t^T + R)^{-1} (C_t * P_t)
\end{aligned}
$$

**Variables**
- $x \in \mathbf{R}^{n_q + n_v + n_a}$: state
- $u \in \mathbf{R}^{n_u}$: action
- $y \in \mathbf{R}^{n_s}$: sensor measurement
- $A \in \mathbf{R}^{(2 n_v + na) \times (2 n_v + na)} = \partial f / \partial x |_{x, u}$ : forward dynamics state Jacobian
- $C \in \mathbf{R}^{n_s \times (2 n_v + na)} = \partial f / \partial x|_{x, u}$ : sensor model state Jacobian
- $Q \in \mathbf{S}_{++}^{2 n_v + n_a}$: process noise
- $R \in \mathbf{S}_{++}^{n_s}$: measurement noise
- $P \in \mathbf{S}_{++}^{(2 n_v + n_a) \times (2 n_v + n_a)}$: state covariance

**Models**
- $f: \mathbf{R}^{n_q + n_v + n_a} \times \mathbf{R}^{n_u} \rightarrow \mathbf{R}^{n_q + n_v + n_a}$: forward dynamics
- $s: \mathbf{R}^{n_q + n_v + n_a} \times \mathbf{R}^{n_u} \rightarrow \mathbf{R}^{n_s}$: sensor model

## Reference
[A New Approach to Linear Filtering and Prediction Problems](https://www.cs.unc.edu/~welch/kalman/media/pdf/Kalman1960.pdf).
Rudolph Kalman. 1960.

[Application Of Statistical Filter Theory To The Optimal Estimation Of Position And Velocity On Board A Circumlunar Vehicle](https://archive.org/details/nasa_techdoc_19620006857/page/n31/mode/2up).
Gerald Smith, Stanley Schmidt, Leonard McGee. 1962.

# Unscented Kalman Filter

## Algorithm

**Sigma points**
$$
\begin{aligned}
z^{(0)} &= x_t\\
z^{(i)} &= x_t + \gamma \cdot \textbf{cholesky}(P_t)^{(i)} \quad \text{for} \quad i = 1, \dots, n_{dx}\\
z^{(n_{dx} + i)} &= x_t - \gamma \cdot \textbf{cholesky}(P_t)^{(i)} \quad \text{for} \quad i = 1, \dots, n_{dx}
\end{aligned}
$$
where $\textbf{cholesky}(\cdot)^{(i)}$ is the $i$-th column of a postive definite matrix's Cholesky factorization.

**Sigma point evaluation**
$$
\begin{aligned}
\hat{x}_{t+1}^{(i)} &= f(z^{(i)}, u_t) \quad \text{for} \quad i = 0, \dots, 2n_{dx}\\
\hat{y}_t^{(i)} &= h(z^{(i)}, u_t) \quad \text{for} \quad i = 0, \dots, 2n_{dx}
\end{aligned}
$$

**Sigma point means**
$$
\begin{aligned}
\bar{x}_{t+1} &= \sum \limits_{i = 0}^{2 n_{dx}} w_m^{(i)} \cdot \hat{x}_{t+1}^{(i)}\\
\bar{y}_{t} &= \sum \limits_{i = 0}^{2 n_{dx}} w_m^{(i)} \cdot \hat{y}_t^{(i)}
\end{aligned}
$$

Note: states containing quaternions are corrected by computing a quaternion "average", [Averaging Quaternions](http://www.acsu.buffalo.edu/~johnc/ave_quat07.pdf).

**Sigma point covariances**
$$
\begin{aligned}
\Sigma_{xx} &= \sum \limits_{i = 0}^{2 n_{dx}} w_c^{(i)} \cdot (\hat{x}_{t+1}^{(i)} - \bar{x}_{t+1}) (\hat{x}_{t+1}^{(i)} - \bar{x}_{t+1})^T + Q\\
\Sigma_{ss} &= \sum \limits_{i = 0}^{2 n_{dx}} w_c^{(i)} \cdot (\hat{y}_{t}^{(i)} - \bar{y}_{t}) (\hat{y}_{t}^{(i)} - \bar{y}_{t})^T + R\\
\Sigma_{xs} &= \sum \limits_{i = 0}^{2 n_{dx}} w_c^{(i)} \cdot (\hat{x}_{t+1}^{(i)} - \bar{x}_{t+1}) (\hat{y}_{t}^{(i)} - \bar{y}_{t})^T\\
\end{aligned}
$$

**Update**
$$
\begin{aligned}
x_{t+1} &= \bar{x}_{t+1} + \Sigma_{xx} \Sigma_{ss}^{-1} (y_t - \bar{y}_t)\\
P_{t+1} &= \Sigma_{xx} - \Sigma_{xs} \Sigma_{yy}^{-1} \Sigma_{xs}^T
\end{aligned}
$$
### Dimensions
- $n_x = n_q + n_v + na$: state dimension
- $n_{dx} = 2 n_v + na$: state derivative dimension

### Weights
- $\lambda = n_{dx} \cdot (\alpha^2 - 1)$
- $\gamma = \sqrt{n_{dx} + \lambda}$: sigma point step size
- $w_m^{(0)} = \lambda / (n_{dx} + \lambda)$: mean weight 0
- $w_c^{(0)} = w_m^{(0)} + 1 - \alpha^2 + \beta$: covariance weight 0
- $w_m^{(i)} = w_c^{(i)} = 1 / (2 (n_{dx} + \lambda))$: weights

### Settings
- $\alpha$: 
- $\beta$:

## Reference
[Unscented Filtering and Nonlinear Estimation](https://www.cs.ubc.ca/~murphyk/Papers/Julier_Uhlmann_mar04.pdf).
Simon Julier, Jeffrey Uhulmann. 2004.