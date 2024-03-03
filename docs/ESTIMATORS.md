# Estimators

**Table of Contents**

- [Extended Kalman Filter](#extended-kalman-filter)
  - [Algorithm](#kalman-algorithm)
  - [Reference](#kalman-reference)
  - [API](#kalman-api)
- [Unscented Kalman Filter](#unscented-kalman-filter)
  - [Algorithm](#unscented-algorithm)
  - [Reference](#unscented-reference)
  - [API](#unscented-api)
- [Batch Estimator](#batch-estimator)
  - [Prior](#batch-prior)
  - [Reference](#reference)


# Extended Kalman Filter

## Kalman Algorithm

### Prediction Update

```math
\begin{aligned}
x_{t+1} &= f(x_t, u_t)\\
P_{t+1} &= A_t P_t A_t^T + Q
\end{aligned}
```

### Measurement Update

```math
\begin{aligned}
x_t &\mathrel{+}= P_t C_t^T (C_t P_t C_t^T + R)^{-1} (y_t - s(x_t, u_t))\\
P_t &\mathrel{+}= P_t C_t^T (C_t P_t C_t^T + R)^{-1} (C_t * P_t)
\end{aligned}
```

**Variables**
- $x \in \mathbf{R}^{n_q + n_v + n_a}$: state
- $u \in \mathbf{R}^{n_u}$: action
- $y \in \mathbf{R}^{n_s}$: sensor measurement
- $A \in \mathbf{R}^{(2 n_v + na) \times (2 n_v + na)} = \partial f / \partial x |_{x, u}$ : forward dynamics state Jacobian
- $C \in \mathbf{R}^{n_s \times (2 n_v + na)} = \partial f / \partial u|_{x, u}$ : sensor model state Jacobian
- $Q \in \mathbf{S}_{++}^{2 n_v + n_a}$: process noise
- $R \in \mathbf{S}_{++}^{n_s}$: measurement noise
- $P \in \mathbf{S}_{++}^{(2 n_v + n_a)}$: state covariance

**Models**
- $f: \mathbf{R}^{n_q + n_v + n_a} \times \mathbf{R}^{n_u} \rightarrow \mathbf{R}^{n_q + n_v + n_a}$: forward dynamics
- $s: \mathbf{R}^{n_q + n_v + n_a} \times \mathbf{R}^{n_u} \rightarrow \mathbf{R}^{n_s}$: sensor model

## Kalman Reference
[A New Approach to Linear Filtering and Prediction Problems](https://www.cs.unc.edu/~welch/kalman/media/pdf/Kalman1960.pdf).
Rudolph Kalman. 1960.

[Application Of Statistical Filter Theory To The Optimal Estimation Of Position And Velocity On Board A Circumlunar Vehicle](https://archive.org/details/nasa_techdoc_19620006857/page/n31/mode/2up).
Gerald Smith, Stanley Schmidt, Leonard McGee. 1962.

## Kalman API
[Kalman API](../mjpc/estimators/kalman.h)

# Unscented Kalman Filter

## Unscented Algorithm

**Sigma points**

```math
\begin{aligned}
z^{(0)} &= x_t\\
z^{(i)} &= x_t + \gamma \cdot \textbf{cholesky}(P_t)^{(i)} \quad \text{for} \quad i = 1, \dots, n_{dx}\\
z^{(n_{dx} + i)} &= x_t - \gamma \cdot \textbf{cholesky}(P_t)^{(i)} \quad \text{for} \quad i = 1, \dots, n_{dx}
\end{aligned}
```
where $\textbf{cholesky}(\cdot)^{(i)}$ is the $i$-th column of a postive definite matrix's Cholesky factorization.

**Sigma point evaluation**

```math
\begin{aligned}
\hat{x}_{t+1}^{(i)} &= f(z^{(i)}, u_t) \quad \text{for} \quad i = 0, \dots, 2n_{dx}\\
\hat{y}_t^{(i)} &= h(z^{(i)}, u_t) \quad \text{for} \quad i = 0, \dots, 2n_{dx}
\end{aligned}
```

**Sigma point means**

```math
\begin{aligned}
\bar{x}_{t+1} &= \sum \limits_{i = 0}^{2 n_{dx}} w_m^{(i)} \cdot \hat{x}_{t+1}^{(i)}\\
\bar{y}_{t} &= \sum \limits_{i = 0}^{2 n_{dx}} w_m^{(i)} \cdot \hat{y}_t^{(i)}
\end{aligned}
```

Note: states containing quaternions are corrected by computing a quaternion "average": [Averaging Quaternions](http://www.acsu.buffalo.edu/~johnc/ave_quat07.pdf).

**Sigma point covariances**

```math
\begin{aligned}
\Sigma_{xx} &= \sum \limits_{i = 0}^{2 n_{dx}} w_c^{(i)} \cdot (\hat{x}_{t+1}^{(i)} - \bar{x}_{t+1}) (\hat{x}_{t+1}^{(i)} - \bar{x}_{t+1})^T + Q\\
\Sigma_{ss} &= \sum \limits_{i = 0}^{2 n_{dx}} w_c^{(i)} \cdot (\hat{y}_{t}^{(i)} - \bar{y}_{t}) (\hat{y}_{t}^{(i)} - \bar{y}_{t})^T + R\\
\Sigma_{xs} &= \sum \limits_{i = 0}^{2 n_{dx}} w_c^{(i)} \cdot (\hat{x}_{t+1}^{(i)} - \bar{x}_{t+1}) (\hat{y}_{t}^{(i)} - \bar{y}_{t})^T\\
\end{aligned}
```

**Update**

```math
\begin{aligned}
x_{t+1} &= \bar{x}_{t+1} + \Sigma_{xx} \Sigma_{ss}^{-1} (y_t - \bar{y}_t) \\
P_{t+1} &= \Sigma_{xx} - \Sigma_{xs} \Sigma_{ss}^{-1} \Sigma_{xs}^T
\end{aligned}
```

### Dimensions
- $n_x = n_q + n_v + n_a$: state dimension
- $n_{dx} = 2 n_v + n_a$: state derivative dimension

**Variables**
- $x \in \mathbf{R}^{n_x}$: state
- $u \in \mathbf{R}^{n_u}$: action
- $y \in \mathbf{R}^{n_s}$: sensor measurement
- $Q \in \mathbf{S}^{n_{dx}}_{+}$: process noise
- $R \in \mathbf{S}_{++}^{n_s}$: measurement noise
- $P \in \mathbf{S}^{n_{dx}}_{++}$: state covariance

**Models**
- $f: \mathbf{R}^{n_q + n_v + n_a} \times \mathbf{R}^{n_u} \rightarrow \mathbf{R}^{n_q + n_v + n_a}$: forward dynamics
- $s: \mathbf{R}^{n_q + n_v + n_a} \times \mathbf{R}^{n_u} \rightarrow \mathbf{R}^{n_s}$: sensor model

### Weights
- $\lambda = n_{dx} \cdot (\alpha^2 - 1)$
- $\gamma = \sqrt{n_{dx} + \lambda}$: sigma point step size
- $w_m^{(0)} = \lambda / (n_{dx} + \lambda)$: mean weight
- $w_c^{(0)} = w_m^{(0)} + 1 - \alpha^2 + \beta$: covariance weight
- $w_m^{(i)} = w_c^{(i)} = 1 / (2 (n_{dx} + \lambda))$: weights

### Unscented Settings
- $\alpha$: controls spread of sigma points around mean
- $\beta$: encodes prior information (Gaussian: $\beta = 2$)

## Unscented Reference
[Unscented Filtering and Nonlinear Estimation](https://www.cs.ubc.ca/~murphyk/Papers/Julier_Uhlmann_mar04.pdf).
Simon Julier, Jeffrey Uhulmann. 2004.

## Unscented API
[Unscented API](../mjpc/estimators/unscented.h)

# Batch Estimator

This estimator utilizes the [direct optimizer](DIRECT.md) to formulate a [fixed-lag smoother](https://en.wikipedia.org/wiki/Kalman_filter#Fixed-lag_smoother).

## Batch Prior

An additional *prior* cost

```math
\frac{1}{2} (q_{0:T} - \bar{q}_{0:T})^T P (q_{0:T} - \bar{q}_{0:T})
```

with $P \in \mathbf{S}_{++}^{n_v  T}$ and overbar denoting a reference configuration is added to the cost.

The prior weights
```math
P_{0:T} = \begin{bmatrix} P_{[0:t, 0:t]} & P_{[0:t, t+1:T]} \\ P_{[t+1:T, 0:t]} & P_{[t+1:T, t+1:T]} \end{bmatrix}
```

are recursively updated by conditioning on a subset of the weights
```math
P_{t+1:T | 0:t} = P_{[t+1:T, t+1:T]} - P_{[t+1:T, 0:t]} P_{[0:t, 0:t]}^{-1} P_{[0:t, t+1:T]}
```

## Batch Reference
[Physically-Consistent Sensor Fusion in Contact-Rich Behaviors](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=31c0842aa1d4808541a64014c24928123e1d949e).
Kendall Lowrey, Svetoslav Kolev, Yuval Tassa, Tom Erez, Emo Todorov. 2014.

[Batch API](../mjpc/estimators/batch.h)