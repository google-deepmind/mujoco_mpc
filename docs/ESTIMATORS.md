# Estimators

**Table of Contents**

- [Batch Estimator](#batch-estimator)
  - [Cost Function](#cost-function)
  - [Filter](#batch-filter)
  - [Settings](#batch-settings)
  - [Reference](#batch-reference)
- [Extended Kalman Filter](#kalman-filter)
  - [Algorithm](#kalman-algorithm)
  - [Reference](#kalman-reference)
- [Unscented Kalman Filter](#unscented-filter)
  - [Algorithm](#unscented-algorithm)
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

### Measurement Update

### Settings

## Reference
[A New Approach to Linear Filtering and Prediction Problems](https://www.cs.unc.edu/~welch/kalman/media/pdf/Kalman1960.pdf).
Rudolph Kalman. 1960.

[Application Of Statistical Filter Theory To The Optimal Estimation Of Position And Velocity On Board A Circumlunar Vehicle](https://archive.org/details/nasa_techdoc_19620006857/page/n31/mode/2up).
Gerald Smith, Stanley Schmidt, Leonard McGee. 1962.

# Unscented Kalman Filter

## Algorithm

### Settings

## Reference
[Unscented Filtering and Nonlinear Estimation](https://www.cs.ubc.ca/~murphyk/Papers/Julier_Uhlmann_mar04.pdf).
Simon Julier, Jeffrey Uhulmann. 2004.