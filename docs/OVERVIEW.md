# Predictive Control

**Table of Contents**

- [Predictive Control](#predictive-control)
  - [Cost Function](#cost-function)
    - [Risk (in)sensitive costs](#risk-insensitive-costs)
  - [Cost Derivatives](#cost-derivatives)
    - [Risk (in)sensitive derivatives](#risk-insensitive-derivatives)
  - [Task Specification](#task-specification)
    - [Settings](#settings)
    - [Residual Specification](#residual-specification)
      - [Examples](#examples)
    - [Task Transitions](#task-transitions)
  - [Planners](#planners)

**MuJoCo MPC (MJPC)** is a tool for the design and experimentation of predictive control algorithms. Such algorithms solve planning problems of the form:

$$
\begin{align*}
    \underset{s_{1:{T}}, a_{1:T}}{\text{minimize }} & \, \sum \limits_{t = 0}^{T} c(s_t, a_t),\\
    \text{s.t.} & \, s_{t+1} = m(s_t, a_t),\\
    & (s_0 \, \text{given}),
\end{align*}
$$

where the goal is to optimize state $s \in \mathbf{S}$, action $a \in \mathbf{A}$ trajectories subject to a model $m : \mathbf{S} \times \mathbf{A} \rightarrow \mathbf{S}$ specifying the transition dynamics of the system. The minimization objective is the total return, which is the sum of step-wise values $c : \mathbf{S} \times \mathbf{A} \rightarrow \mathbf{R}_{+}$ along the trajectory from the current time until the horizon $T$.

## Cost Function

Behaviors are designed by specifying a per-timestep cost $c$ of the form:

$$
c(s, a) = \sum_i^N w_i \cdot \text{n}_i\Big(\textbf{r}_i(s, a)\Big)
$$

The cost is a sum of $N$ terms, each comprising:

- A nonnegative scalar weight $w \ge 0$ that determines the relative importance of this term.
- A norm function $\rm n(\cdot)$ taking a vector and returning a non-negative scalar that achieves its minimum at $\bf 0$.
- The residual $\bf r$ is a vector of elements that is "small when the task is solved". Residuals are implemented as custom [MuJoCo sensors](https://mujoco.readthedocs.io/en/latest/XMLreference.html#sensor).

### Risk (in)sensitive costs

We additionally provide a convenient family of exponential transformations $J(s, a; R) = f(c(s, a), R)$ that can be applied to the cost function to model risk-seeking and risk-averse behaviors. This family is parameterized by a scalar $R \in \mathbf{R}$ (our default is $R=0$) and is given by:

$$
f(x, R) = \frac{e^{R \cdot x} - 1}{R}.
$$

The mapping, $f : \mathbf{R}_ {+} \times \mathbf{R} \rightarrow \mathbf{R}_ {+}$, has the following properties:

- It is defined and smooth for any $R$
- At $R=0$, it reduces to the identity $f(x; 0) = x$
- 0 is a fixed point of the function: $f(0; R) = 0$
- It is nonnegative: $f(x; R) \geq 0$ if $x \geq 0$
- It is strictly monotonic: $f(x; R) \gt f(y; R)$ if $x > y$
- The value of $f(x;R)$ has the same unit as $x$

$R=0$ can be interpreted as risk neutral, $R>0$ as risk-averse, and $R<0$ as risk-seeking. See [Whittle et. al, 1981](https://www.jstor.org/stable/1426972) for more details on risk sensitive control. Furthermore, when $R < 0$, this transformation creates cost functions that are equivalent to the rewards commonly used in the Reinforcement Learning literature. For example using the quadratic norm $\text{n}(\textbf{r}) = \textbf{r}^T Q \textbf{r}$ for some symmetric positive definite (SPD) matrix $Q=\Sigma^{-1}$ and a risk parameter $R=-1$ leads to an inverted-Gaussian cost $c=1 - e^{-\textbf{r}^T \Sigma^{-1} \textbf{r}}$, whose minimisation is equivalent to performing maximum likelihood estimation for the Gaussian.

## Cost Derivatives

Many planners (e.g., Gradient Descent and [iLQG](https://homes.cs.washington.edu/~todorov/papers/LiICINCO04.pdf)) utilize derivative information in order to optimize a policy.

Gradients are computed exactly:

$$ \frac{\partial c}{\partial s} = \sum \limits_{i} w_i \cdot \frac{\partial \text{n}_i}{\partial \textbf{r}} \frac{\partial \textbf{r}_i}{\partial s},$$

$$ \frac{\partial c}{\partial a} = \sum \limits_{i} w_i \cdot \frac{\partial \text{n}_i}{\partial \textbf{r}} \frac{\partial \textbf{r}_i}{\partial a}.$$

Hessians are approximated:

$$ \frac{\partial^2 c}{\partial s^2} \approx \sum \limits_{i} w_i \cdot \Big(\frac{\partial \textbf{r}_i}{\partial s}\Big)^T\frac{\partial^2 \text{n}_i}{\partial \textbf{r}^2} \frac{\partial \textbf{r}_i}{\partial s},$$

$$ \frac{\partial^2 c}{\partial a^2} \approx \sum \limits_{i} w_i \cdot \Big(\frac{\partial \textbf{r}_i}{\partial a}\Big)^T\frac{\partial^2 \text{n}_i}{\partial \textbf{r}^2} \frac{\partial \textbf{r}_i}{\partial a},$$

$$ \frac{\partial^2 c}{\partial s \, \partial a} \approx \sum \limits_{i} w_i \cdot \Big(\frac{\partial \textbf{r}_i}{\partial s}\Big)^T \frac{\partial^2 \text{n}_i}{\partial \textbf{r}^2} \frac{\partial \textbf{r}_i}{\partial a},$$

via [Gauss-Newton](https://en.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm).

Derivatives of $\text{norm}$, i.e., $\frac{\partial \text{norm}}{\partial \text{residual}}$, $\frac{\partial^2 \text{norm}}{\partial \text{residual}^2}$, are computed [analytically](../mjpc/norm.cc).

The $\text{residual}$ Jacobians, i.e., $\frac{\partial \text{residual}}{\partial \text{state}}$, $\frac{\partial \text{residual}}{\partial \text{action}}$, are computed efficiently using MuJoCo's efficient [finite-difference utility](https://mujoco.readthedocs.io/en/latest/APIreference.html#mjd-transitionfd).

### Risk (in)sensitive derivatives

Likewise, we also provide derivaties for the risk (in)sensitive cost functions.

Gradients are computed exactly:

$$ \frac{\partial J}{\partial s} = e^{R \cdot c} \cdot \frac{\partial c}{\partial s},$$

$$ \frac{\partial J}{\partial a} = e^{R \cdot c} \cdot \frac{\partial c}{\partial a}.$$

Hessians are approximated:

$$ \frac{\partial^2 J}{\partial s^2} \approx e^{R \cdot c} \cdot \frac{\partial^2 c}{\partial s^2} + R \cdot e^{R \cdot c} \cdot \Big(\frac{\partial c}{\partial s}\Big)^T \frac{\partial c}{\partial s},$$

$$ \frac{\partial^2 J}{\partial a^2} \approx e^{R \cdot c} \cdot \frac{\partial^2 c}{\partial a^2} + R \cdot e^{R \cdot c} \cdot \Big(\frac{\partial c}{\partial a}\Big)^T \frac{\partial c}{\partial a},$$

$$ \frac{\partial^2 J}{\partial s \, \partial a} \approx e^{R \cdot c} \cdot \frac{\partial^2 c}{\partial s \partial a} + R \cdot e^{R \cdot c} \cdot \Big(\frac{\partial c}{\partial s}\Big)^T \frac{\partial c}{\partial a},$$

as before, via Gauss-Newton.

## Task Specification

In order to create a new task, at least two files are needed:

1. An [MJCF](https://mujoco.readthedocs.io/en/latest/XMLreference.html?highlight=xml#xml-reference) file that describes the simulated model and the task parameters.
2. A C++ file that implements the computation of the residual.

It's considered good practice to separate the physics model and task description into two XML files, and include the model file into the task file.

### Settings

Configuration parameters for the `Planner` and `Agent` are specified using MJCF [custom numeric](https://mujoco.readthedocs.io/en/latest/XMLreference.html#custom-numeric) elements.

```c++
<custom>
    <numeric
        name="[setting_name]"
        data="[setting_value(s)...]"
    />
</custom>
```

Values in brackets should be replaced by the designer.

- `setting_name`: String specifying setting name
- `setting_value(s)`: Real(s) specifying setting value(s)

`Agent` settings can be specified by prepending `agent_` for corresponding class members.

`Planner` settings can similarly be specified by prepending the corresponding optimizer name, (e.g., `sampling_`, `gradient_`, `ilqg_`).

It is also possible to create GUI elements for  parameters that are passed to the residual function.  These are specified by the prefix `residual_`, when the suffix will be the display name of the slider:

```c++
<custom>
    <numeric
        name="residual_[name]"
        data="[value] [lower_bound] [upper_bound]"
    />
</custom>
```

- `value`: Real number for the initial value of the parameter.
- `lower_bound`: Real specifying lower bound of GUI slider.
- `upper_bound`: Real specifying upper bound of GUI slider.

### Residual Specification

As mentioned above, the cost is a sum of terms, each computed as a (scalar) norm of a (vector) residual. Each term is defined as a [user sensor](https://mujoco.readthedocs.io/en/latest/XMLreference.html#sensor-user). The sensor values constitute the residual vector (implemented by the residual function, see below). The norm for each term is defined by the [`user` attribute](https://mujoco.readthedocs.io/en/latest/modeling.html#cuser) of the respective user sensor, according to the following format:

```c++
<sensor>
    <user
        name="[term_name]"
        dim="[residual_dimension]"
        user="
            [norm_type]
            [weight]
            [weight_lower_bound]
            [weight_upper_bound]
            [norm_parameters...]"
    />
</sensor>
```

- `term_name`: String describing term name (to appear in GUI).
- `residual_dimension`: number of elements in residual.
- `norm_type`: Integer corresponding to the NormType enum, see [norms](../mjpc/norm.h).
- `weight`: Real number specifying weight value of the cost.
- `weight_lower_bound`, `weight_upper_bound`: Real numbers specifying lower and upper bounds on $\text{weight}$, used for configuring the GUI slider.
- `parameters`: Optional real numbers specifying $\text{norm}$-specific parameters, see [norm implementation](../mjpc/norm.cc).

The user sensors defining the residuals should be declared first. Therefore, all sensors should be defined in the task file. After the cost terms are specified, more sensors can be defined to specify the task, e.g., when the residual implementation relies on reading a sensor value. If a sensor named `trace%i` exists, the GUI will use it to draw the traces of the tentative and nominal trajectories.

Computing the residual vectors is specified via a C++ function with the following interface:

```c++
void residual(
    double* residual,
    const double* parameters,
    const mjModel* model,
    const mjData* data)
```

- `residual`: The function's output. Each element in the double array should be set by this function, corresponding to the ordering and sizes provided by the user sensors in the task XML. See example below.
- `parameters`: Array of doubles, this input is comprised of elements specified using `residual_`
- `model`: the task's `mjModel`.
- `data`: the task's `mjData`. Marked `const`, because this function should not change mjData values.

*NOTE*: User sensors corresponding to cost terms should be specified before any other sensors.

#### Examples

As an example, consider these snippets specifying the Swimmer task:

```xml
<sensor>
  <user name="Control" dim="5" user="0 0.1 0 0.1" />
  <user name="Distance" dim="2" user="2 30 0 100 0.04" />
  <framepos name="trace0" objtype="geom" objname="nose"/>
  <framepos name="nose" objtype="geom" objname="nose"/>
  <framepos name="target" objtype="body" objname="target"/>
</sensor>
```
```c++
void Swimmer::Residual(double* residual, const double* parameters,
                        const mjModel* model, const mjData* data) {
  mju_copy(residual, data->ctrl, model->nu);
  double* target = mjpc::SensorByName(model, data, "target");
  double* nose = mjpc::SensorByName(model, data, "nose");
  mju_sub(residual + model->nu, nose, target, 2);
}
```

The swimmer's cost has two terms:

1. The first user sensor declares a quadratic norm (index 0) on a vector of 5 entries with weight 0.1.  The first line of the `Residual` function copies the control into this memory space.
2. The second user sensor declares an L2 norm (index 2) with weight 30 and one parameters: 0.04.  The `Residual` function computes the XY coordinates of the target's position relative to the nose frame, and writes these two numbers to the remaining 2 entries of the residual vector.

The repository includes additional example tasks:

- Humanoid [Stand](../mjpc/tasks/humanoid/task_stand.xml) | [Walk](../mjpc/tasks/humanoid/task_walk.xml)
- Quadruped [Terrain](../mjpc/tasks/quadruped/task_hill.xml) | [Flat](../mjpc/tasks/quadruped/task_flat.xml)
- [Walker](../mjpc/tasks/walker/task.xml)
- [In-Hand Manipulation](../mjpc/tasks/hand/task.xml)
- [Cart-Pole](../mjpc/tasks/cartpole/task.xml)
- [Particle](../mjpc/tasks/particle/task.xml)
- [Quadrotor](../mjpc/tasks/quadrotor/task.xml)

### Task Transitions

The `task` [class](../mjpc/task.h) supports *transitions* for subgoal-reaching tasks.  The transition function is called at every step, and is meant to allow the task specification to adapt to the agent's behavior (e.g., moving the target whenever the agent reaches it).  This function should test whether a transition condition is fulfilled, and at that event it should mutate the task specification.  It has the following interface:

```c++
int Swimmer::Transition(
  int state,
  const mjModel* model,
  mjData* data)
```

- `state`: an integer marking the current task mode.  This is useful when the sub-goals are enumerable (e.g., when cycling between keyframes).  If the user uses this feature, the function should return a new state upon transition.
- `model`: the task's `mjModel`.  It is marked `const` because changes to it here will only affect the GUI and will not affect the planner's copies.
- `data`: the task's `mjData`.  A transition should mutate fields of `mjData`, see below.

Because we only sync the `mjData` [state](https://mujoco.readthedocs.io/en/latest/computation.html?highlight=state#the-state) between the GUI and the agent's planner, tasks that need transitions should use `mocap` [fields](https://mujoco.readthedocs.io/en/latest/modeling.html#cmocap) to specify goals.  For code examples, see the `Transition` functions in these example tasks: [Swimmer](../mjpc/tasks/swimmer/swimmer.cc) (relocating the target), [Quadruped](../mjpc/tasks/quadruped/quadruped.cc) (iterating along a fixed set of targets) and [Hand](../mjpc/tasks/hand/hand.cc) (recovering when the cube is dropped).

## Planners

The purpose of `Planner` is to find improved policies using numerical optimization.

This library includes three planners that use different techniques to perform this search:

- **Zero-Order**
  - Random Search
  - derivative free
- **First-Order**
  - Gradient Descent
  - requires gradients
- **Second-Order**
  - Iterative Linear Quadratic Gaussian (iLQG)
  - requires gradients and Hessians
