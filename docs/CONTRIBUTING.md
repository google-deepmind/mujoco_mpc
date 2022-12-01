# Contributing to MuJoCo MPC

We want MuJoCo MPC to be a true community-driven effort that continuously improves and grows over time for the benefit of the entire research community. As such, we welcome contributions such as:

- Adding new optimizers and estimators
- Adding new tasks and models

Note that MJPC follows [Google's Open Source Community Guidelines](https://opensource.google/conduct/).

## How to contribute

Whether you want to fix an existing issue or work on adding new tasks and models, please get in touch with us first (ideally _before_ starting work if it's something major) by opening a new [issue](https://github.com/deepmind/mujoco_mpc/issues). Coordinating up front makes it much easier to avoid frustration later on.

Once we reach an agreement on the proposed change, please submit a [pull request](https://github.com/deepmind/mujoco_mpc/pulls) (PR) so that we can review your implementation.

## Code Style

This code adheres to the [Google style](https://google.github.io/styleguide/).

## Unit Tests

Before submitting your PR, you can test your change locally by invoking ctest:

```
cd build
ctest --output-on-failure .
```

This same test will run on GitHub CI once you open your PR.

## Contributor License Agreement

Contributions to this project must be accompanied by a Contributor License Agreement (CLA). You (or your employer) retain the copyright to your contribution; this simply gives us permission to use and redistribute your contributions as part of the project. Head over to <https://cla.developers.google.com/> to see your current agreements on file or to sign a new one.

You generally only need to submit a CLA once, so if you've already submitted one (even if it was for a different project), you probably don't need to do it again.
