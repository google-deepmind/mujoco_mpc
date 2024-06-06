//
// Created by Moritz Meser on 21.05.24.
//

#include "H1_kitchen.h"
#include <string>
# include <limits>
#include <cmath>
#include <algorithm>

#include "mujoco/mujoco.h"
#include "mjpc/utilities.h"

#include "mjpc/tasks/humanoid_bench/utility/dm_control_utils_rewards.h"

namespace mjpc {
// ----------------- Residuals for humanoid_bench kitchen task ---------------- //
// ------------------------------------------------------------------------------- //
    void H1_kitchen::ResidualFn::Residual(const mjModel *model, const mjData *data,
                                          double *residual) const {
        // the reward of this task is a count of how many objects are at the target location

        // ----- set parameters ----- //
        double const BONUS_THRESH = 0.3;
        // ----- initialize reward ----- //
        int reward = 0;


        bool all_completed_so_far = true;
        // ----- loop over all tasks ----- //
        for (const std::string &task: task_->tasks_to_complete_) {
            double distance = mjpc::H1_kitchen::CalculateDistance(task, data, model->nq - 22);

            // ----- check if the object is at the target location ----- //
            bool completed = distance < BONUS_THRESH;
            if (completed && (all_completed_so_far || !task_->ENFORCE_TASK_ORDER)) {
                reward++;
                all_completed_so_far = all_completed_so_far && completed;
            }
        }

        // ----- residuals ----- //
        residual[0] = std::exp(-reward);
    }

// -------- Transition for humanoid_bench kitchen task -------- //
// ------------------------------------------------------------ //
    void H1_kitchen::TransitionLocked(mjModel *model, mjData *data) {
        if (!REMOVE_TASKS_WHEN_COMPLETE) {
            return;
        }
        double const BONUS_THRESH = 0.3;

        bool all_completed_so_far = true;
        std::vector<std::string> completed_tasks;
        // ----- loop over all tasks ----- //
        for (const std::string &task: tasks_to_complete_) {
            double distance = CalculateDistance(task, data, model->nq - 22);
            // ----- check if the object is at the target location ----- //
            bool completed = distance < BONUS_THRESH;
            if (completed && (all_completed_so_far || !ENFORCE_TASK_ORDER)) {
                completed_tasks.push_back(task);
                all_completed_so_far = all_completed_so_far && completed;
            }
        }
        // remove all completed tasks
        for (const std::string &completedTask: completed_tasks) {
            tasks_to_complete_.erase(
                    std::remove(tasks_to_complete_.begin(), tasks_to_complete_.end(), completedTask),
                    tasks_to_complete_.end());
            printf("Completed task: %s\n", completedTask.c_str());
        }

    }

    double H1_kitchen::CalculateDistance(const std::string &task, const mjData *data, int robot_dof) {
        std::map<std::string, std::vector<double>> obs_element_goals = {
                {"bottom burner", {-0.88, -0.01}},
                {"top burner",    {-0.92, -0.01}},
                {"light switch",  {-0.69, -0.05}},
                {"slide cabinet", {0.37}},
                {"hinge cabinet", {0.0,   1.45}},
                {"microwave",     {-0.75}},
                {"kettle",        {-0.23, 0.75, 1, 0.99, 0.0, 0.0, -0.06}}
        };
        std::map<std::string, std::vector<int>> obs_element_indices = {
                {"bottom burner", {2,  3}},
                {"top burner",    {6,  7}},
                {"light switch",  {8,  9}},
                {"slide cabinet", {10}},
                {"hinge cabinet", {11, 12}},
                {"microwave",     {13}},
                {"kettle",        {14, 15, 16, 17, 18, 19, 20}}
        };

        std::vector<int> obs_element_index = obs_element_indices.at(task);
        double *observation = data->qpos + robot_dof + obs_element_index[0];

        std::vector<double> obs_element_goal = obs_element_goals.at(task);

        double distance = 0;
        for (int i = 0; i < obs_element_goal.size(); i++) {
            distance += std::pow(observation[i] - obs_element_goal[i], 2);
        }
        return std::sqrt(distance);
    }

}  // namespace mjpc
