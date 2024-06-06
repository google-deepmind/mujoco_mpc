// Copyright 2022 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mjpc/tasks/tasks.h"

#include <memory>
#include <vector>

#include "mjpc/task.h"
#include "mjpc/tasks/acrobot/acrobot.h"
#include "mjpc/tasks/allegro/allegro.h"
#include "mjpc/tasks/bimanual/handover/handover.h"
#include "mjpc/tasks/bimanual/reorient/reorient.h"
#include "mjpc/tasks/cartpole/cartpole.h"
#include "mjpc/tasks/fingers/fingers.h"
#include "mjpc/tasks/humanoid/stand/stand.h"
#include "mjpc/tasks/humanoid/tracking/tracking.h"
#include "mjpc/tasks/humanoid/walk/walk.h"
#include "mjpc/tasks/manipulation/manipulation.h"
// DEEPMIND INTERNAL IMPORT
#include "mjpc/tasks/op3/stand.h"
#include "mjpc/tasks/panda/panda.h"
#include "mjpc/tasks/particle/particle.h"
#include "mjpc/tasks/quadrotor/quadrotor.h"
#include "mjpc/tasks/quadruped/quadruped.h"
#include "mjpc/tasks/rubik/solve.h"
#include "mjpc/tasks/shadow_reorient/hand.h"
#include "mjpc/tasks/swimmer/swimmer.h"
#include "mjpc/tasks/walker/walker.h"
// Humanoid Bench Tasks
#include "mjpc/tasks/humanoid_bench/basic_locomotion/walk/H1_walk.h"
#include "mjpc/tasks/humanoid_bench/basic_locomotion/stand/H1_stand.h"
#include "mjpc/tasks/humanoid_bench/basic_locomotion/run/H1_run.h"
#include "mjpc/tasks/humanoid_bench/basic_locomotion/stairs/H1_stairs.h"
#include "mjpc/tasks/humanoid_bench/basic_locomotion/slide/H1_slide.h"
#include "mjpc/tasks/humanoid_bench/basic_locomotion/crawl/H1_crawl.h"
#include "mjpc/tasks/humanoid_bench/basic_locomotion/hurdle/H1_hurdle.h"
#include "mjpc/tasks/humanoid_bench/basic_locomotion/sit/H1_sit.h"
#include "mjpc/tasks/humanoid_bench/maze/H1_maze.h"
#include "mjpc/tasks/humanoid_bench/poles/H1_poles.h"
#include "mjpc/tasks/humanoid_bench/push/H1_push.h"
#include "mjpc/tasks/humanoid_bench/cabinet/H1_cabinet.h"
#include "mjpc/tasks/humanoid_bench/high_bar/H1_highbar.h"
#include "mjpc/tasks/humanoid_bench/door/H1_door.h"
#include "mjpc/tasks/humanoid_bench/truck/H1_truck.h"
#include "mjpc/tasks/humanoid_bench/cube/H1_cube.h"
#include "mjpc/tasks/humanoid_bench/bookshelf/H1_bookshelf.h"
#include "mjpc/tasks/humanoid_bench/basketball/H1_basketball.h"
#include "mjpc/tasks/humanoid_bench/window/H1_window.h"
#include "mjpc/tasks/humanoid_bench/spoon/H1_spoon.h"
#include "mjpc/tasks/humanoid_bench/kitchen/H1_kitchen.h"
#include "mjpc/tasks/humanoid_bench/package/H1_package.h"
#include "mjpc/tasks/humanoid_bench/powerlift/H1_powerlift.h"
#include "mjpc/tasks/humanoid_bench/room/H1_room.h"
#include "mjpc/tasks/humanoid_bench/insert/H1_insert.h"


#include "mjpc/tasks/humanoid_bench/balance/H1_balance.h"

#include "mjpc/tasks/humanoid_bench/reach/H1_reach.h"

namespace mjpc {

    std::vector<std::shared_ptr<Task>> GetTasks() {
        return {
                //Humanoid Bench Tasks

                //Balance Task
                std::make_shared<Balance_Simple_position>(),
                std::make_shared<Balance_Simple_hand>(),
                std::make_shared<Balance_Simple_gripper>(),
                std::make_shared<Balance_Simple_simple_hand>(),
                std::make_shared<Balance_Simple_strong>(),
                std::make_shared<Balance_Simple_touch>(),


                // Walk Task
                std::make_shared<H1_walk_position>(),
                std::make_shared<H1_walk_hand>(),
                std::make_shared<H1_walk_gripper>(),
                std::make_shared<H1_walk_simple_hand>(),
                std::make_shared<H1_walk_strong>(),
                std::make_shared<H1_walk_touch>(),


                //Slide Task
                std::make_shared<H1_slide_position>(),
                std::make_shared<H1_slide_hand>(),
                std::make_shared<H1_slide_gripper>(),
                std::make_shared<H1_slide_simple_hand>(),
                std::make_shared<H1_slide_strong>(),
                std::make_shared<H1_slide_touch>(),

                //Stand Task
                std::make_shared<H1_stand_position>(),
                std::make_shared<H1_stand_hand>(),
                std::make_shared<H1_stand_gripper>(),
                std::make_shared<H1_stand_simple_hand>(),
                std::make_shared<H1_stand_strong>(),
                std::make_shared<H1_stand_touch>(),

                //Run Task
                std::make_shared<H1_run_position>(),
                std::make_shared<H1_run_hand>(),
                std::make_shared<H1_run_gripper>(),
                std::make_shared<H1_run_simple_hand>(),
                std::make_shared<H1_run_strong>(),
                std::make_shared<H1_run_touch>(),

                //Stairs Task
                std::make_shared<H1_stairs_position>(),
                std::make_shared<H1_stairs_hand>(),
                std::make_shared<H1_stairs_gripper>(),
                std::make_shared<H1_stairs_simple_hand>(),
                std::make_shared<H1_stairs_strong>(),
                std::make_shared<H1_stairs_touch>(),

                //Crawl Task
                std::make_shared<H1_crawl_position>(),
                std::make_shared<H1_crawl_hand>(),
                std::make_shared<H1_crawl_gripper>(),
                std::make_shared<H1_crawl_simple_hand>(),
                std::make_shared<H1_crawl_strong>(),
                std::make_shared<H1_crawl_touch>(),

                //Sit Task
                std::make_shared<H1_sit_position>(),
                std::make_shared<H1_sit_hand>(),
                std::make_shared<H1_sit_gripper>(),
                std::make_shared<H1_sit_simple_hand>(),
                std::make_shared<H1_sit_strong>(),
                std::make_shared<H1_sit_touch>(),

                //Hurdle Task
                std::make_shared<H1_hurdle_position>(),
                std::make_shared<H1_hurdle_hand>(),
                std::make_shared<H1_hurdle_gripper>(),
                std::make_shared<H1_hurdle_simple_hand>(),
                std::make_shared<H1_hurdle_strong>(),
                std::make_shared<H1_hurdle_touch>(),

                //Basketball Task
                std::make_shared<H1_basketball_position>(),
                std::make_shared<H1_basketball_hand>(),
                std::make_shared<H1_basketball_gripper>(),
                std::make_shared<H1_basketball_simple_hand>(),
                std::make_shared<H1_basketball_strong>(),
                std::make_shared<H1_basketball_touch>(),

                //Bookshelf Task
                std::make_shared<H1_bookshelf_position>(),
                std::make_shared<H1_bookshelf_hand>(),
                std::make_shared<H1_bookshelf_gripper>(),
                std::make_shared<H1_bookshelf_simple_hand>(),
                std::make_shared<H1_bookshelf_strong>(),
                std::make_shared<H1_bookshelf_touch>(),

                //Cabinet Task
                std::make_shared<H1_cabinet_position>(),
                std::make_shared<H1_cabinet_hand>(),
                std::make_shared<H1_cabinet_gripper>(),
                std::make_shared<H1_cabinet_simple_hand>(),
                std::make_shared<H1_cabinet_strong>(),
                std::make_shared<H1_cabinet_touch>(),

                //Cube Task
                std::make_shared<H1_cube_position>(),
                std::make_shared<H1_cube_hand>(),
                std::make_shared<H1_cube_gripper>(),
                std::make_shared<H1_cube_simple_hand>(),
                std::make_shared<H1_cube_strong>(),
                std::make_shared<H1_cube_touch>(),

                //Door Task
                std::make_shared<H1_door_position>(),
                std::make_shared<H1_door_hand>(),
                std::make_shared<H1_door_gripper>(),
                std::make_shared<H1_door_simple_hand>(),
                std::make_shared<H1_door_strong>(),
                std::make_shared<H1_door_touch>(),

                //High Bar Task
                std::make_shared<H1_highbar_position>(),
                std::make_shared<H1_highbar_hand>(),
                std::make_shared<H1_highbar_gripper>(),
                std::make_shared<H1_highbar_simple_hand>(),
                std::make_shared<H1_highbar_strong>(),
                std::make_shared<H1_highbar_touch>(),

                //Insert Task
                std::make_shared<H1_insert_position>(),
                std::make_shared<H1_insert_hand>(),
                std::make_shared<H1_insert_gripper>(),
                std::make_shared<H1_insert_simple_hand>(),
                std::make_shared<H1_insert_strong>(),
                std::make_shared<H1_insert_touch>(),

                //Kitchen Task
                std::make_shared<H1_kitchen_position>(),
                std::make_shared<H1_kitchen_hand>(),
                std::make_shared<H1_kitchen_gripper>(),
                std::make_shared<H1_kitchen_simple_hand>(),
                std::make_shared<H1_kitchen_strong>(),
                std::make_shared<H1_kitchen_touch>(),

                //Maze Task
                std::make_shared<H1_maze_position>(),
                std::make_shared<H1_maze_hand>(),
                std::make_shared<H1_maze_gripper>(),
                std::make_shared<H1_maze_simple_hand>(),
                std::make_shared<H1_maze_strong>(),
                std::make_shared<H1_maze_touch>(),

                //Package Task
                std::make_shared<H1_package_position>(),
                std::make_shared<H1_package_hand>(),
                std::make_shared<H1_package_gripper>(),
                std::make_shared<H1_package_simple_hand>(),
                std::make_shared<H1_package_strong>(),
                std::make_shared<H1_package_touch>(),

                //Poles Task
                std::make_shared<H1_poles_position>(),
                std::make_shared<H1_poles_hand>(),
                std::make_shared<H1_poles_gripper>(),
                std::make_shared<H1_poles_simple_hand>(),
                std::make_shared<H1_poles_strong>(),
                std::make_shared<H1_poles_touch>(),

                //Powerlift Task
                std::make_shared<H1_powerlift_position>(),
                std::make_shared<H1_powerlift_hand>(),
                std::make_shared<H1_powerlift_gripper>(),
                std::make_shared<H1_powerlift_simple_hand>(),
                std::make_shared<H1_powerlift_strong>(),
                std::make_shared<H1_powerlift_touch>(),

                //Push Task
                std::make_shared<H1_push_position>(),
                std::make_shared<H1_push_hand>(),
                std::make_shared<H1_push_gripper>(),
                std::make_shared<H1_push_simple_hand>(),
                std::make_shared<H1_push_strong>(),
                std::make_shared<H1_push_touch>(),

                //Reach Task
                std::make_shared<H1_reach_position>(),
                std::make_shared<H1_reach_hand>(),
                std::make_shared<H1_reach_gripper>(),
                std::make_shared<H1_reach_simple_hand>(),
                std::make_shared<H1_reach_strong>(),
                std::make_shared<H1_reach_touch>(),

                //Room Task
                std::make_shared<H1_room_position>(),
                std::make_shared<H1_room_hand>(),
                std::make_shared<H1_room_gripper>(),
                std::make_shared<H1_room_simple_hand>(),
                std::make_shared<H1_room_strong>(),
                std::make_shared<H1_room_touch>(),

                //Spoon Task
                std::make_shared<H1_spoon_position>(),
                std::make_shared<H1_spoon_hand>(),
                std::make_shared<H1_spoon_gripper>(),
                std::make_shared<H1_spoon_simple_hand>(),
                std::make_shared<H1_spoon_strong>(),
                std::make_shared<H1_spoon_touch>(),

                //Truck Task
                std::make_shared<H1_truck_position>(),
                std::make_shared<H1_truck_hand>(),
                std::make_shared<H1_truck_gripper>(),
                std::make_shared<H1_truck_simple_hand>(),
                std::make_shared<H1_truck_strong>(),
                std::make_shared<H1_truck_touch>(),

                //Window Task
                std::make_shared<H1_window_position>(),
                std::make_shared<H1_window_hand>(),
                std::make_shared<H1_window_gripper>(),
                std::make_shared<H1_window_simple_hand>(),
                std::make_shared<H1_window_strong>(),
                std::make_shared<H1_window_touch>(),


//                // original tasks from MuJoCo MPC

//                std::make_shared<Acrobot>(),
//                std::make_shared<Allegro>(),
//                std::make_shared<aloha::Handover>(),
//                std::make_shared<aloha::Reorient>(),
//                std::make_shared<Cartpole>(),
//                std::make_shared<Fingers>(),
//                std::make_shared<humanoid::Stand>(),
//                std::make_shared<humanoid::Tracking>(),
//                std::make_shared<humanoid::Walk>(),
//                std::make_shared<manipulation::Bring>(),
//                // DEEPMIND INTERNAL TASKS
//                std::make_shared<OP3>(),
//                std::make_shared<Panda>(),
//                std::make_shared<Particle>(),
//                std::make_shared<ParticleFixed>(),
//                std::make_shared<Rubik>(),
//                std::make_shared<ShadowReorient>(),
//                std::make_shared<Quadrotor>(),
//                std::make_shared<QuadrupedFlat>(),
//                std::make_shared<QuadrupedHill>(),
//                std::make_shared<Swimmer>(),
//                std::make_shared<Walker>(),
        };
    }
}  // namespace mjpc
