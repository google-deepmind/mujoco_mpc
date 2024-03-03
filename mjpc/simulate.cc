// Copyright 2021 DeepMind Technologies Limited
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

#include "mjpc/simulate.h"  // mjpc fork

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <ratio>
#include <string>

#include "lodepng.h"
#include <mujoco/mjmodel.h>
#include <mujoco/mjvisualize.h>
#include <mujoco/mjxmacro.h>
#include <mujoco/mujoco.h>
#include <platform_ui_adapter.h>
#include "mjpc/array_safety.h"
#include "mjpc/agent.h"
#include "mjpc/utilities.h"

// When launched via an App Bundle on macOS, the working directory is the path
// to the App Bundle's resource directory. This causes files to be saved into
// the bundle, which is not the desired behavior. Instead, we open a save dialog
// box to ask the user where to put the file. Since the dialog box logic needs
// to be written in Cost-C, we separate it into a different source file.
#ifdef __APPLE__
std::string GetSavePath(const char* filename);
#else
static std::string GetSavePath(const char* filename) {
  return filename;
}
#endif

namespace {
namespace mj = ::mujoco;
namespace mju = ::mujoco::util_mjpc;

using Seconds = std::chrono::duration<double>;
using Milliseconds = std::chrono::duration<double, std::milli>;

//------------------------------------------- global -----------------------------------------------

const int maxgeom = 5000;            // preallocated geom array in mjvScene
const double zoom_increment = 0.02;  // ratio of one click-wheel zoom increment to vertical extent

// section ids
enum {
  // left ui
  SECT_FILE = 0,
  SECT_OPTION,
  SECT_SIMULATION,
  SECT_WATCH,
  SECT_TASK,
  SECT_AGENT,
  SECT_ESTIMATOR,
  SECT_PHYSICS,
  SECT_RENDERING,
  SECT_GROUP,
  NSECT0,

  // right ui
  SECT_JOINT = 0,
  SECT_CONTROL,
  NSECT1
};

// file section of UI
const mjuiDef defFile[] = {
  {mjITEM_SECTION,   "File",          0, nullptr,                    "AF"},
  {mjITEM_BUTTON,    "Save xml",      2, nullptr,                    ""},
  {mjITEM_BUTTON,    "Save mjb",      2, nullptr,                    ""},
  {mjITEM_BUTTON,    "Print model",   2, nullptr,                    "CM"},
  {mjITEM_BUTTON,    "Print data",    2, nullptr,                    "CD"},
  {mjITEM_BUTTON,    "Quit",          1, nullptr,                    "CQ"},
  {mjITEM_BUTTON,    "Screenshot",    2, nullptr,                    "CP"},
  {mjITEM_END}
};

// help strings
const char help_content[] =
  "Space\n"
  "+  -\n"
  "Right arrow\n"
  "[  ]\n"
  "Esc\n"
  "Double-click\n"
  "Page Up\n"
  "Right double-click\n"
  "Ctrl Right double-click\n"
  "Scroll, middle drag\n"
  "Left drag\n"
  "[Shift] right drag\n"
  "Ctrl [Shift] drag\n"
  "Ctrl [Shift] right drag\n"
  "F1\n"
  "F2\n"
  "F3\n"
  "F4\n"
  "F5\n"
  "UI right hold\n"
  "UI title double-click";

const char help_title[] =
  "Play / Pause\n"
  "Speed up / down\n"
  "Step\n"
  "Cycle cameras\n"
  "Free camera\n"
  "Select\n"
  "Select parent\n"
  "Center\n"
  "Tracking camera\n"
  "Zoom\n"
  "View rotate\n"
  "View translate\n"
  "Object rotate\n"
  "Object translate\n"
  "Help\n"
  "Info\n"
  "Profiler\n"
  "Sensors\n"
  "Full screen\n"
  "Show UI shortcuts\n"
  "Expand/collapse all";


//-------------------------------- profiler, sensor, info, watch -----------------------------------

// number of lines in the Constraint ("Counts") and Cost ("Convergence") figures
static constexpr int kConstraintNum = 5;
static constexpr int kCostNum = 3;

// init profiler figures
void InitializeProfiler(mj::Simulate* sim) {
  // set figures to default
  mjv_defaultFigure(&sim->figconstraint);
  mjv_defaultFigure(&sim->figcost);
  mjv_defaultFigure(&sim->figtimer);
  mjv_defaultFigure(&sim->figsize);

  // titles
  mju::strcpy_arr(sim->figconstraint.title, "Counts");
  mju::strcpy_arr(sim->figcost.title, "Convergence (log 10)");
  mju::strcpy_arr(sim->figsize.title, "Dimensions");
  mju::strcpy_arr(sim->figtimer.title, "CPU time (msec)");

  // x-labels
  mju::strcpy_arr(sim->figconstraint.xlabel, "Solver iteration");
  mju::strcpy_arr(sim->figcost.xlabel, "Solver iteration");
  mju::strcpy_arr(sim->figsize.xlabel, "Video frame");
  mju::strcpy_arr(sim->figtimer.xlabel, "Video frame");

  // y-tick number formats
  mju::strcpy_arr(sim->figconstraint.yformat, "%.0f");
  mju::strcpy_arr(sim->figcost.yformat, "%.1f");
  mju::strcpy_arr(sim->figsize.yformat, "%.0f");
  mju::strcpy_arr(sim->figtimer.yformat, "%.2f");

  // colors
  sim->figconstraint.figurergba[0] = 0.1f;
  sim->figcost.figurergba[2]       = 0.2f;
  sim->figsize.figurergba[0]       = 0.1f;
  sim->figtimer.figurergba[2]      = 0.2f;
  sim->figconstraint.figurergba[3] = 0.5f;
  sim->figcost.figurergba[3]       = 0.5f;
  sim->figsize.figurergba[3]       = 0.5f;
  sim->figtimer.figurergba[3]      = 0.5f;

  // repeat line colors for constraint and cost figures
  mjvFigure* fig = &sim->figcost;
  for (int i=kCostNum; i<mjMAXLINE; i++) {
    fig->linergb[i][0] = fig->linergb[i - kCostNum][0];
    fig->linergb[i][1] = fig->linergb[i - kCostNum][1];
    fig->linergb[i][2] = fig->linergb[i - kCostNum][2];
  }
  fig = &sim->figconstraint;
  for (int i=kConstraintNum; i<mjMAXLINE; i++) {
    fig->linergb[i][0] = fig->linergb[i - kConstraintNum][0];
    fig->linergb[i][1] = fig->linergb[i - kConstraintNum][1];
    fig->linergb[i][2] = fig->linergb[i - kConstraintNum][2];
  }

  // legends
  mju::strcpy_arr(sim->figconstraint.linename[0], "total");
  mju::strcpy_arr(sim->figconstraint.linename[1], "active");
  mju::strcpy_arr(sim->figconstraint.linename[2], "changed");
  mju::strcpy_arr(sim->figconstraint.linename[3], "evals");
  mju::strcpy_arr(sim->figconstraint.linename[4], "updates");
  mju::strcpy_arr(sim->figcost.linename[0], "improvement");
  mju::strcpy_arr(sim->figcost.linename[1], "gradient");
  mju::strcpy_arr(sim->figcost.linename[2], "lineslope");
  mju::strcpy_arr(sim->figsize.linename[0], "dof");
  mju::strcpy_arr(sim->figsize.linename[1], "body");
  mju::strcpy_arr(sim->figsize.linename[2], "constraint");
  mju::strcpy_arr(sim->figsize.linename[3], "sqrt(nnz)");
  mju::strcpy_arr(sim->figsize.linename[4], "contact");
  mju::strcpy_arr(sim->figsize.linename[5], "iteration");
  mju::strcpy_arr(sim->figtimer.linename[0], "total");
  mju::strcpy_arr(sim->figtimer.linename[1], "collision");
  mju::strcpy_arr(sim->figtimer.linename[2], "prepare");
  mju::strcpy_arr(sim->figtimer.linename[3], "solve");
  mju::strcpy_arr(sim->figtimer.linename[4], "other");

  // grid sizes
  sim->figconstraint.gridsize[0] = 5;
  sim->figconstraint.gridsize[1] = 5;
  sim->figcost.gridsize[0] = 5;
  sim->figcost.gridsize[1] = 5;
  sim->figsize.gridsize[0] = 3;
  sim->figsize.gridsize[1] = 5;
  sim->figtimer.gridsize[0] = 3;
  sim->figtimer.gridsize[1] = 5;

  // minimum ranges
  sim->figconstraint.range[0][0] = 0;
  sim->figconstraint.range[0][1] = 20;
  sim->figconstraint.range[1][0] = 0;
  sim->figconstraint.range[1][1] = 80;
  sim->figcost.range[0][0] = 0;
  sim->figcost.range[0][1] = 20;
  sim->figcost.range[1][0] = -15;
  sim->figcost.range[1][1] = 5;
  sim->figsize.range[0][0] = -200;
  sim->figsize.range[0][1] = 0;
  sim->figsize.range[1][0] = 0;
  sim->figsize.range[1][1] = 100;
  sim->figtimer.range[0][0] = -200;
  sim->figtimer.range[0][1] = 0;
  sim->figtimer.range[1][0] = 0;
  sim->figtimer.range[1][1] = 0.4f;

  // init x axis on history figures (do not show yet)
  for (int n=0; n<6; n++) {
    for (int i=0; i<mjMAXLINEPNT; i++) {
      sim->figtimer.linedata[n][2*i] = -i;
      sim->figsize.linedata[n][2*i] = -i;
    }
  }
}

// update profiler figures
void UpdateProfiler(mj::Simulate* sim) {
  // reset lines in Constraint and Cost figures
  memset(sim->figconstraint.linepnt, 0, mjMAXLINE*sizeof(int));
  memset(sim->figcost.linepnt, 0, mjMAXLINE*sizeof(int));

  // number of islands that have diagnostics
  int nisland = mjMIN(sim->d->solver_nisland, mjNISLAND);

  // iterate over islands
  for (int k=0; k < nisland; k++) {
    // ==== update Constraint ("Counts") figure

    // number of points to plot, starting line
    int npoints = mjMIN(mjMIN(sim->d->solver_niter[k], mjNSOLVER), mjMAXLINEPNT);
    int start = kConstraintNum * k;

    sim->figconstraint.linepnt[start + 0] = npoints;
    for (int i=1; i < kConstraintNum; i++) {
      sim->figconstraint.linepnt[start + i] = npoints;
    }
    if (sim->m->opt.solver == mjSOL_PGS) {
      sim->figconstraint.linepnt[start + 3] = 0;
      sim->figconstraint.linepnt[start + 4] = 0;
    }
    if (sim->m->opt.solver == mjSOL_CG) {
      sim->figconstraint.linepnt[start + 4] = 0;
    }
    for (int i=0; i<npoints; i++) {
      // x
      sim->figconstraint.linedata[start + 0][2*i] = i;
      sim->figconstraint.linedata[start + 1][2*i] = i;
      sim->figconstraint.linedata[start + 2][2*i] = i;
      sim->figconstraint.linedata[start + 3][2*i] = i;
      sim->figconstraint.linedata[start + 4][2*i] = i;

      // y
      int nefc = nisland == 1 ? sim->d->nefc : sim->d->island_efcnum[k];
      sim->figconstraint.linedata[start + 0][2*i+1] = nefc;
      const mjSolverStat* stat = sim->d->solver + k*mjNSOLVER + i;
      sim->figconstraint.linedata[start + 1][2*i+1] = stat->nactive;
      sim->figconstraint.linedata[start + 2][2*i+1] = stat->nchange;
      sim->figconstraint.linedata[start + 3][2*i+1] = stat->neval;
      sim->figconstraint.linedata[start + 4][2*i+1] = stat->nupdate;
    }

    // update cost figure
    sim->figcost.linepnt[start + 0] = npoints;
    for (int i=1; i<kCostNum; i++) {
      sim->figcost.linepnt[start + i] = npoints;
    }
    if (sim->m->opt.solver==mjSOL_PGS) {
      sim->figcost.linepnt[start + 1] = 0;
      sim->figcost.linepnt[start + 2] = 0;
    }

    for (int i=0; i<sim->figcost.linepnt[0]; i++) {
      // x
      sim->figcost.linedata[start + 0][2*i] = i;
      sim->figcost.linedata[start + 1][2*i] = i;
      sim->figcost.linedata[start + 2][2*i] = i;

      // y
      const mjSolverStat* stat = sim->d->solver + k*mjNSOLVER + i;
      sim->figcost.linedata[start + 0][2*i + 1] =
          mju_log10(mju_max(mjMINVAL, stat->improvement));
      sim->figcost.linedata[start + 1][2*i + 1] =
          mju_log10(mju_max(mjMINVAL, stat->gradient));
      sim->figcost.linedata[start + 2][2*i + 1] =
          mju_log10(mju_max(mjMINVAL, stat->lineslope));
    }
  }

  // get timers: total, collision, prepare, solve, other
  mjtNum total = sim->d->timer[mjTIMER_STEP].duration;
  int number = sim->d->timer[mjTIMER_STEP].number;
  if (!number) {
    total = sim->d->timer[mjTIMER_FORWARD].duration;
    number = sim->d->timer[mjTIMER_FORWARD].number;
  }
  number = mjMAX(1, number);
  float tdata[5] = {
    static_cast<float>(total/number),
    static_cast<float>(sim->d->timer[mjTIMER_POS_COLLISION].duration/number),
    static_cast<float>(sim->d->timer[mjTIMER_POS_MAKE].duration/number) +
    static_cast<float>(sim->d->timer[mjTIMER_POS_PROJECT].duration/number),
    static_cast<float>(sim->d->timer[mjTIMER_CONSTRAINT].duration/number),
    0
  };
  tdata[4] = tdata[0] - tdata[1] - tdata[2] - tdata[3];

  // update figtimer
  int pnt = mjMIN(201, sim->figtimer.linepnt[0]+1);
  for (int n=0; n<5; n++) {
    // shift data
    for (int i=pnt-1; i>0; i--) {
      sim->figtimer.linedata[n][2*i+1] = sim->figtimer.linedata[n][2*i-1];
    }

    // assign new
    sim->figtimer.linepnt[n] = pnt;
    sim->figtimer.linedata[n][1] = tdata[n];
  }

  // get total number of iterations and nonzeros
  mjtNum sqrt_nnz = 0;
  int solver_niter = 0;
  for (int island=0; island < nisland; island++) {
    sqrt_nnz += mju_sqrt(sim->d->solver_nnz[island]);
    solver_niter += sim->d->solver_niter[island];
  }

  // get sizes: nv, nbody, nefc, sqrt(nnz), ncont, iter
  float sdata[6] = {
    static_cast<float>(sim->m->nv),
    static_cast<float>(sim->m->nbody),
    static_cast<float>(sim->d->nefc),
    static_cast<float>(sqrt_nnz),
    static_cast<float>(sim->d->ncon),
    static_cast<float>(solver_niter)
  };

  // update figsize
  pnt = mjMIN(201, sim->figsize.linepnt[0]+1);
  for (int n=0; n<6; n++) {
    // shift data
    for (int i=pnt-1; i>0; i--) {
      sim->figsize.linedata[n][2*i+1] = sim->figsize.linedata[n][2*i-1];
    }

    // assign new
    sim->figsize.linepnt[n] = pnt;
    sim->figsize.linedata[n][1] = sdata[n];
  }
}

// show profiler figures
void ShowProfiler(mj::Simulate* sim, mjrRect rect) {
  mjrRect viewport = {
    rect.left + rect.width - rect.width/4,
    rect.bottom,
    rect.width/4,
    rect.height/4
  };
  mjr_figure(viewport, &sim->figtimer, &sim->platform_ui->mjr_context());
  viewport.bottom += rect.height/4;
  mjr_figure(viewport, &sim->figsize, &sim->platform_ui->mjr_context());
  viewport.bottom += rect.height/4;
  mjr_figure(viewport, &sim->figcost, &sim->platform_ui->mjr_context());
  viewport.bottom += rect.height/4;
  mjr_figure(viewport, &sim->figconstraint, &sim->platform_ui->mjr_context());
}


// init sensor figure
void InitializeSensor(mj::Simulate* sim) {
  mjvFigure& figsensor = sim->figsensor;

  // set figure to default
  mjv_defaultFigure(&figsensor);
  figsensor.figurergba[3] = 0.5f;

  // set flags
  figsensor.flg_extend = 1;
  figsensor.flg_barplot = 1;
  figsensor.flg_symmetric = 1;

  // title
  mju::strcpy_arr(figsensor.title, "Sensor data");

  // y-tick nubmer format
  mju::strcpy_arr(figsensor.yformat, "%.0f");

  // grid size
  figsensor.gridsize[0] = 2;
  figsensor.gridsize[1] = 3;

  // minimum range
  figsensor.range[0][0] = 0;
  figsensor.range[0][1] = 0;
  figsensor.range[1][0] = -1;
  figsensor.range[1][1] = 1;
}

// update sensor figure
void UpdateSensor(mj::Simulate* sim) {
  mjModel* m = sim->m;
  mjvFigure& figsensor = sim->figsensor;
  static const int maxline = 10;

  // clear linepnt
  for (int i=0; i<maxline; i++) {
    figsensor.linepnt[i] = 0;
  }

  // start with line 0
  int lineid = 0;

  // loop over sensors
  for (int n=0; n<m->nsensor; n++) {
    // go to next line if type is different
    if (n>0 && m->sensor_type[n]!=m->sensor_type[n-1]) {
      lineid = mjMIN(lineid+1, maxline-1);
    }

    // get info about this sensor
    mjtNum cutoff = (m->sensor_cutoff[n]>0 ? m->sensor_cutoff[n] : 1);
    int adr = m->sensor_adr[n];
    int dim = m->sensor_dim[n];

    // data pointer in line
    int p = figsensor.linepnt[lineid];

    // fill in data for this sensor
    for (int i=0; i<dim; i++) {
      // check size
      if ((p+2*i)>=mjMAXLINEPNT/2) {
        break;
      }

      // x
      figsensor.linedata[lineid][2*p+4*i] = adr+i;
      figsensor.linedata[lineid][2*p+4*i+2] = adr+i;

      // y
      figsensor.linedata[lineid][2*p+4*i+1] = 0;
      figsensor.linedata[lineid][2*p+4*i+3] = sim->d->sensordata[adr+i]/cutoff;
    }

    // update linepnt
    figsensor.linepnt[lineid] = mjMIN(mjMAXLINEPNT-1,
                                       figsensor.linepnt[lineid]+2*dim);
  }
}

// show sensor figure
void ShowSensor(mj::Simulate* sim, mjrRect rect) {
  // constant width with and without profiler
  int width = sim->profiler ? rect.width/3 : rect.width/4;

  // render figure on the right
  mjrRect viewport = {
    rect.left + rect.width - width,
    rect.bottom,
    width,
    rect.height/3
  };
  mjr_figure(viewport, &sim->figsensor, &sim->platform_ui->mjr_context());
}

// prepare info text
void UpdateInfoText(mj::Simulate* sim,
                    char (&title)[mj::Simulate::kMaxFilenameLength],
                    char (&content)[mj::Simulate::kMaxFilenameLength],
                    double interval) {
  mjModel* m = sim->m;
  mjData* d = sim->d;

  // compute solver error
  int island = 0;  // first island only
  mjtNum solerr = 0;
  if (d->solver_niter[island]) {
    int ind = mjMIN(sim->d->solver_niter[island]-1, mjNSOLVER-1);
    const mjSolverStat* stat = sim->d->solver + island*mjNSOLVER + ind;
    solerr = mju_min(stat->improvement, stat->gradient);
    if (solerr == 0) {
      solerr = mju_max(stat->improvement, stat->gradient);
    }
  }
  solerr = mju_log10(mju_max(mjMINVAL, solerr));

  // prepare info text
  mju::strcpy_arr(title, "Objective\nDoFs\nControls\nParameters\nTime\nMemory");
  const mjpc::Trajectory* best_trajectory =
      sim->agent->ActivePlanner().BestTrajectory();
  if (best_trajectory) {
    int nparam = sim->agent->ActivePlanner().NumParameters();
    mju::sprintf_arr(content, "%.3f\n%d\n%d\n%d\n%-9.3f\n%.2g of %s",
                     best_trajectory->total_return, m->nv, m->nu, nparam,
                     d->time, d->maxuse_arena / (double)(d->narena),
                     mju_writeNumBytes(d->narena));
  }

  // add Energy if enabled
  {
    if (mjENABLED(mjENBL_ENERGY)) {
      char tmp[20];
      mju::sprintf_arr(tmp, "\n%.3f", d->energy[0]+d->energy[1]);
      mju::strcat_arr(content, tmp);
      mju::strcat_arr(title, "\nEnergy");
    }

    // add FwdInv if enabled
    if (mjENABLED(mjENBL_FWDINV)) {
      char tmp[20];
      mju::sprintf_arr(tmp, "\n%.1f %.1f",
                       mju_log10(mju_max(mjMINVAL, d->solver_fwdinv[0])),
                       mju_log10(mju_max(mjMINVAL, d->solver_fwdinv[1])));
      mju::strcat_arr(content, tmp);
      mju::strcat_arr(title, "\nFwdInv");
    }

    // add islands if enabled
    if (mjENABLED(mjENBL_ISLAND)) {
      char tmp[20];
      mju::sprintf_arr(tmp, "\n%d", d->nisland);
      mju::strcat_arr(content, tmp);
      mju::strcat_arr(title, "\nIslands");
    }
  }
}

// sprintf forwarding, to avoid compiler warning in x-macro
void PrintField(char (&str)[mjMAXUINAME], void* ptr) {
  mju::sprintf_arr(str, "%g", *static_cast<mjtNum*>(ptr));
}

// update watch
void UpdateWatch(mj::Simulate* sim) {
  // clear
  sim->ui0.sect[SECT_WATCH].item[2].multi.nelem = 1;
  mju::strcpy_arr(sim->ui0.sect[SECT_WATCH].item[2].multi.name[0], "invalid field");

  // prepare symbols needed by xmacro
  MJDATA_POINTERS_PREAMBLE(sim->m);

  // find specified field in mjData arrays, update value
  #define X(TYPE, NAME, NR, NC)                                                                  \
    if (!mju::strcmp_arr(#NAME, sim->field) &&                                                   \
        !mju::strcmp_arr(#TYPE, "mjtNum")) {                                                     \
      if (sim->index >= 0 && sim->index < sim->m->NR * NC) {                                     \
        PrintField(sim->ui0.sect[SECT_WATCH].item[2].multi.name[0], sim->d->NAME + sim->index);  \
      } else {                                                                                   \
        mju::strcpy_arr(sim->ui0.sect[SECT_WATCH].item[2].multi.name[0], "invalid index");       \
      }                                                                                          \
      return;                                                                                    \
    }

  MJDATA_POINTERS
#undef X
}


//---------------------------------- UI construction -----------------------------------------------

// make physics section of UI
void MakePhysicsSection(mj::Simulate* sim, int oldstate) {
  mjOption* opt = &sim->m->opt;

  mjuiDef defPhysics[] = {
    {mjITEM_SECTION,   "Physics",       oldstate, nullptr,           "AP"},
    {mjITEM_SELECT,    "Integrator",    2, &(opt->integrator),        "Euler\nRK4\nimplicit\nimplicitfast"},
    {mjITEM_SELECT,    "Cone",          2, &(opt->cone),              "Pyramidal\nElliptic"},
    {mjITEM_SELECT,    "Jacobian",      2, &(opt->jacobian),          "Dense\nSparse\nAuto"},
    {mjITEM_SELECT,    "Solver",        2, &(opt->solver),            "PGS\nCG\nNewton"},
    {mjITEM_SEPARATOR, "Algorithmic Parameters", 1},
    {mjITEM_EDITNUM,   "Timestep",      2, &(opt->timestep),          "1 0 1"},
    {mjITEM_EDITINT,   "Iterations",    2, &(opt->iterations),        "1 0 1000"},
    {mjITEM_EDITNUM,   "Tolerance",     2, &(opt->tolerance),         "1 0 1"},
    {mjITEM_EDITINT,   "LS Iter",       2, &(opt->ls_iterations),     "1 0 100"},
    {mjITEM_EDITNUM,   "LS Tol",        2, &(opt->ls_tolerance),      "1 0 0.1"},
    {mjITEM_EDITINT,   "Noslip Iter",   2, &(opt->noslip_iterations), "1 0 1000"},
    {mjITEM_EDITNUM,   "Noslip Tol",    2, &(opt->noslip_tolerance),  "1 0 1"},
    {mjITEM_EDITINT,   "MPR Iter",      2, &(opt->mpr_iterations),    "1 0 1000"},
    {mjITEM_EDITNUM,   "MPR Tol",       2, &(opt->mpr_tolerance),     "1 0 1"},
    {mjITEM_EDITNUM,   "API Rate",      2, &(opt->apirate),           "1 0 1000"},
    {mjITEM_EDITINT,   "SDF Iter",      2, &(opt->sdf_iterations),    "1 1 20"},
    {mjITEM_EDITINT,   "SDF Init",      2, &(opt->sdf_initpoints),    "1 1 100"},
    {mjITEM_SEPARATOR, "Physical Parameters", 1},
    {mjITEM_EDITNUM,   "Gravity",       2, opt->gravity,              "3"},
    {mjITEM_EDITNUM,   "Wind",          2, opt->wind,                 "3"},
    {mjITEM_EDITNUM,   "Magnetic",      2, opt->magnetic,             "3"},
    {mjITEM_EDITNUM,   "Density",       2, &(opt->density),           "1"},
    {mjITEM_EDITNUM,   "Viscosity",     2, &(opt->viscosity),         "1"},
    {mjITEM_EDITNUM,   "Imp Ratio",     2, &(opt->impratio),          "1"},
    {mjITEM_SEPARATOR, "Disable Flags", 1},
    {mjITEM_END}
  };
  mjuiDef defEnableFlags[] = {
    {mjITEM_SEPARATOR, "Enable Flags", 1},
    {mjITEM_END}
  };
  mjuiDef defOverride[] = {
    {mjITEM_SEPARATOR, "Contact Override", 1},
    {mjITEM_EDITNUM,   "Margin",        2, &(opt->o_margin),          "1"},
    {mjITEM_EDITNUM,   "Sol Imp",       2, &(opt->o_solimp),          "5"},
    {mjITEM_EDITNUM,   "Sol Ref",       2, &(opt->o_solref),          "2"},
    {mjITEM_END}
  };

  // add physics
  mjui_add(&sim->ui0, defPhysics);

  // add flags programmatically
  mjuiDef defFlag[] = {
    {mjITEM_CHECKINT,  "", 2, nullptr, ""},
    {mjITEM_END}
  };
  for (int i=0; i<mjNDISABLE; i++) {
    mju::strcpy_arr(defFlag[0].name, mjDISABLESTRING[i]);
    defFlag[0].pdata = sim->disable + i;
    mjui_add(&sim->ui0, defFlag);
  }
  mjui_add(&sim->ui0, defEnableFlags);
  for (int i=0; i<mjNENABLE; i++) {
    mju::strcpy_arr(defFlag[0].name, mjENABLESTRING[i]);
    defFlag[0].pdata = sim->enable + i;
    mjui_add(&sim->ui0, defFlag);
  }

  // add contact override
  mjui_add(&sim->ui0, defOverride);
}



// make rendering section of UI
void MakeRenderingSection(mj::Simulate* sim, int oldstate) {
  mjuiDef defRendering[] = {
      {mjITEM_SECTION, "Rendering", oldstate, nullptr, "AR"},
      {mjITEM_SELECT, "Camera", 2, &(sim->camera), "Free\nTracking"},
      {mjITEM_SELECT, "Label", 2, &(sim->opt.label),
       "None\nBody\nJoint\nGeom\nSite\nCamera\nLight\nTendon\n"
       "Actuator\nConstraint\nSkin\nSelection\nSel "
       "Pnt\nContact\nForce\nIsland"},
      {mjITEM_SELECT, "Frame", 2, &(sim->opt.frame),
       "None\nBody\nGeom\nSite\nCamera\nLight\nContact\nWorld"},
      {mjITEM_BUTTON, "Copy camera", 2, nullptr, ""},
      {mjITEM_BUTTON, "Copy state", 2, nullptr, ""},
      {mjITEM_SEPARATOR, "Model Elements", 1},
      {mjITEM_END}};
  mjuiDef defOpenGL[] = {
    {mjITEM_SEPARATOR, "OpenGL Effects", 1},
    {mjITEM_END}
  };

  // add model cameras, up to UI limit
  for (int i=0; i<mjMIN(sim->m->ncam, mjMAXUIMULTI-2); i++) {
    // prepare name
    char camname[mjMAXUINAME] = "\n";
    if (sim->m->names[sim->m->name_camadr[i]]) {
      mju::strcat_arr(camname, sim->m->names+sim->m->name_camadr[i]);
    } else {
      mju::sprintf_arr(camname, "\nCamera %d", i);
    }

    // check string length
    if (mju::strlen_arr(camname) + mju::strlen_arr(defRendering[1].other)>=mjMAXUITEXT-1) {
      break;
    }

    // add camera
    mju::strcat_arr(defRendering[1].other, camname);
  }

  // add rendering standard
  mjui_add(&sim->ui0, defRendering);

  // add flags programmatically
  mjuiDef defFlag[] = {
    {mjITEM_CHECKBYTE,  "", 2, nullptr, ""},
    {mjITEM_END}
  };
  for (int i=0; i<mjNVISFLAG; i++) {
    // set name, remove "&"
    mju::strcpy_arr(defFlag[0].name, mjVISSTRING[i][0]);
    for (int j=0; j<strlen(mjVISSTRING[i][0]); j++) {
      if (mjVISSTRING[i][0][j]=='&') {
        mju_strncpy(
          defFlag[0].name+j, mjVISSTRING[i][0]+j+1, mju::sizeof_arr(defFlag[0].name)-j);
        break;
      }
    }

    // set shortcut and data
    if (mjVISSTRING[i][2][0]) {
      mju::sprintf_arr(defFlag[0].other, " %s", mjVISSTRING[i][2]);
    } else {
      mju::sprintf_arr(defFlag[0].other, "");
    }
    defFlag[0].pdata = sim->opt.flags + i;
    mjui_add(&sim->ui0, defFlag);
  }

  // create tree slider
  mjuiDef defTree[] = {
      {mjITEM_SLIDERINT, "Tree depth", 2, &sim->opt.bvh_depth, "0 20"},
      {mjITEM_END}
  };
  mjui_add(&sim->ui0, defTree);

  // add rendering flags
  mjui_add(&sim->ui0, defOpenGL);
  for (int i=0; i<mjNRNDFLAG; i++) {
    mju::strcpy_arr(defFlag[0].name, mjRNDSTRING[i][0]);
    if (mjRNDSTRING[i][2][0]) {
      mju::sprintf_arr(defFlag[0].other, " %s", mjRNDSTRING[i][2]);
    } else {
      mju::sprintf_arr(defFlag[0].other, "");
    }
    defFlag[0].pdata = sim->scn.flags + i;
    mjui_add(&sim->ui0, defFlag);
  }
}



// make group section of UI
void MakeGroupSection(mj::Simulate* sim, int oldstate) {
  mjvOption& vopt = sim->opt;
  mjuiDef defGroup[] = {
    {mjITEM_SECTION,    "Group enable",     oldstate, nullptr,          "AG"},
    {mjITEM_SEPARATOR,  "Geom groups",  1},
    {mjITEM_CHECKBYTE,  "Geom 0",           2, vopt.geomgroup,          " 0"},
    {mjITEM_CHECKBYTE,  "Geom 1",           2, vopt.geomgroup+1,        " 1"},
    {mjITEM_CHECKBYTE,  "Geom 2",           2, vopt.geomgroup+2,        " 2"},
    {mjITEM_CHECKBYTE,  "Geom 3",           2, vopt.geomgroup+3,        " 3"},
    {mjITEM_CHECKBYTE,  "Geom 4",           2, vopt.geomgroup+4,        " 4"},
    {mjITEM_CHECKBYTE,  "Geom 5",           2, vopt.geomgroup+5,        " 5"},
    {mjITEM_SEPARATOR,  "Site groups",  1},
    {mjITEM_CHECKBYTE,  "Site 0",           2, vopt.sitegroup,          "S0"},
    {mjITEM_CHECKBYTE,  "Site 1",           2, vopt.sitegroup+1,        "S1"},
    {mjITEM_CHECKBYTE,  "Site 2",           2, vopt.sitegroup+2,        "S2"},
    {mjITEM_CHECKBYTE,  "Site 3",           2, vopt.sitegroup+3,        "S3"},
    {mjITEM_CHECKBYTE,  "Site 4",           2, vopt.sitegroup+4,        "S4"},
    {mjITEM_CHECKBYTE,  "Site 5",           2, vopt.sitegroup+5,        "S5"},
    {mjITEM_SEPARATOR,  "Joint groups", 1},
    {mjITEM_CHECKBYTE,  "Joint 0",          2, vopt.jointgroup,         ""},
    {mjITEM_CHECKBYTE,  "Joint 1",          2, vopt.jointgroup+1,       ""},
    {mjITEM_CHECKBYTE,  "Joint 2",          2, vopt.jointgroup+2,       ""},
    {mjITEM_CHECKBYTE,  "Joint 3",          2, vopt.jointgroup+3,       ""},
    {mjITEM_CHECKBYTE,  "Joint 4",          2, vopt.jointgroup+4,       ""},
    {mjITEM_CHECKBYTE,  "Joint 5",          2, vopt.jointgroup+5,       ""},
    {mjITEM_SEPARATOR,  "Tendon groups",    1},
    {mjITEM_CHECKBYTE,  "Tendon 0",         2, vopt.tendongroup,        ""},
    {mjITEM_CHECKBYTE,  "Tendon 1",         2, vopt.tendongroup+1,      ""},
    {mjITEM_CHECKBYTE,  "Tendon 2",         2, vopt.tendongroup+2,      ""},
    {mjITEM_CHECKBYTE,  "Tendon 3",         2, vopt.tendongroup+3,      ""},
    {mjITEM_CHECKBYTE,  "Tendon 4",         2, vopt.tendongroup+4,      ""},
    {mjITEM_CHECKBYTE,  "Tendon 5",         2, vopt.tendongroup+5,      ""},
    {mjITEM_SEPARATOR,  "Actuator groups", 1},
    {mjITEM_CHECKBYTE,  "Actuator 0",       2, vopt.actuatorgroup,      ""},
    {mjITEM_CHECKBYTE,  "Actuator 1",       2, vopt.actuatorgroup+1,    ""},
    {mjITEM_CHECKBYTE,  "Actuator 2",       2, vopt.actuatorgroup+2,    ""},
    {mjITEM_CHECKBYTE,  "Actuator 3",       2, vopt.actuatorgroup+3,    ""},
    {mjITEM_CHECKBYTE,  "Actuator 4",       2, vopt.actuatorgroup+4,    ""},
    {mjITEM_CHECKBYTE,  "Actuator 5",       2, vopt.actuatorgroup+5,    ""},
    {mjITEM_SEPARATOR,  "Skin groups", 1},
    {mjITEM_CHECKBYTE,  "Skin 0",           2, vopt.skingroup,          ""},
    {mjITEM_CHECKBYTE,  "Skin 1",           2, vopt.skingroup+1,        ""},
    {mjITEM_CHECKBYTE,  "Skin 2",           2, vopt.skingroup+2,        ""},
    {mjITEM_CHECKBYTE,  "Skin 3",           2, vopt.skingroup+3,        ""},
    {mjITEM_CHECKBYTE,  "Skin 4",           2, vopt.skingroup+4,        ""},
    {mjITEM_CHECKBYTE,  "Skin 5",           2, vopt.skingroup+5,        ""},
    {mjITEM_END}
  };

  // add section
  mjui_add(&sim->ui0, defGroup);
}

// make joint section of UI
void MakeJointSection(mj::Simulate* sim, int oldstate) {
  mjuiDef defJoint[] = {
    {mjITEM_SECTION, "Joint", oldstate, nullptr, "AJ"},
    {mjITEM_END}
  };
  mjuiDef defSlider[] = {
    {mjITEM_SLIDERNUM, "", 2, nullptr, "0 1"},
    {mjITEM_END}
  };

  // add section
  mjui_add(&sim->ui1, defJoint);
  defSlider[0].state = 4;

  // add scalar joints, exit if UI limit reached
  int itemcnt = 0;
  for (int i=0; i<sim->m->njnt && itemcnt<mjMAXUIITEM; i++)
    if ((sim->m->jnt_type[i]==mjJNT_HINGE || sim->m->jnt_type[i]==mjJNT_SLIDE)) {
      // skip if joint group is disabled
      if (!sim->opt.jointgroup[mjMAX(0, mjMIN(mjNGROUP-1, sim->m->jnt_group[i]))]) {
        continue;
      }

      // set data and name
      defSlider[0].pdata = sim->d->qpos + sim->m->jnt_qposadr[i];
      if (sim->m->names[sim->m->name_jntadr[i]]) {
        mju::strcpy_arr(defSlider[0].name, sim->m->names+sim->m->name_jntadr[i]);
      } else {
        mju::sprintf_arr(defSlider[0].name, "joint %d", i);
      }

      // set range
      if (sim->m->jnt_limited[i])
        mju::sprintf_arr(defSlider[0].other, "%.4g %.4g",
                         sim->m->jnt_range[2*i], sim->m->jnt_range[2*i+1]);
      else if (sim->m->jnt_type[i]==mjJNT_SLIDE) {
        mju::strcpy_arr(defSlider[0].other, "-1 1");
      } else {
        mju::strcpy_arr(defSlider[0].other, "-3.1416 3.1416");
      }

      // add and count
      mjui_add(&sim->ui1, defSlider);
      itemcnt++;
    }
}

// make control section of UI
void MakeControlSection(mj::Simulate* sim, int oldstate) {
  mjuiDef defControl[] = {
    {mjITEM_SECTION, "Control", oldstate, nullptr, "AC"},
    {mjITEM_BUTTON,  "Clear all", 2},
    {mjITEM_END}
  };
  mjuiDef defSlider[] = {
    {mjITEM_SLIDERNUM, "", 2, nullptr, "0 1"},
    {mjITEM_END}
  };

  // add section
  mjui_add(&sim->ui1, defControl);
  defSlider[0].state = 2;

  // add controls, exit if UI limit reached (Clear button already added)
  int itemcnt = 1;
  for (int i=0; i<sim->m->nu && itemcnt<mjMAXUIITEM; i++) {
    // skip if actuator group is disabled
    if (!sim->opt.actuatorgroup[mjMAX(0, mjMIN(mjNGROUP-1, sim->m->actuator_group[i]))]) {
      continue;
    }

    // set data and name
    defSlider[0].pdata = sim->d->ctrl + i;
    if (sim->m->names[sim->m->name_actuatoradr[i]]) {
      mju::strcpy_arr(defSlider[0].name, sim->m->names+sim->m->name_actuatoradr[i]);
    } else {
      mju::sprintf_arr(defSlider[0].name, "control %d", i);
    }

    // set range
    if (sim->m->actuator_ctrllimited[i])
      mju::sprintf_arr(defSlider[0].other, "%.4g %.4g",
                       sim->m->actuator_ctrlrange[2*i], sim->m->actuator_ctrlrange[2*i+1]);
    else {
      mju::strcpy_arr(defSlider[0].other, "-1 1");
    }

    // add and count
    mjui_add(&sim->ui1, defSlider);
    itemcnt++;
  }
}

// make model-dependent UI sections
void MakeUiSections(mj::Simulate* sim) {
  // get section open-close state, UI 0
  int oldstate0[NSECT0];
  for (int i=0; i<NSECT0; i++) {
    oldstate0[i] = 0;
    if (sim->ui0.nsect>i) {
      oldstate0[i] = sim->ui0.sect[i].state;
    }
  }

  // get section open-close state, UI 1
  int oldstate1[NSECT1];
  for (int i=0; i<NSECT1; i++) {
    oldstate1[i] = 0;
    if (sim->ui1.nsect>i) {
      oldstate1[i] = sim->ui1.sect[i].state;
    }
  }

  // clear model-dependent sections of UI
  sim->ui0.nsect = SECT_TASK;
  sim->ui1.nsect = 0;

  // make
  sim->agent->GUI(sim->ui0);
  MakePhysicsSection(sim, oldstate0[SECT_PHYSICS]);
  MakeRenderingSection(sim, oldstate0[SECT_RENDERING]);
  MakeGroupSection(sim, oldstate0[SECT_GROUP]);
  MakeJointSection(sim, oldstate1[SECT_JOINT]);
  MakeControlSection(sim, oldstate1[SECT_CONTROL]);
}

//---------------------------------- utility functions ---------------------------------------------

// align and scale view
void AlignAndScaleView(mj::Simulate* sim) {
  // use default free camera parameters
  mjv_defaultFreeCamera(sim->m, &sim->cam);
}


// copy qpos to clipboard as key
void CopyKey(mj::Simulate* sim) {
  char clipboard[5000] = "<key qpos='";
  char buf[200];

  // prepare string
  for (int i=0; i<sim->m->nq; i++) {
    mju::sprintf_arr(buf, i==sim->m->nq-1 ? "%g" : "%g ", sim->d->qpos[i]);
    mju::strcat_arr(clipboard, buf);
  }
  mju::strcat_arr(clipboard, "'/>");

  // copy to clipboard
  sim->platform_ui->SetClipboardString(clipboard);
}

// millisecond timer, for MuJoCo built-in profiler
mjtNum Timer() {
  return Milliseconds(mj::Simulate::Clock::now().time_since_epoch()).count();
}

// clear all times
void ClearTimeres(mjData* d) {
  for (int i = 0; i < mjNTIMER; i++) {
    d->timer[i].duration = 0;
    d->timer[i].number = 0;
  }
}

// copy current camera to clipboard as MJCF specification
void CopyCamera(mj::Simulate* sim) {
  mjvGLCamera* camera = sim->scn.camera;

  char clipboard[500];
  mjtNum cam_right[3];
  mjtNum cam_forward[3];
  mjtNum cam_up[3];

  // get camera spec from the GLCamera
  mju_f2n(cam_forward, camera[0].forward, 3);
  mju_f2n(cam_up, camera[0].up, 3);
  mju_cross(cam_right, cam_forward, cam_up);

  // make MJCF camera spec
  mju::sprintf_arr(clipboard,
                   "<camera pos=\"%.3f %.3f %.3f\" xyaxes=\"%.3f %.3f %.3f %.3f %.3f %.3f\"/>\n",
                   (camera[0].pos[0] + camera[1].pos[0]) / 2,
                   (camera[0].pos[1] + camera[1].pos[1]) / 2,
                   (camera[0].pos[2] + camera[1].pos[2]) / 2,
                   cam_right[0], cam_right[1], cam_right[2],
                   camera[0].up[0], camera[0].up[1], camera[0].up[2]);

  // copy spec into clipboard
  sim->platform_ui->SetClipboardString(clipboard);
}

// update UI 0 when MuJoCo structures change (except for joint sliders)
void UpdateSettings(mj::Simulate* sim) {
  // physics flags
  for (int i=0; i<mjNDISABLE; i++) {
    sim->disable[i] = ((sim->m->opt.disableflags & (1<<i)) !=0);
  }
  for (int i=0; i<mjNENABLE; i++) {
    sim->enable[i] = ((sim->m->opt.enableflags & (1<<i)) !=0);
  }

  // camera
  if (sim->cam.type==mjCAMERA_FIXED) {
    sim->camera = 2 + sim->cam.fixedcamid;
  } else if (sim->cam.type==mjCAMERA_TRACKING) {
    sim->camera = 1;
  } else {
    sim->camera = 0;
  }

  // update UI
  mjui_update(-1, -1, &sim->ui0, &sim->uistate, &sim->platform_ui->mjr_context());
}


// Compute suitable font scale.
int ComputeFontScale(const mj::PlatformUIAdapter& platform_ui) {
  // compute framebuffer-to-window ratio
  auto [buf_width, buf_height] = platform_ui.GetFramebufferSize();
  auto [win_width, win_height] = platform_ui.GetWindowSize();
  double b2w = static_cast<double>(buf_width) / win_width;

  // compute PPI
  double PPI = b2w * platform_ui.GetDisplayPixelsPerInch();

  // estimate font scaling, guard against unrealistic PPI
  int fs;
  if (buf_width > win_width) {
    fs = mju_round(b2w * 100);
  } else if (PPI>50 && PPI<350) {
    fs = mju_round(PPI);
  } else {
    fs = 150;
  }
  fs = mju_round(fs * 0.02) * 50;
  fs = mjMIN(300, mjMAX(100, fs));

  return fs;
}

//---------------------------------- UI handlers ---------------------------------------------------

// determine enable/disable item state given category
int UiPredicate(int category, void* userdata) {
  mj::Simulate* sim = static_cast<mj::Simulate*>(userdata);

  switch (category) {
  case 2:                 // require model
    return (sim->m != nullptr);

  case 3:                 // require model and nkey
    return (sim->m && sim->m->nkey);

  case 4:                 // require model and paused
    return (sim->m && !sim->run);

  default:
    return 1;
  }
}

// set window layout
void UiLayout(mjuiState* state) {
  mj::Simulate* sim = static_cast<mj::Simulate*>(state->userdata);
  mjrRect* rect = state->rect;

  // set number of rectangles
  state->nrect = 4;

  // rect 1: UI 0
  rect[1].left = 0;
  rect[1].width = sim->ui0_enable ? sim->ui0.width : 0;
  rect[1].bottom = 0;
  rect[1].height = rect[0].height;

  // rect 2: UI 1
  rect[2].width = sim->ui1_enable ? sim->ui1.width : 0;
  rect[2].left = mjMAX(0, rect[0].width - rect[2].width);
  rect[2].bottom = 0;
  rect[2].height = rect[0].height;

  // rect 3: 3D plot (everything else is an overlay)
  rect[3].left = rect[1].width;
  rect[3].width = mjMAX(0, rect[0].width - rect[1].width - rect[2].width);
  rect[3].bottom = 0;
  rect[3].height = rect[0].height;
}

void UiModify(mjUI* ui, mjuiState* state, mjrContext* con) {
  mjui_resize(ui, con);
  mjr_addAux(ui->auxid, ui->width, ui->maxheight, ui->spacing.samples, con);
  UiLayout(state);
  mjui_update(-1, -1, ui, state, con);
}

// handle UI event
void UiEvent(mjuiState* state) {
  mj::Simulate* sim = static_cast<mj::Simulate*>(state->userdata);
  mjModel* m = sim->m;
  mjData* d = sim->d;
  int i;
  char err[200];

  // call UI 0 if event is directed to it
  if ((state->dragrect==sim->ui0.rectid) ||
      (state->dragrect==0 && state->mouserect==sim->ui0.rectid) ||
      state->type==mjEVENT_KEY) {
    // process UI event
    mjuiItem* it = mjui_event(&sim->ui0, state, &sim->platform_ui->mjr_context());

    // file section
    if (it && it->sectionid==SECT_FILE) {
      switch (it->itemid) {
      case 0:             // Save xml
        {
          const std::string path = GetSavePath("mjmodel.xml");
          if (!path.empty() && !mj_saveLastXML(path.c_str(), m, err, 200)) {
            std::printf("Save XML error: %s", err);
          }
        }
        break;

      case 1:             // Save mjb
        {
          const std::string path = GetSavePath("mjmodel.mjb");
          if (!path.empty()) {
            mj_saveModel(m, path.c_str(), nullptr, 0);
          }
        }
        break;

      case 2:             // Print model
        mj_printModel(m, "MJMODEL.TXT");
        break;

      case 3:             // Print data
        mj_printData(m, d, "MJDATA.TXT");
        break;

      case 4:             // Quit
        sim->exitrequest.store(1);
        break;

      case 5:             // Screenshot
        sim->screenshotrequest.store(true);
        break;
      }
    } else if (it && it->sectionid == SECT_OPTION) {
      if (it->pdata == &sim->spacing) {
        sim->ui0.spacing = mjui_themeSpacing(sim->spacing);
        sim->ui1.spacing = mjui_themeSpacing(sim->spacing);
      } else if (it->pdata == &sim->color) {
        sim->ui0.color = mjui_themeColor(sim->color);
        sim->ui1.color = mjui_themeColor(sim->color);
      } else if (it->pdata == &sim->font) {
        mjr_changeFont(50 * (sim->font + 1), &sim->platform_ui->mjr_context());
      } else if (it->pdata == &sim->fullscreen) {
        sim->platform_ui->ToggleFullscreen();
      } else if (it->pdata == &sim->vsync) {
        sim->platform_ui->SetVSync(sim->vsync);
      }

      // modify UI
      UiModify(&sim->ui0, state, &sim->platform_ui->mjr_context());
      UiModify(&sim->ui1, state, &sim->platform_ui->mjr_context());

    } else if (it && it->sectionid == SECT_SIMULATION) {
      switch (it->itemid) {
      case 1:             // Reset
        if (m) {
          mj_resetDataKeyframe(m, d, mj_name2id(m, mjOBJ_KEY, "home"));
          mj_forward(m, d);
          UpdateProfiler(sim);
          UpdateSensor(sim);
          UpdateSettings(sim);
          sim->agent->PlotReset();
        }
        break;

      case 2:             // Reload
        sim->uiloadrequest.fetch_add(1);
        break;

      case 3:             // Align
        AlignAndScaleView(sim);
        UpdateSettings(sim);
        break;

      case 4:             // Copy pose
        CopyKey(sim);
        break;

      case 5:             // Adjust key
      case 6:             // Load key
        i = sim->key;
        d->time = m->key_time[i];
        mju_copy(d->qpos, m->key_qpos+i*m->nq, m->nq);
        mju_copy(d->qvel, m->key_qvel+i*m->nv, m->nv);
        mju_copy(d->act, m->key_act+i*m->na, m->na);
        mju_copy(d->mocap_pos, m->key_mpos+i*3*m->nmocap, 3*m->nmocap);
        mju_copy(d->mocap_quat, m->key_mquat+i*4*m->nmocap, 4*m->nmocap);
        mju_copy(d->ctrl, m->key_ctrl+i*m->nu, m->nu);
        mj_forward(m, d);
        UpdateProfiler(sim);
        UpdateSensor(sim);
        UpdateSettings(sim);
        break;

      case 7:             // Save key
        i = sim->key;
        m->key_time[i] = d->time;
        mju_copy(m->key_qpos+i*m->nq, d->qpos, m->nq);
        mju_copy(m->key_qvel+i*m->nv, d->qvel, m->nv);
        mju_copy(m->key_act+i*m->na, d->act, m->na);
        mju_copy(m->key_mpos+i*3*m->nmocap, d->mocap_pos, 3*m->nmocap);
        mju_copy(m->key_mquat+i*4*m->nmocap, d->mocap_quat, 4*m->nmocap);
        mju_copy(m->key_ctrl+i*m->nu, d->ctrl, m->nu);
        break;
      }
    }

    // task section
    else if (it && it->sectionid == SECT_TASK) {
      sim->agent->TaskEvent(it, sim->d, sim->uiloadrequest, sim->run);
    }

    // agent section
    else if (it && it->sectionid == SECT_AGENT) {
      sim->agent->AgentEvent(it, sim->d, sim->uiloadrequest, sim->run);
    }

    // estimator section
    else if (it && it->sectionid == SECT_ESTIMATOR) {
      sim->agent->EstimatorEvent(it, sim->d, sim->uiloadrequest, sim->run);
    }

    // physics section
    else if (it && it->sectionid==SECT_PHYSICS) {
      // update disable flags in mjOption
      m->opt.disableflags = 0;
      for (i=0; i<mjNDISABLE; i++)
        if (sim->disable[i]) {
          m->opt.disableflags |= (1<<i);
        }

      // update enable flags in mjOption
      m->opt.enableflags = 0;
      for (i=0; i<mjNENABLE; i++)
        if (sim->enable[i]) {
          m->opt.enableflags |= (1<<i);
        }
    }

    // rendering section
    else if (it && it->sectionid==SECT_RENDERING) {
      // set camera in mjvCamera
      if (sim->camera==0) {
        sim->cam.type = mjCAMERA_FREE;
      } else if (sim->camera==1) {
        if (sim->pert.select>0) {
          sim->cam.type = mjCAMERA_TRACKING;
          sim->cam.trackbodyid = sim->pert.select;
          sim->cam.fixedcamid = -1;
        } else {
          sim->cam.type = mjCAMERA_FREE;
          sim->camera = 0;
          mjui_update(SECT_RENDERING, -1, &sim->ui0, &sim->uistate,
                      &sim->platform_ui->mjr_context());
        }
      } else {
        sim->cam.type = mjCAMERA_FIXED;
        sim->cam.fixedcamid = sim->camera - 2;
      }
      // copy camera spec to clipboard (as MJCF element)
      if (it->itemid == 3) {
        CopyCamera(sim);
      }
    }

    // group section
    else if (it && it->sectionid==SECT_GROUP) {
      // remake joint section if joint group changed
      if (it->name[0]=='J' && it->name[1]=='o') {
        sim->ui1.nsect = SECT_JOINT;
        MakeJointSection(sim, sim->ui1.sect[SECT_JOINT].state);
        sim->ui1.nsect = NSECT1;
        UiModify(&sim->ui1, state, &sim->platform_ui->mjr_context());
      }

      // remake control section if actuator group changed
      if (it->name[0]=='A' && it->name[1]=='c') {
        sim->ui1.nsect = SECT_CONTROL;
        MakeControlSection(sim, sim->ui1.sect[SECT_CONTROL].state);
        sim->ui1.nsect = NSECT1;
        UiModify(&sim->ui1, state, &sim->platform_ui->mjr_context());
      }
    }

    // stop if UI processed event
    if (it!=nullptr || (state->type==mjEVENT_KEY && state->key==0)) {
      return;
    }
  }

  // call UI 1 if event is directed to it
  if ((state->dragrect==sim->ui1.rectid) ||
      (state->dragrect==0 && state->mouserect==sim->ui1.rectid) ||
      state->type==mjEVENT_KEY) {
    // process UI event
    mjuiItem* it = mjui_event(&sim->ui1, state, &sim->platform_ui->mjr_context());

    // control section
    if (it && it->sectionid==SECT_CONTROL) {
      // clear controls
      if (it->itemid==0) {
        mju_zero(d->ctrl, m->nu);
        mjui_update(SECT_CONTROL, -1, &sim->ui1, &sim->uistate, &sim->platform_ui->mjr_context());
      }
    }

    // stop if UI processed event
    if (it!=nullptr || (state->type==mjEVENT_KEY && state->key==0)) {
      return;
    }
  }

  // shortcut not handled by UI
  if (state->type==mjEVENT_KEY && state->key!=0) {
    switch (state->key) {
    case ' ':                   // Mode
      if (m) {
        sim->run = 1 - sim->run;
        sim->pert.active = 0;
        mjui_update(-1, -1, &sim->ui0, state, &sim->platform_ui->mjr_context());
      }
      break;

    case mjKEY_RIGHT:           // step forward
      if (m && !sim->run) {
        ClearTimeres(d);
        mj_step(m, d);
        UpdateProfiler(sim);
        UpdateSensor(sim);
        UpdateSettings(sim);
      }
      break;

    case mjKEY_PAGE_UP:         // select parent body
      if (m && sim->pert.select>0) {
        sim->pert.select = m->body_parentid[sim->pert.select];
        sim->pert.skinselect = -1;

        // stop perturbation if world reached
        if (sim->pert.select<=0) {
          sim->pert.active = 0;
        }
      }

      break;

    case ']':                   // cycle up fixed cameras
      if (m && m->ncam) {
        sim->cam.type = mjCAMERA_FIXED;
        // simulate->camera = {0 or 1} are reserved for the free and tracking cameras
        if (sim->camera < 2 || sim->camera == 2 + m->ncam-1) {
          sim->camera = 2;
        } else {
          sim->camera += 1;
        }
        sim->cam.fixedcamid = sim->camera - 2;
        mjui_update(SECT_RENDERING, -1, &sim->ui0, &sim->uistate, &sim->platform_ui->mjr_context());
      }
      break;

    case '[':                   // cycle down fixed cameras
      if (m && m->ncam) {
        sim->cam.type = mjCAMERA_FIXED;
        // settings.camera = {0 or 1} are reserved for the free and tracking cameras
        if (sim->camera <= 2) {
          sim->camera = 2 + m->ncam-1;
        } else {
          sim->camera -= 1;
        }
        sim->cam.fixedcamid = sim->camera - 2;
        mjui_update(SECT_RENDERING, -1, &sim->ui0, &sim->uistate, &sim->platform_ui->mjr_context());
      }
      break;

    case mjKEY_F6:                   // cycle frame visualisation
      if (m) {
        sim->opt.frame = (sim->opt.frame + 1) % mjNFRAME;
        mjui_update(SECT_RENDERING, -1, &sim->ui0, &sim->uistate, &sim->platform_ui->mjr_context());
      }
      break;

    case mjKEY_F7:                   // cycle label visualisation
      if (m) {
        sim->opt.label = (sim->opt.label + 1) % mjNLABEL;
        mjui_update(SECT_RENDERING, -1, &sim->ui0, &sim->uistate, &sim->platform_ui->mjr_context());
      }
      break;

    case mjKEY_ESCAPE:          // free camera
      sim->cam.type = mjCAMERA_FREE;
      sim->camera = 0;
      mjui_update(SECT_RENDERING, -1, &sim->ui0, &sim->uistate, &sim->platform_ui->mjr_context());
      break;

    case '-':                   // slow down
      {
        int numclicks = sizeof(sim->percentRealTime) / sizeof(sim->percentRealTime[0]);
        if (sim->real_time_index < numclicks-1 && !state->shift) {
          sim->real_time_index++;
          sim->speed_changed = true;
        }
      }
      break;

    case '=':                   // speed up
      if (sim->real_time_index > 0 && !state->shift) {
        sim->real_time_index--;
        sim->speed_changed = true;
      }
      break;

    // agent keys
    case mjKEY_ENTER:
      sim->agent->plan_enabled = !sim->agent->plan_enabled;
      break;

    case '\\':
      sim->agent->action_enabled = !sim->agent->action_enabled;
      break;

    case '9':
      sim->agent->visualize_enabled = !sim->agent->visualize_enabled;
      break;
    }

    return;
  }

  // 3D scroll
  if (state->type==mjEVENT_SCROLL && state->mouserect==3 && m) {
    // emulate vertical mouse motion = 2% of window height
    mjv_moveCamera(m, mjMOUSE_ZOOM, 0, -zoom_increment*state->sy, &sim->scn, &sim->cam);

    return;
  }

  // 3D press
  if (state->type==mjEVENT_PRESS && state->mouserect==3 && m) {
    // set perturbation
    int newperturb = 0;
    if (state->control && sim->pert.select>0) {
      // right: translate;  left: rotate
      if (state->right) {
        newperturb = mjPERT_TRANSLATE;
      } else if (state->left) {
        newperturb = mjPERT_ROTATE;
      }

      // perturbation onset: reset reference
      if (newperturb && !sim->pert.active) {
        mjv_initPerturb(m, d, &sim->scn, &sim->pert);
      }
    }
    sim->pert.active = newperturb;

    // handle double-click
    if (state->doubleclick) {
      // determine selection mode
      int selmode;
      if (state->button==mjBUTTON_LEFT) {
        selmode = 1;
      } else if (state->control) {
        selmode = 3;
      } else {
        selmode = 2;
      }

      // find geom and 3D click point, get corresponding body
      mjrRect r = state->rect[3];
      mjtNum selpnt[3];
      int selgeom, selflex, selskin;
      int selbody = mjv_select(m, d, &sim->opt,
                               static_cast<mjtNum>(r.width)/r.height,
                               (state->x - r.left)/r.width,
                               (state->y - r.bottom)/r.height,
                               &sim->scn, selpnt, &selgeom, &selflex, &selskin);

      // set lookat point, start tracking is requested
      if (selmode==2 || selmode==3) {
        // copy selpnt if anything clicked
        if (selbody>=0) {
          mju_copy3(sim->cam.lookat, selpnt);
        }

        // switch to tracking camera if dynamic body clicked
        if (selmode==3 && selbody>0) {
          // mujoco camera
          sim->cam.type = mjCAMERA_TRACKING;
          sim->cam.trackbodyid = selbody;
          sim->cam.fixedcamid = -1;

          // UI camera
          sim->camera = 1;
          mjui_update(SECT_RENDERING, -1, &sim->ui0, &sim->uistate, &sim->platform_ui->mjr_context());
        }
      }

      // set body selection
      else {
        if (selbody>=0) {
          // record selection
          sim->pert.select = selbody;
          sim->pert.skinselect = selskin;
          sim->pert.flexselect = selflex;

          // compute localpos
          mjtNum tmp[3];
          mju_sub3(tmp, selpnt, d->xpos+3*sim->pert.select);
          mju_mulMatTVec(sim->pert.localpos, d->xmat+9*sim->pert.select, tmp, 3, 3);
        } else {
          sim->pert.select = 0;
          sim->pert.skinselect = -1;
          sim->pert.flexselect = -1;
        }
      }

      // stop perturbation on select
      sim->pert.active = 0;
    }

    return;
  }

  // 3D release
  if (state->type==mjEVENT_RELEASE && state->dragrect==3 && m) {
    // stop perturbation
    sim->pert.active = 0;

    return;
  }

  // 3D move
  if (state->type==mjEVENT_MOVE && state->dragrect==3 && m) {
    // determine action based on mouse button
    mjtMouse action;
    if (state->right) {
      action = state->shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
    } else if (state->left) {
      action = state->shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
    } else {
      action = mjMOUSE_ZOOM;
    }

    // move perturb or camera
    mjrRect r = state->rect[3];
    if (sim->pert.active)
      mjv_movePerturb(m, d, action, state->dx/r.height, -state->dy/r.height,
                      &sim->scn, &sim->pert);
    else
      mjv_moveCamera(m, action, state->dx/r.height, -state->dy/r.height,
                     &sim->scn, &sim->cam);

    return;
  }

  // Dropped files
  if (state->type == mjEVENT_FILESDROP && state->dropcount > 0) {
    while (sim->droploadrequest.load()) {}
    sim->dropfilename = state->droppaths[0];
    sim->droploadrequest.store(true);
    return;
  }

  // Redraw
  if (state->type == mjEVENT_REDRAW) {
    sim->Render();
    return;
  }
}
}  // namespace

namespace mujoco {
namespace mju = ::mujoco::util_mjpc;

Simulate::Simulate(std::unique_ptr<PlatformUIAdapter> platform_ui,
                   std::shared_ptr<mjpc::Agent> a)
    : platform_ui(std::move(platform_ui)),
      uistate(this->platform_ui->state()),
      agent(std::move(a)) {}

//------------------------------------ apply pose perturbations ------------------------------------
void Simulate::ApplyPosePerturbations(int flg_paused) {
  if (this->m != nullptr) {
    mjv_applyPerturbPose(this->m, this->d, &this->pert, flg_paused);  // move mocap bodies only
  }
}

//----------------------------------- apply force perturbations ------------------------------------
void Simulate::ApplyForcePerturbations() {
  if (this->m != nullptr) {
    mjv_applyPerturbForce(this->m, this->d, &this->pert);
  }
}

//------------------------- Tell the render thread to load a file and wait -------------------------
void Simulate::Load(mjModel* m,
                    mjData* d,
                    std::string displayed_filename,
                    bool delete_old_m_d) {
  this->mnew = m;
  this->dnew = d;
  this->delete_old_m_d = delete_old_m_d;
  this->filename = std::move(displayed_filename);

  {
    std::unique_lock<std::mutex> lock(mtx);
    this->loadrequest = 2;

    // Wait for the render thread to be done loading
    // so that we know the old model and data's memory can
    // be free'd by the other thread (sometimes python)
    cond_loadrequest.wait(lock, [this]() { return this->loadrequest == 0; });
  }
}

//------------------------------------- load mjb or xml model --------------------------------------
void Simulate::LoadOnRenderThread() {
  if (this->delete_old_m_d) {
    // delete old model if requested
    if (this->d) {
      mj_deleteData(d);
    }
    if (this->m) {
      mj_deleteModel(m);
    }
  }

  this->m = this->mnew;
  this->d = this->dnew;

  // re-create scene and context
  mjv_makeScene(this->m, &this->scn, maxgeom);
  if (!this->platform_ui->IsGPUAccelerated()) {
    this->scn.flags[mjRND_SHADOW] = 0;
    this->scn.flags[mjRND_REFLECTION] = 0;
  }
  this->platform_ui->RefreshMjrContext(this->m, 50*(this->font+1));

  // clear perturbation state
  this->pert.active = 0;
  this->pert.select = 0;
  this->pert.skinselect = -1;

  // align and scale view unless reloading the same file
  if (this->filename != this->previous_filename) {
    AlignAndScaleView(this);
    this->previous_filename = this->filename;
  }

  // update scene
  mjv_updateScene(this->m, this->d, &this->opt, &this->pert, &this->cam, mjCAT_ALL, &this->scn);

  // set window title to model name
  if (this->m->names) {
    char title[200] = "MuJoCo MPC : ";
    mju::strcat_arr(title, this->m->names);
    platform_ui->SetWindowTitle(title);
  }

  // set keyframe range and divisions
  this->ui0.sect[SECT_SIMULATION].item[5].slider.range[0] = 0;
  this->ui0.sect[SECT_SIMULATION].item[5].slider.range[1] = mjMAX(0, this->m->nkey - 1);
  this->ui0.sect[SECT_SIMULATION].item[5].slider.divisions = mjMAX(1, this->m->nkey - 1);

  // rebuild UI sections
  MakeUiSections(this);

  // full ui update
  UiModify(&this->ui0, &this->uistate, &this->platform_ui->mjr_context());
  UiModify(&this->ui1, &this->uistate, &this->platform_ui->mjr_context());
  UpdateSettings(this);

  // clear request
  this->loadrequest = 0;
  cond_loadrequest.notify_all();
}

//------------------------------------------- rendering --------------------------------------------


// prepare to render
void Simulate::PrepareScene() {
  // data for FPS calculation
  static std::chrono::time_point<Clock> lastupdatetm;

  // update interval, save update time
  auto tmnow = Clock::now();
  double interval = Seconds(tmnow - lastupdatetm).count();
  interval = mjMIN(1, mjMAX(0.0001, interval));
  lastupdatetm = tmnow;

  // no model: nothing to do
  if (!this->m) {
    return;
  }

  // update scene
  mjv_updateScene(this->m, this->d, &this->opt, &this->pert, &this->cam, mjCAT_ALL, &this->scn);

  // update watch
  if (this->ui0_enable && this->ui0.sect[SECT_WATCH].state) {
    UpdateWatch(this);
    mjui_update(SECT_WATCH, -1, &this->ui0, &this->uistate, &this->platform_ui->mjr_context());
  }

  // update joint
  if (this->ui1_enable && this->ui1.sect[SECT_JOINT].state) {
    mjui_update(SECT_JOINT, -1, &this->ui1, &this->uistate, &this->platform_ui->mjr_context());
  }

  // update info text
  if (this->info) {
    UpdateInfoText(this, this->info_title, this->info_content, interval);
  }

  // update control
  if (this->ui1_enable && this->ui1.sect[SECT_CONTROL].state) {
    mjui_update(SECT_CONTROL, -1, &this->ui1, &this->uistate, &this->platform_ui->mjr_context());
  }

  // update profiler
  if (this->profiler && this->run) {
    UpdateProfiler(this);
  }

  // update sensor
  if (this->sensor && this->run) {
    UpdateSensor(this);
  }

  // update task
  if (this->ui0_enable && this->ui0.sect[SECT_TASK].state) {
    if (!this->agent->allocate_enabled && this->uiloadrequest.load() == 0) {
      mjui_update(SECT_TASK, -1, &this->ui0, &this->uistate, &this->platform_ui->mjr_context());
    }
  }

  // update agent
  if (this->ui0_enable && this->ui0.sect[SECT_AGENT].state) {
    mjui_update(SECT_AGENT, -1, &this->ui0, &this->uistate, &this->platform_ui->mjr_context());
  }

  // update agent profiler
  if (this->agent->plot_enabled && this->uiloadrequest.load() == 0) {
    this->agent->Plots(this->d, this->run);
  }

  // clear timers once profiler info has been copied
  ClearTimeres(this->d);
}

// render the ui to the window
void Simulate::Render() {
  if (this->platform_ui->RefreshMjrContext(this->m, 50*(this->font+1))) {
    UiModify(&this->ui0, &this->uistate, &this->platform_ui->mjr_context());
    UiModify(&this->ui1, &this->uistate, &this->platform_ui->mjr_context());
  }

  // get 3D rectangle and reduced for profiler
  mjrRect rect = this->uistate.rect[3];
  mjrRect smallrect = rect;
  if (this->profiler) {
    smallrect.width = rect.width - rect.width/4;
  }

  // no model
  if (!this->m) {
    // blank screen
    mjr_rectangle(rect, 0.2f, 0.3f, 0.4f, 1);

    // label
    if (this->loadrequest) {
      mjr_overlay(mjFONT_BIG, mjGRID_TOPRIGHT, smallrect, "loading", nullptr,
                  &this->platform_ui->mjr_context());
    } else {
      char intro_message[Simulate::kMaxFilenameLength];
      mju::sprintf_arr(intro_message,
                       "MuJoCo version %s\nDrag-and-drop model file here", mj_versionString());
      mjr_overlay(mjFONT_NORMAL, mjGRID_TOPLEFT, rect, intro_message, 0,
                  &this->platform_ui->mjr_context());
    }

    // show last loading error
    if (this->load_error[0]) {
      mjr_overlay(mjFONT_NORMAL, mjGRID_BOTTOMLEFT, rect, this->load_error, 0,
                  &this->platform_ui->mjr_context());
    }

    // render uis
    if (this->ui0_enable) {
      mjui_render(&this->ui0, &this->uistate, &this->platform_ui->mjr_context());
    }
    if (this->ui1_enable) {
      mjui_render(&this->ui1, &this->uistate, &this->platform_ui->mjr_context());
    }

    // finalize
    this->platform_ui->SwapBuffers();

    return;
  }

  // visualization
  if (this->uiloadrequest.load() == 0) {
    // task-specific
    if (this->agent->ActiveTask()->visualize) {
      this->agent->ActiveTask()->ModifyScene(this->m, this->d, &this->scn);
    }
    // common to all tasks
    this->agent->ModifyScene(&this->scn);
  }

  // render scene
  mjr_render(rect, &this->scn, &this->platform_ui->mjr_context());

  // show last loading error
  if (this->load_error[0]) {
    mjr_overlay(mjFONT_NORMAL, mjGRID_BOTTOMLEFT, rect, this->load_error, 0,
                &this->platform_ui->mjr_context());
  }

  // make pause/loading label
  std::string pauseloadlabel;
  if (!this->run || this->loadrequest) {
    pauseloadlabel = this->loadrequest ? "loading" : "pause";
  }

  // get desired and actual percent-of-real-time
  float desiredRealtime = this->percentRealTime[this->real_time_index];
  float actualRealtime = 100 / this->measured_slowdown;

  // if running, check for misalignment of more than 10%
  float realtime_offset = mju_abs(actualRealtime - desiredRealtime);
  bool misaligned = this->run && realtime_offset > 0.1 * desiredRealtime;

  // make realtime overlay label
  char rtlabel[30] = {'\0'};
  if (desiredRealtime != 100.0 || misaligned) {
    // print desired realtime
    int labelsize = std::snprintf(rtlabel,
                                  sizeof(rtlabel), "%g%%", desiredRealtime);

    // if misaligned, append to label
    if (misaligned) {
      std::snprintf(rtlabel+labelsize,
                    sizeof(rtlabel)-labelsize, " (%-4.1f%%)", actualRealtime);
    }
  }

  // draw top left overlay
  if (!pauseloadlabel.empty() || rtlabel[0]) {
    std::string newline = !pauseloadlabel.empty() && rtlabel[0] ? "\n" : "";
    std::string topleftlabel = rtlabel + newline + pauseloadlabel;
    mjr_overlay(mjFONT_BIG, mjGRID_TOPLEFT, smallrect,
                topleftlabel.c_str(), nullptr, &this->platform_ui->mjr_context());
  }

  // show ui 0
  if (this->ui0_enable) {
    mjui_render(&this->ui0, &this->uistate, &this->platform_ui->mjr_context());
  }

  // show ui 1
  if (this->ui1_enable) {
    mjui_render(&this->ui1, &this->uistate, &this->platform_ui->mjr_context());
  }

  // show help
  if (this->help) {
    mjr_overlay(mjFONT_NORMAL, mjGRID_TOPLEFT, rect, help_title, help_content,
                &this->platform_ui->mjr_context());
  }

  // show info
  if (this->info) {
    mjr_overlay(mjFONT_NORMAL, mjGRID_BOTTOMLEFT, rect, this->info_title, this->info_content,
                &this->platform_ui->mjr_context());
  }

  // show profiler
  if (this->profiler) {
    ShowProfiler(this, rect);
  }

  // show sensor
  if (this->sensor) {
    ShowSensor(this, smallrect);
  }

  // show agent plots
  if (this->agent->plot_enabled && this->uiloadrequest.load() == 0) {
    this->agent->PlotShow(&smallrect, &this->platform_ui->mjr_context());
  }

  // take screenshot, save to file
  if (this->screenshotrequest.exchange(false)) {
    const unsigned int h = uistate.rect[0].height;
    const unsigned int w = uistate.rect[0].width;
    std::unique_ptr<unsigned char[]> rgb(new unsigned char[3*w*h]);
    if (!rgb) {
      mju_error("could not allocate buffer for screenshot");
    }
    mjr_readPixels(rgb.get(), nullptr, uistate.rect[0], &this->platform_ui->mjr_context());

    // flip up-down
    for (int r = 0; r < h/2; ++r) {
      unsigned char* top_row = &rgb[3*w*r];
      unsigned char* bottom_row = &rgb[3*w*(h-1-r)];
      std::swap_ranges(top_row, top_row+3*w, bottom_row);
    }

    // save as PNG
    // TODO(b/241577466): Parse the stem of the filename and use a .PNG extension.
    // Unfortunately, if we just yank ".xml"/".mjb" from the filename and append .PNG, the macOS
    // file dialog does not automatically open that location. Thus, we defer to a default
    // "screenshot.png" for now.
    const std::string path = GetSavePath("screenshot.png");
    if (!path.empty()) {
      if (lodepng::encode(path, rgb.get(), w, h, LCT_RGB)) {
        mju_error("could not save screenshot");
      } else {
        std::printf("saved screenshot: %s\n", path.c_str());
      }
    }
  }

  // finalize
  this->platform_ui->SwapBuffers();
}

void Simulate::InitializeRenderLoop() {
  // Set timer callback (milliseconds)
  mjcb_time = Timer;

  // init abstract visualization
  mjv_defaultCamera(&this->cam);
  mjv_defaultOption(&this->opt);
  InitializeProfiler(this);
  InitializeSensor(this);

  // make empty scene
  mjv_defaultScene(&this->scn);
  mjv_makeScene(nullptr, &this->scn, maxgeom);
  if (!this->platform_ui->IsGPUAccelerated()) {
    this->scn.flags[mjRND_SHADOW] = 0;
    this->scn.flags[mjRND_REFLECTION] = 0;
  }

  // select default font
  int fontscale = ComputeFontScale(*this->platform_ui);
  this->font = fontscale/50 - 1;

  // make empty context
  this->platform_ui->RefreshMjrContext(nullptr, fontscale);

  // init state and uis
  std::memset(&this->uistate, 0, sizeof(mjuiState));
  std::memset(&this->ui0, 0, sizeof(mjUI));
  std::memset(&this->ui1, 0, sizeof(mjUI));

  auto [buf_width, buf_height] = this->platform_ui->GetFramebufferSize();
  this->uistate.nrect = 1;
  this->uistate.rect[0].width = buf_width;
  this->uistate.rect[0].height = buf_height;

  this->ui0.spacing = mjui_themeSpacing(this->spacing);
  this->ui0.color = mjui_themeColor(this->color);
  this->ui0.predicate = UiPredicate;
  this->ui0.rectid = 1;
  this->ui0.auxid = 0;

  this->ui1.spacing = mjui_themeSpacing(this->spacing);
  this->ui1.color = mjui_themeColor(this->color);
  this->ui1.predicate = UiPredicate;
  this->ui1.rectid = 2;
  this->ui1.auxid = 1;

  // set GUI adapter callbacks
  this->uistate.userdata = this;
  this->platform_ui->SetEventCallback(UiEvent);
  this->platform_ui->SetLayoutCallback(UiLayout);

  // populate uis with standard sections
  this->ui0.userdata = this;
  this->ui1.userdata = this;
  mjui_add(&this->ui0, defFile);
  mjui_add(&this->ui0, this->def_option);
  mjui_add(&this->ui0, this->def_simulation);
  mjui_add(&this->ui0, this->def_watch);
  UiModify(&this->ui0, &this->uistate, &this->platform_ui->mjr_context());
  UiModify(&this->ui1, &this->uistate, &this->platform_ui->mjr_context());

  // set VSync to initial value
  this->platform_ui->SetVSync(this->vsync);
}

void Simulate::RenderLoop() {
  // run event loop
  while (!this->platform_ui->ShouldCloseWindow() && !this->exitrequest.load()) {
    {
      const std::lock_guard<std::mutex> lock(this->mtx);

      // load model (not on first pass, to show "loading" label)
      if (this->loadrequest==1) {
        this->LoadOnRenderThread();
      } else if (this->loadrequest>1) {
        this->loadrequest = 1;
      }

      // poll and handle events
      this->platform_ui->PollEvents();

      // prepare to render
      this->PrepareScene();
    }  // std::lock_guard<std::mutex> (unblocks simulation thread)

    // render while simulation is running
    this->Render();
  }

  this->exitrequest.store(true);

  mjv_freeScene(&this->scn);
}

}  // namespace mujoco
