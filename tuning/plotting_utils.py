import matplotlib.pyplot as plt
import numpy as np
import math
from multiprocessing import Process, Queue
import time
import dearpygui.dearpygui as dpg
from collections import deque

DEQUE_MAX_LEN = 200

class _Plot:
    def __init__(self, id, name, window_id, y_range = None):
        self._id = id
        self._window_id = window_id
        self._name = name
        self._data_count = 0
        self._y_range = y_range
        self._xdata = deque(maxlen=DEQUE_MAX_LEN)
        self._ydata = deque(maxlen=DEQUE_MAX_LEN)
        
    def _initialize_gui(self):
        with dpg.plot(height=200, width=500):
            dpg.add_plot_axis(dpg.mvXAxis, label="Time", tag=f"{self._window_id}-{self._name}-xaxis", time=True, no_tick_labels=True)
            dpg.add_plot_axis(dpg.mvYAxis, label=self._name, tag=f"{self._window_id}-{self._name}-yaxis")
            dpg.add_line_series([], [], tag=f'{self._window_id}-{self._name}-line', parent=f"{self._window_id}-{self._name}-yaxis")
            if self._y_range:
                dpg.set_axis_limits(f"{self._window_id}-{self._name}-yaxis", self._y_range[0], self._y_range[1])
                
        
    def _update_plot(self, x_data, y_data):
        assert len(x_data) == len(y_data)
        self._xdata.extend(x_data)
        self._ydata.extend(y_data)
        dpg.configure_item(f'{self._window_id}-{self._name}-line', x=list(self._xdata), y=list(self._ydata))
        dpg.fit_axis_data(f"{self._window_id}-{self._name}-xaxis")
        if not self._y_range:
            dpg.fit_axis_data(f"{self._window_id}-{self._name}-yaxis")
    
class _Window:
    def __init__(self, id, shape: tuple[2], name: str):
        self._id = id
        self._grid_shape: tuple[2] = shape
        self._plots: list[_Plot] = []
        self._name = name
        
    @property
    def max_num_plots(self):
        return math.prod(self._grid_shape)
    
    @property
    def num_plots(self):
        return len(self._plots)
    
    def _initialize_gui(self):
        with dpg.tab(tag = self._name, label=self._name):
            with dpg.subplots(rows=self._grid_shape[0], columns=self._grid_shape[1], tag=f"{self._id}-subplots", height=1300, width=1100):
                for plot in self._plots:
                    plot._initialize_gui()
    
    def _add_plot(self, name, plot_id = -1, y_range = None):
        if plot_id == -1:
            plot_id = len(self._plots)
        assert plot_id < self.max_num_plots
        plot = _Plot(id = plot_id, name=name, window_id=self._id, y_range=y_range)
        self._plots.insert(plot_id, plot)
        return plot

    def _add_points(self, x_data, y_data: list[np.ndarray]):
        #assert (type(y_data) == list and len(points) == self.num_plots) or (type(points) == np.ndarray and points.shape[0] == self.num_plots)
        y_data = np.stack(y_data, axis=-1)
        for i, plot in enumerate(self._plots):
            plot._update_plot(x_data, y_data[i, :])

class PlotManager(Process):
    def __init__(self):
        super(PlotManager, self).__init__()
        self._windows: list[_Window] = []
        self._queues: list[Queue] = []
        self.daemon = True
    
    @property
    def num_windows(self):
        return len(self._windows)
    
    @property
    def num_plots(self):
        return sum(map(lambda window: window.num_plots, self._windows))
    
    def add_window(self, shape: tuple[2], name: str):
        w = _Window(id = len(self._windows), shape=shape, name=name)
        q = Queue(maxsize=DEQUE_MAX_LEN)
        self._windows.append(w)
        self._queues.append(q)
        return w._id
    
    def add_plot(self, window_id: int, plot_name: str, plot_range = None):
        return self._windows[window_id]._add_plot(plot_name, y_range=plot_range)
    
    def send_data(self, window_id: int, data: tuple[float, np.ndarray]):
        self._queues[window_id].put(data)
    
    def __initialize_gui(self):
        dpg.create_context()
        with dpg.window(tag = "Main Window"):
            with dpg.tab_bar():
                for window in self._windows:
                    window._initialize_gui()
    
    def __flush_queue(self, queue_id: int):
        tmp = []
        while not self._queues[queue_id].empty():
            tmp.append(self._queues[queue_id].get())
        if len(tmp) == 0:
            return
        x_data, y_data = zip(*tmp)
        self._windows[queue_id]._add_points(x_data, y_data)
    
    def run(self):
        self.__initialize_gui()
        dpg.create_viewport(width=1100, height=1300, title='Updating plot data')
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("Main Window", True)
        while dpg.is_dearpygui_running():
            start = time.perf_counter_ns()
            for i in range(self.num_windows):
                self.__flush_queue(i)
            dpg.render_dearpygui_frame()
            end = time.perf_counter_ns()
            print(f"Time taken to render frame: {(end - start) / 1e6} ms")
            print(f"FPS: {1 / ((end - start) / 1e9)}")
        dpg.destroy_context()

if __name__ == "__main__":
    manager = PlotManager()
    w = manager.add_window((2, 1), "Test window")
    manager.add_plot(w, "Foot distance")
    manager.add_plot(w, "Test distance")
    w = manager.add_window((1, 2), "Test window 2")
    manager.add_plot(w, "Foot distance")
    manager.add_plot(w, "Test distance")
    
    manager.start()
    for i in range(100000):
        time.sleep(0.02)
        manager.send_data(0, (time.time(), np.asarray([i, 100-i])))
        manager.send_data(1, (time.time(), np.asarray([i*2, np.sin(i/2.0)])))
    manager.join()