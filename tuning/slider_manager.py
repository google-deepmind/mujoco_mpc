import dearpygui.dearpygui as dpg
from multiprocessing import Process, Array
import time 

class SliderManager(Process):
    def __init__(self, name = "Slider Manager"):
        super(SliderManager, self).__init__()
        self.daemon = True
        self.name = name
        self.sliders_metadata = []
        self.__slider_values = None
        self.slider_values_local = []
    
    def __initialize_gui(self):
        print("Initializing sliders")
        for metadata in self.sliders_metadata:
            slider = dpg.add_slider_float(label=metadata["name"], default_value=metadata["default_value"], min_value=metadata["min_value"], max_value=metadata["max_value"], callback=self.__update_slider_values)
            metadata.update({"ref": slider})
        dpg.add_button(label="Reset", callback=self.__reset_sliders)

    def __update_slider_values(self):
        with self.__slider_values.get_lock():
            for i, metadata in enumerate(self.sliders_metadata):
                self.__slider_values[i] = dpg.get_value(metadata["ref"])
    
    def __reset_sliders(self):
        with self.__slider_values.get_lock():
            for i, metadata in enumerate(self.sliders_metadata):
                dpg.set_value(metadata["ref"], metadata["default_value"])
                self.__slider_values[i] = metadata["default_value"]
    
    def start(self) -> None:
        self.__slider_values = Array('f', [metadata["default_value"] for metadata in self.sliders_metadata])
        with self.__slider_values.get_lock():
            self.slider_values_local = self.__slider_values[:]
        return super().start()
    
    def run(self):
        dpg.create_context()

        with dpg.window(tag = "Main Window"):
            self.__initialize_gui()

        dpg.create_viewport(width=900, height=600, title='Updating plot data')
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_viewport_title(self.name)
        dpg.set_primary_window("Main Window", True)
        dpg.start_dearpygui()
        dpg.destroy_context()
        
    def add_slider(self, name: str, default_value: float, min_value: float, max_value: float):
        metadata = {
            "name": name,
            "default_value": default_value,
            "min_value": min_value,
            "max_value": max_value
        }
        self.sliders_metadata.append(metadata)
        
    def get_slider_values(self, blocking = True):
        lock = self.__slider_values.get_lock()
        is_lock_acquired = lock.acquire(block=blocking)
        if is_lock_acquired:
            self.slider_values_local = self.__slider_values[:]
            lock.release()
        return self.slider_values_local
        

if __name__ == "__main__":
    slider_manager = SliderManager()
    slider_manager.add_slider("Slider 1", 0.0, 0.0, 1.0)
    slider_manager.add_slider("Slider 2", 0.0, 0.0, 1.0)
    slider_manager.start()
    
    try:
        while True:
            print(slider_manager.get_slider_values(blocking=False))
            time.sleep(0.01)
    except KeyboardInterrupt:
        slider_manager.kill()
        slider_manager.join()