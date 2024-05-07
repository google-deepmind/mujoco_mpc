import mujoco

class Geometry:
    def __init__(self, geom_id, geom_type, geom_rgba):
        self.geom_id = geom_id
        self.geom_type = geom_type
        self.geom_rgba = geom_rgba
        self.geom_pos = [0, 0, 0]
        self.geom_mat = [1, 0, 0, 0, 1, 0, 0, 0, 1]
        self.geom_size = [1, 1, 1]
        
    def update(self, ref):
        raise NotImplementedError("update() must be implemented by the subclass")
    
class Line(Geometry):
    def __init__(self, geom_id, geom_rgba):
        super().__init__(geom_id, mujoco.mjtGeom.mjGEOM_LINE, geom_rgba)
        self.start = None
        self.end = None
        self.width = 1
    
    def update(self, ref):
        mujoco.mjv_connector(ref, mujoco.mjtGeom.mjGEOM_LINE, self.width, self.start(), self.end())
    

class GeomManager:
    def __init__(self, scene):
        if not isinstance(scene, mujoco._structs.MjvScene):
            raise ValueError("scene must be a mujoco.mjvScene")
        self.scene = scene
        self.geoms = []
    
    def add_geom(self, geom):
        self.geoms.append(geom)
        mujoco.mjv_initGeom(self.scene.geoms[self.scene.ngeom],
                            geom.geom_type,
                            geom.geom_size,
                            geom.geom_pos,
                            geom.geom_mat,
                            geom.geom_rgba)
        self.scene.ngeom += 1
        return geom
    
    def add_line(self, geom_rgba = [1, 0, 0, 1]):
        geom = Line(self.scene.ngeom, geom_rgba)
        return self.add_geom(geom)
    
    def update(self):
        for geom in self.geoms:
            geom.update(self.scene.geoms[geom.geom_id])