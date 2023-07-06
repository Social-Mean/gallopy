class BoundaryCondition(object):
    pass

class DirichletBoundaryCondition(BoundaryCondition):
    def __init__(self, x_val, y_val):
        self.x_val = x_val
        self.y_val = y_val