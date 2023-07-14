import numpy as np
import matplotlib.pyplot as plt
from . import rcParams


def area(p1, p2, p3):
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)
    
    l1 = np.linalg.norm(p2 - p3)
    l2 = np.linalg.norm(p3 - p1)
    l3 = np.linalg.norm(p1 - p2)
    
    s = (l1 + l2 + l3) / 2
    a = np.sqrt(s * (s-l1) * (s-l2) * (s-l3))
    
    return a
    
    