"""Double-click launcher for Windows installations with .pyw associated with Python."""
from multiprocessing import freeze_support

if __name__ == '__main__':
    freeze_support()
    from hardware_setup import open_setup
    from simulator import Actuators, Guidance, Mission, Plant, Wind
    open_setup((Plant(), Actuators(), Guidance(), Wind(), Mission(planner_budget=8)), 1., False)
