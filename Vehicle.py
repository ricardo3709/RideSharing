"""
Vehicle class
"""


class Vehicle:
    request = None
    charging = False
    stateOfCharge = 50
    path = []
    estimated_done_riding = -1
    estimated_done_charging = -1
    last_update = 0
    capacity = 4
    sharing = 0
    removed = True
    def __init__(self, position):
        self.position = position
