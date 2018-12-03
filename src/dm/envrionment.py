import tools.file_loader as fl
import numpy as np


class DMSystem:
    def __init__(self, intention_path='../../../data/intention/default_intention.json'):
        self.actios = fl.load_as_list(intention_path)
        self.ternimal_state = []
        # self.states =

    def d_zero(self):
        return 0

    def dynamics(self):
        pass

    def P_and_R(self, s, a):
        pass


class DNUser:
    def __init__(self, slot_path='../../../data/slot/default_slot.json'):
        self.actios = fl.load_as_list(slot_path)
        self.ternimal_state = []
        # self.states =

    def d_zero(self):
        return 0

    def dynamics(self):
        pass

    def P_and_R(self, s, a):
        pass
