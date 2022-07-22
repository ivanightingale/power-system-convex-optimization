import numpy as np
import os


class SolutionFilesManager:
    def __init__(self, project_folder, case_name, solution_type, custom_case_type=None):
        self.folder = "%s/solutions/%s" % (project_folder.rstrip('/'), case_name)
        if custom_case_type is not None:
            self.solution_path = "%s/%s_%s_%s_sol.npy" % (self.folder, case_name, custom_case_type, solution_type)
        else:
            self.solution_path = "%s/%s_%s_sol.npy" % (self.folder, case_name, solution_type)

    def save_solution(self, cost, X, p_g=None, q_g=None):
        # note that X can be either a matrix (VV^*) or a vector (V itself)
        os.makedirs(self.folder, exist_ok=True)
        np.save(self.solution_path, np.array([cost, X, p_g, q_g], dtype=object), allow_pickle=True)

    def load_solution(self):
        return np.load(self.solution_path, allow_pickle=True)
