import numpy as np
import os


class SolutionFilesManager:
    def __init__(self, project_folder, case_name, case_type, solution_type):
        self.folder = "%s/solutions/%s"    %(project_folder.rstrip('/'), case_name)
        self.solution_path = "%s/%s_%s_%s_sol.npy"    %(self.folder, case_name, case_type, solution_type)

    def save_solution(self, cost, V, p_g, q_g):
        os.makedirs(self.folder, exist_ok=True)
        np.save(self.solution_path, np.array([cost, V, p_g, q_g], dtype=object), allow_pickle=True)

    def load_solution(self):
        return np.load(self.solution_path, allow_pickle=True)
