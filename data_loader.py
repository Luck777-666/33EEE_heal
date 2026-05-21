import os
import pandas as pd
import json
import numpy as np

class DataLoader:
    """
    仿真数据加载器：读取 Simulink_AI_Data 目录下的所有文件
    """
    def __init__(self, data_root="data"):
        self.data_root = data_root
        self.static_dir = os.path.join(data_root, "static")
        self.cases_dir = os.path.join(data_root, "cases")

    def load_static_csv(self, filename):
        filepath = os.path.join(self.static_dir, filename)
        if os.path.exists(filepath):
            return pd.read_csv(filepath)
        return None

    def load_case_csv(self, case_id, filename):
        filepath = os.path.join(self.cases_dir, case_id, filename)
        if os.path.exists(filepath):
            return pd.read_csv(filepath)
        return None

    def load_case_json(self, case_id, filename):
        filepath = os.path.join(self.cases_dir, case_id, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)
        return None

    def get_case_list(self):
        cases = [d for d in os.listdir(self.cases_dir) if os.path.isdir(os.path.join(self.cases_dir, d))]
        return cases

    def get_noisy_signals(self, case_id):
        return self.load_case_csv(case_id, "signals_noisy.csv")

    def get_clean_signals(self, case_id):
        return self.load_case_csv(case_id, "signals_clean.csv")