import torch
import matplotlib.pyplot as plt
import os
import yaml
import numpy as np


class Visualizer():
    def __init__(self, log_path, name="modality_scores") -> None:
        self.log_path = log_path
        self.name = name
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

    def save_mod_score_histogram(self, mod_scores_dict: dict[str, float]):
        fig, ax = plt.subplots(figsize=(20,10),layout='constrained')
        ax.bar(mod_scores_dict.keys(), mod_scores_dict.values(), 
                color='royalblue', alpha=0.7)
        ax.grid(color='#95a5a6', linestyle='--', linewidth=2, 
                    axis='y', alpha=0.6)
        ax.set_xlabel('modalitiy', fontsize=20)
        ax.set_ylabel('score', fontsize=20)
        num_categories = len(mod_scores_dict.values())
        ax.set_xticks(ticks=np.arange(num_categories),
                    label=mod_scores_dict.keys())
        ax.tick_params(axis='x',labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
        # ax.set_frame_on(True)
        img_path = os.path.join(self.log_path, f"{self.name}.png")
        plt.savefig(img_path, dpi=100, pad_inches=4)
        plt.close('all')
        
    def save_mod_scores(self, mod_scores_dict: dict[str, float]):
        yaml_path = os.path.join(self.log_path, f"{self.name}.yaml")
        with open(yaml_path, 'w') as f:
            yaml.dump(mod_scores_dict, f, default_flow_style=False)
    
    @staticmethod
    def get_mod_scores(model: torch.nn.Module, p:float):
        #TODO: One of the use case where a Base class is for multimodal module is necessary
        model.eval()
        modality_scores = model.modality_scores(p)
        modality_scores = {key:float(f'{value:.2f}') for (key,value) 
                            in modality_scores.items()}
        return modality_scores
