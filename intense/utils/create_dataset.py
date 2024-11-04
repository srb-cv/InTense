from typing import Union
from omegaconf import DictConfig, ListConfig, OmegaConf
import random
from numpy.random import default_rng
import pandas as pd
import os


class MultiModalStringData:
    def __init__(self, cfg: Union[DictConfig, ListConfig]):
        self.pos_sub_seq: str = cfg.data.pos_sub_seq
        self.neg_sub_seq: str = cfg.data.neg_sub_seq
        self.num_modalities: int = cfg.data.num_modalities
        self.seq_chars: str = cfg.data.seq_chars
        self.insert_p: list = cfg.data.insert_p
        self.seq_len: int = cfg.data.seq_len
        self.num_samples = cfg.data.num_samples
        self.num_classes = cfg.data.num_classes
        self.rng = default_rng()

    def create_dataframe(self):
        assert self.num_classes == 2, "only two classes are supported"
        num_class_samples = self.num_samples // self.num_classes
        rows = []
        for _ in range(num_class_samples):
            pos_row = self.generate_multimodal_sample("positive")
            neg_row = self.generate_multimodal_sample("negative")
            rows.append(pos_row)
            rows.append(neg_row)
        return pd.DataFrame(rows)

    def generate_multimodal_sample(self, label: str):
        """Generate a multimodal sample with one label and
        given data modalities. Use it to create the complete dataset
        Args:
        label: class label. 'positive' or 'negative'
        """
        sample_dict = {}
        assert self.num_modalities == len(
            self.insert_p
        ), "num of modalities does not match the num of probabilities provided"
        for i, prob in enumerate(self.insert_p):
            modality_sample = self.generate_sample(prob, label)
            key: str = f"mod_{i+1}_{prob:.2f}"
            sample_dict[key] = modality_sample
        if label == "positive":
            label_int = 1
        elif label == "negative":
            label_int = -1
        else:
            raise ValueError("label argument should only be 'positive' or 'negative'")
        sample_dict["label"] = label_int
        return sample_dict

    def generate_sample(self, prob: float, label: str):
        """Returns a sequence of given length using a sequence of chars.
        A given subsequence of length 7 is inserted at a random postion
        with probabilty p
        Args:
        prob: the probability that whether a sample contains a sub_sequence
        label: class label. 'positie' or 'negative'
        """
        if label == "positive":
            sub_seq = self.pos_sub_seq
        elif label == "negative":
            sub_seq = self.neg_sub_seq
        else:
            raise ValueError("label argument should only be 'positive' or 'negative'")

        seq = "".join([random.choice(self.seq_chars) for i in range(self.seq_len)])
        if prob == 0:
            return seq

        random_insert = self.rng.binomial(1, prob)
        if random_insert == 1:
            seq = self.insert_subsequence(seq, sub_seq)
        return seq

    @staticmethod
    def insert_subsequence(seq, sub_seq):
        """Returns a sequence by inserting the given
        subsequence at a random postion
        """
        assert len(seq) > len(
            sub_seq
        ), "the length of the subsequence\
             is greater than the sequence"
        random_position = random.randint(0, len(seq) - len(sub_seq))
        replaced_seq = (
            seq[:random_position] + sub_seq + seq[random_position + len(sub_seq) :]
        )
        assert len(replaced_seq) == len(seq)
        return replaced_seq


class MultiModalXorData:
    def __init__(self, cfg: Union[DictConfig, ListConfig]):
        self.pos_sub_seq: str = cfg.data.pos_sub_seq
        self.neg_sub_seq: str = cfg.data.neg_sub_seq
        self.num_modalities: int = cfg.data.num_modalities
        self.seq_chars: str = cfg.data.seq_chars
        self.insert_p: list = cfg.data.insert_p
        self.seq_len: int = cfg.data.seq_len
        self.num_samples: int = cfg.data.num_samples
        self.num_classes: int = cfg.data.num_classes
        self.rng = default_rng()

    def create_dataframe(self):
        assert self.num_classes == 2, "only two classes are supported"
        rows = []
        for _ in range(self.num_samples):
            row = self.generate_multimodal_sample()
            rows.append(row)
        return pd.DataFrame(rows)

    def generate_multimodal_sample(self):
        """generates a multimodal sample and assings a label
        according to the XOR logic. If both modalities contains
        same subsequence then label is 0 otherwise 1"""
        sample_dict = {}
        seq_tokens = []
        assert (
            self.num_modalities > 2
        ), "Atleast three modalities should be present for the XOR dataset"
        for i, prob in enumerate(self.insert_p):
            seq_token, modality_sample = self.generate_sample(prob)
            key = f"mod_{i+1}"
            sample_dict[key] = modality_sample
            seq_tokens.append(seq_token)
        if self.num_modalities == 3:
            sample_dict["label"] = 0 if seq_tokens[0] == seq_tokens[1] else 1
        else:
            count_pos = seq_tokens[:3].count(1)
            sample_dict["label"] = 0 if count_pos % 2 == 0 else 1
        return sample_dict

    def generate_sample(self, prob: float):
        seq = "".join([random.choice(self.seq_chars) for _ in range(self.seq_len)])
        if prob == 0:
            return None, seq
        seq_token = self.rng.binomial(1, prob)
        sub_seq = self.pos_sub_seq if seq_token == 1 else self.neg_sub_seq
        seq = MultiModalStringData.insert_subsequence(seq, sub_seq)
        return seq_token, seq


def create_multimodal_string_data(config_path,
                                  data_dir="synthetic_datasets"):
    cfg = OmegaConf.load(config_path)
    data_obj = MultiModalStringData(cfg)
    df = data_obj.create_dataframe()
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    df.to_csv(f"{data_dir}/data.csv")


def create_multimodal_xor_data(config_path):
    if config_path is not None:
        cfg = OmegaConf.load(config_path)
    else:
        print("No config file is passed, using a default one..")
        cfg = OmegaConf.load("tmf/config/xor_data_config_v3.yaml")
    data_obj = MultiModalXorData(cfg)
    df = data_obj.create_dataframe()
    df.to_csv("tests/xor_data.csv")


if __name__ == "__main__":
    config_path = "data/synthetic_data/config/data_config_v7.yaml"
    create_multimodal_string_data(config_path,
                                  data_dir="data/synthetic_data")
