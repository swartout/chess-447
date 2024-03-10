import utils
from transformers import GPT2Config
from train import Trainer
import pickle


# TEST VALUES
DATA_PATH = "data/medium.pkl"

with open(DATA_PATH, "rb") as f:
    data, char_emb, board_emb = pickle.load(f)


tc = utils.TrainConfig("test", "predict", char_emb)
