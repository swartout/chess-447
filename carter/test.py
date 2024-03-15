import utils
from transformers import GPT2Config
from train import Trainer
import pickle


# TEST VALUES
DATA_PATH = "data/medium.pkl"
with open(DATA_PATH, "rb") as f:
    data, char_emb, board_emb = pickle.load(f)

tc = utils.TrainConfig("test", "prediction", char_emb)
tc.batch_size = 10
tc.folder_exist_ok = True

small_ds = utils.PGNDataset(data[:100], 64)

model_config = GPT2Config(
    vocab_size=len(char_emb),
    n_layer=4,
    n_head=4,
    n_embd=256,
    resid_pdrop=0,
    embd_pdrop=0,
    attn_pdrop=0,
)

model = utils.create_autoregressive_model(model_config)
utils.convert_to_predict(model, model_config)

trainer = Trainer(tc, model, small_ds, small_ds)
trainer.run()
