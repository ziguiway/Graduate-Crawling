
This is the official implementation of **LLM-MFEFND**
##dataset
Original dataset from weibo21 and FineFake


## Requirements

- Python 3.6
- PyTorch > 1.0
- Pandas
- Numpy
- Tqdm


## Run

Parameter Configuration:

- dataset: the English or Chinese dataset, default for `ch`
- early_stop: default for `5`
- epoch: training epoches, default for `50`
- gpu: the index of gpu you will use, default for `0`
- lr: learning_rate, default for `0.0001`(en:0.0002)
- You can set the list of learning rates in grid_search.py's train_param
- 

You can run this code through:

```powershell
python main.py
```

The public repository contains only a 200-row FineFake auxiliary subset, so
the reconstructed entrypoint is intended for data-flow and development
experiments until the authors provide the complete data. From the parent
reproduction directory, the current checks are:

```bash
uv run python scripts/smoke_mlime.py
uv run python scripts/compare_hpt.py
uv run python scripts/train_finefake_aux.py --epochs 1 --max-steps 1
```

`mlime.py` is a post-hoc explainer. It perturbs one modality while the caller
keeps the other modalities fixed and supplies the complete model prediction;
it does not alter classifier training.

## Reference



```
