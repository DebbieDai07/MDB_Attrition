# MDB_Attrition
MDB attrition capstone of Columbia DSI students  

### Data
Data from MBD-mini dataset released by sb-ai-lab on Hugging Face, licensed under CC-BY-4.0
https://huggingface.co/datasets/ai-lab/MBD-mini/tree/main

### Processed data and training

Processed clean train/test CSVs for this project are hosted as a Hugging Face dataset: `Saravanan1999/MDBAttrition`.

To use them in the training scripts, install the Hugging Face Hub client:

```bash
pip install huggingface_hub
```

Both `scripts/train_clean_model.py` and `scripts/train_tcn_clean.py` will:
- First look for `data/train_clean.csv` and `data/test_clean.csv` locally.
- If they are not found, automatically download `train_clean.csv` and `test_clean.csv` from the `Saravanan1999/MDBAttrition` dataset on Hugging Face.

You can also download the files explicitly into the `data/` directory with:

```bash
hf download Saravanan1999/MDBAttrition train_clean.csv test_clean.csv --repo-type dataset -D data
```