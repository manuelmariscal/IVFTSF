# Hormone‑E2 Time‑Series Forecasting with a Custom Transformer

A minimal yet complete project that learns to predict **Estradiol (E2)** levels
along the menstrual cycle using a Transformer encoder built from first
principles (only PyTorch basics are used).

---

## 📁 Project Structure

```bash
dataset.py       # dataset & preprocessing
model.py         # Transformer architecture
train.py         # training / validation / test CLI
forecast.py      # interactive daily‑curve generation
requirements.txt # third‑party deps
README.md        # this guide
```

---

## ⚙️ Setup

```bash
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
pip install -r requirements.txt
```

---

## 🏃‍ Training

```bash
python train.py --csv path/to/data.csv
```

*Training/validation/test splits are **by patient**, so the
model is evaluated on completely unseen patients.*

The best checkpoint is saved to **`model.pt`**
and already contains the normalisation statistics needed for inference.

---

## 🔮 Forecasting a full curve

```bash
python forecast.py \
  --csv path/to/data.csv \
  --patient_id 1594127142 \
  --age 33
```

*If the patient appears in the CSV, the script overlays the observed E2
values (red dots) on top of the forecasted daily curve.*

---

## 📝 Notes

* The model **never** receives `e2` as an input feature;
  it is used **only** as the training target.
* Feature normalisation (mean/std) is learned **from the training split only**
  and stored inside `model.pt`.
* Padding is handled via a boolean mask so that loss & attention ignore
  artificial timesteps.
* You can tweak hyper‑parameters inside `train.py`
  (e.g. `CFG["d_model"]`, `CFG["epochs"]`, etc.).

\Real‑world accuracy depends on data quantity/quality and
hyper‑parameter tuning; feel free to adapt the code for your specific needs.

---

**Ready!**  
Run `train.py` to build the model, then `forecast.py` to generate complete
E2 curves for any patient (observed or new).
