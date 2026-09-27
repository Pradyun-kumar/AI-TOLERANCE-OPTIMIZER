# AI-Based Optimization of Dimensional Tolerances in Mechanical Assemblies

A PLM + AI project that predicts whether a machined part's tolerance
will PASS or FAIL an assembly fit requirement, and estimates its
manufacturing cost — using a trained machine learning model instead of
manual tolerance-stack-up calculations.

## What's in this folder

```
project/
├── data/
│   ├── tolerance_training_data.csv   500 rows of training data (auto-generated)
│   └── new_parts.csv                 Parts you want the AI to evaluate
├── model/                            Saved trained AI model (created after training)
├── outputs/                          Results appear here after you run a prediction
│   ├── predictions.csv
│   ├── report.txt
│   ├── tolerance_vs_status.png
│   └── predicted_cost_by_part.png
├── generate_dataset.py               Creates the training dataset
├── train_and_predict.py              Trains the model + predicts (terminal version)
├── app.py                            Point-and-click GUI (no terminal typing needed)
└── README.md                         This file
```

## What each file does (plain English)

- **generate_dataset.py** — Creates a realistic pretend "history" of 500
  previously manufactured shaft-bearing-assembly parts (shaft, bearing
  bore, housing, retaining ring, spacer), each with a tolerance, the
  process used to make it (turning/grinding/milling/drilling), material,
  fit type, and whether it ended up PASS or FAIL, plus its cost. This is
  what the AI learns from. You only need to re-run this if you want a
  fresh training set.

- **train_and_predict.py** — The main engine. It teaches the AI model
  using the 500-row dataset, checks how accurate it is, then reads your
  `new_parts.csv` and predicts PASS/FAIL + cost for each part. Saves
  everything into `outputs/`. This is the terminal version — one command,
  no GUI.

- **app.py** — A visual, button-based version of the same thing, so you
  don't have to touch the terminal. Has Upload / Predict / View Results /
  Export Report all as clickable buttons in your browser.

- **data/new_parts.csv** — This is the file you edit with your actual
  SolidWorks part dimensions/tolerances before predicting. A template
  with 8 example rows is already filled in — just replace the values
  with your real parts, keeping the same column headers.

## How to run it — Option A: GUI (recommended for your demo)

1. Open the VS Code terminal in this project folder.
2. Install the needed libraries (only once):
   ```
   pip install streamlit pandas numpy scikit-learn matplotlib joblib
   ```
3. Start the app:
   ```
   streamlit run app.py
   ```
4. Your browser opens automatically. In the sidebar:
   - Click **"(Re)train model on training dataset"** (first time only)
   - Upload your own CSV, or tick "Use existing data/new_parts.csv"
   - Click **Predict**
   - View the results table and charts on screen
   - Click **Download predictions.csv** / **Download report.txt** to export

## How to run it — Option B: Terminal only

1. Edit `data/new_parts.csv` with your part details (Excel or VS Code).
2. In the terminal, in this folder, run:
   ```
   python train_and_predict.py
   ```
3. Open the `outputs/` folder — you'll find `predictions.csv`, `report.txt`,
   and two chart images.

## Columns your parts CSV needs

| Column | Meaning | Example |
|---|---|---|
| Part_ID | Any unique ID you choose | NP001 |
| Part_Name | Shaft / Bearing_Bore / Housing / Retaining_Ring / Spacer | Shaft |
| Nominal_Dimension_mm | Design/nominal size | 25.0 |
| Tolerance_mm | Tolerance you're asking the machine shop to hold | 0.02 |
| Manufacturing_Process | Turning / Grinding / Milling / Drilling | Grinding |
| Material | Steel / Aluminum / Brass / Cast_Iron | Steel |
| Fit_Type | Clearance / Transition / Interference | Interference |
| Allowable_Tolerance_mm | Max tolerance allowed by the design for this fit | 0.03 |

## Workflow for your final-year demo

1. Design a part in SolidWorks.
2. Note its dimension, tolerance, process, material, and fit type.
3. Add a row for it in `data/new_parts.csv` (or type it into the app).
4. Click Predict (or run `train_and_predict.py`).
5. The AI tells you PASS/FAIL and the estimated cost — show the charts
   and `report.txt` as your project output.

## Why this is better than manual tolerance-stack-up

Traditional worst-case / RSS calculations are done by hand per part and
tend to be overly conservative, which drives up cost. This model was
trained on 500 examples that already encode the tradeoff between
tolerance tightness, process capability, and cost — so it can flag a
likely FAIL and estimate cost instantly, without you redoing the stack-up
math for every design iteration.
