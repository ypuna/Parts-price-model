# Market Price Model for Unbenchmarked Auto Parts

Estimating what the market charges for aftermarket collision parts that no competitor lists, so they can be priced with the same rule as every other part.

## The problem

At an aftermarket auto-parts distributor, list prices are set from competitor list prices: each part is priced 1% below the lowest competitor, with a 30% gross-margin floor. That works for the **48,645 parts** that at least one of four competitors also lists.

It does not work for the other **8,707 parts (15% of the catalog)**, which no competitor lists. Those prices are set by hand or by a markup on cost, and they still drew about 800,000 quote requests in a year. This project asks:

> Can a part's catalog description and unit cost predict its market price well enough to price those 8,707 parts?

## Data

One row per part (57,352 parts, 24 categories, 27 makes):

| Field | Example |
|---|---|
| `description` | `TOYOTA CAMRY 18-23 - LT FRONT BUMPER COVER RETAINER (CAPA)` |
| `category` | Bumper, Lamp, Fender, Grille, Mirror, … |
| `unit_cost` | average purchase cost |
| `comp_a` … `comp_d` | each competitor's list price, if they list the part |

**Target:** the market price, defined as the median of the competitor list prices for that part.

**Confidentiality.** The real data comes from my employer and stays private. `data/parts.csv` is excluded from this repo. `data/sample_parts.csv` is a 1,500-row synthetic file with the same columns: real catalog descriptions, but simulated costs and prices. With it, anyone can run the code end to end; the results below come from the full private file. In that private file, internal part numbers are removed, competitor names are replaced with `comp_a`–`comp_d`, and every dollar value is multiplied by one undisclosed constant. All results are reported as percentages, which that scaling does not change.

## Approach

**1. Features from the description** (`src/features.py`). Every description follows the same pattern, so a regular expression splits it into:
- make and model (`TOYOTA CAMRY`);
- model years (2018–2023), how many years the part fits, and how old the model is;
- side (LT / RT);
- the part name (`front bumper cover retainer`) and its options (`w/o parking sensors`, `LED`, `CAPA`).

The part name and options become TF-IDF word features; make, model, category and side are one-hot encoded; unit cost goes in as log(cost).

**2. Three models, compared on the same split** (`src/train_evaluate.py`). All three predict log(market price).

| Model | What it is |
|---|---|
| Cost-plus (baseline) | cost × the category's median market-to-cost ratio — the usual fallback today |
| Ridge regression | linear model on the features above |
| LightGBM | gradient-boosted trees on the same features |

**3. Validation that doesn't flatter the model.** The left and right versions of a part, and the certified and non-certified versions, are near-copies with nearly the same price. A random split would put one version in training and its twin in the test set, which makes the scores look better than they really are. Instead I group those versions into **22,315 part families** and use **5-fold GroupKFold**, so a family is never split across train and test.

**4. A realistic target for "good".** Competitors don't agree with each other either. Scoring each competitor's price against the median of the others gives a **14% median difference**. No model of "the market price" should be expected to beat that by much.

## Results (5-fold grouped cross-validation, 48,645 parts)

| Model | Median abs. % error | Within ±10% | Within ±20% |
|---|---|---|---|
| Cost-plus by category | 25.2% | 21.0% | 40.6% |
| Ridge regression | 21.2% | 25.3% | 47.7% |
| **LightGBM** | **18.2%** | **29.5%** | **54.0%** |
| *Competitors vs. each other* | *14.0%* | *39.2%* | *62.4%* |

![Error distribution](reports/figures/error_cdf.png)

- LightGBM cuts the median error by about **28%** versus cost-plus (25.2% → 18.2%). That closes about 63% of the gap between cost-plus and the 14% competitor-agreement level.
- On parts with two or more competitors, where the market price is steadier, LightGBM's median error is 17.2%.
- Accuracy varies by category. Lamps, cooling, hoods and fenders are about 16–17%. Mirrors (30%) and door handles (39%) are much worse, because price depends on options (heated, power, signal, paint code) that the descriptions encode inconsistently.

![Error by category](reports/figures/error_by_category.png)

## Pricing the 8,707 unbenchmarked parts (`src/score_unbenchmarked.py`)

The final LightGBM is trained on all 48,645 benchmarked parts and predicts a market price for each unbenchmarked part. Each part then gets a proposed list price from the same rule: 1% below the estimate, never below a 30% gross margin. A part goes to **manual review** instead if its category had a median held-out error above 25% (Mirror, Handle) or too few parts to judge.

| Outcome | Parts |
|---|---|
| Priced automatically | 7,835 |
| Sent to manual review | 872 |
| Margin floor raised the price | 1,112 |

Comparing today's list price with the model's market estimate:

- **49%** are within ±20% of the estimate;
- **32%** are more than 20% above it, which suggests they may be overpriced and losing quotes;
- **19%** are more than 20% below it, which suggests they may be underpriced.

These are leads for a pricing analyst, not automatic changes. The model is still a best guess for any one part.

## Limitations

- **Cross-sectional data.** The model learns how prices relate to part attributes today. It says nothing about how demand would respond to a new price, which needs before-and-after sales data.
- **Missing competitor prices aren't random.** Unbenchmarked parts skew toward older, low-volume and unusual parts, so errors on them are likely somewhat higher than the cross-validation numbers.
- **Unit cost is an average** across purchases, so it carries noise from supplier changes.
- **Descriptions are inconsistent** for option-heavy parts (mirrors, handles). Better option parsing is the clearest next step.

## What I'd try next

1. Parse mirror and handle options into structured flags (heated, power, signal, paint-to-match).
2. Add prediction intervals, so review is decided per part instead of per category.
3. Re-run monthly, and track how accepted prices perform on quote-to-order conversion.

## Run it

```bash
pip install -r requirements.txt

# on the synthetic sample (public)
PARTS_CSV=data/sample_parts.csv python src/train_evaluate.py
PARTS_CSV=data/sample_parts.csv python src/score_unbenchmarked.py

# on the private data
COMPETITOR_COLUMNS="..." PRICE_SCALE="..." python src/prepare_data.py path/to/pricing_export.xlsx data/parts.csv
python src/train_evaluate.py          # about 3 minutes: CV, results.json, figures
python src/score_unbenchmarked.py     # proposed prices for unbenchmarked parts
```

```
src/
  prepare_data.py          anonymize the private export into data/parts.csv
  features.py              description parsing + feature matrix
  train_evaluate.py        grouped CV, three models, competitor-agreement check, figures
  score_unbenchmarked.py   final model + proposed prices + review flags
  make_sample.py           builds the synthetic sample file
reports/
  results.json             all CV metrics
  unbenchmarked_summary.json
  figures/
```
