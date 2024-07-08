# <div align="center" style="font-weight: 400; background: darkslateblue; padding: 1rem; border-radius: 1rem">`changepoint-detector` 🔎 📉</div>

This tool is inspired by the paper
*Exploring Breaks in the Distribution of Stock Returns: Empirical Evidence from Apple Inc.*

Full credit goes to the original authors of the paper
for the ideas herein.
I owe a tremendous debt to these authors.

## 📊 Changepoints

The paper above defines a changepoint as follows.

The **changepoints** of a time series $\{Y_t\}_{1:T}$
form a partition $\rho_{1:T}$ of the set $\{1, \dots, T\}$
such that certain statistical properties are the same
within a sub-sequence and different between sub-sequences.

> This tool focuses on mean and variance of log returns,
> but any statistic can be examined.

The problem is essentially an $O(2^{T-1})$ search problem
in partition space.

## 📄 Overview of methods

### Evaluation criteria

The paper penalizes the training and evaluates performance
across model classes with Akaike Information Criterion (AIC),
Hannan–Quinn Information Criterion (HQIC), and Bayesian Information Criterion (BIC).
For now, this tool uses only BIC to compare results across models.

### Detection methods

#### Hidden Markov Models

Hidden Markov Models (HMMs) is implemented using `hmmlearn`.
Like in the paper, $N = 1, \dots, 8$ hidden states are considered.

#### Likelihood ratio

This tools makes use of the package `ruptures`,
based on [1]

#### Changepoint detection methods

TODO

#### Bayesian methods

TODO

## 📄 How to use

This tool can be used as a library to use methods individually,
or it can be used as a CLI to get a summary of each method.

```bash
changepoint-detector -f data/aapl.csv
```

The input data must be a csv of daily price data
in the following format:
```csv
Date,AAPL
1980-12-12,0.09905774146318436
1980-12-15,0.09388983249664307
```

This is the default format when exporting price data from `yfinance`.
```python
yf.download("SPY")["Adj Close"].to_csv("./spy_daily.csv")
```
Only a single security is supported at this time.

![sample output](./data/sample_output.png "Sample Output")


## 📄 Devlog

- 3 statistics: mean, var, and mean + var
- for some reason, `numpy==1.26.1` is required for `hmmlearn` to behave nicely and converge
- `python3.11` is a no-go; `pychangepoint` can't install for some reason.

## 📄 TODO

- usic BIC to find optimal penalty hyperparam for LRM
- usic BIC to score all results of final changepoints
- GLR test for CM
- convert HMM back to individual instances
- "tool is indeded to benchmark methods"
- find a solution for segments of length 1

[1] C. Truong, L. Oudre, N. Vayatis. Selective review of offline change point detection methods. Signal Processing, 167:107299, 2020. [journal] [pdf]