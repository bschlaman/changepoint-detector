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

- Akaike Information Criterion (AIC): $2 k - 2 \ln L$
- Hannan–Quinn Information Criterion (HQIC): $k \ln n - 2 \ln L$
- Bayesian Information Criterion (BIC): $2 k \ln \ln n - 2 \ln L$

### Detection methods

#### Hidden Markov Models

Hidden Markov Models (HMMs) is implemented using `hmmlearn`.
Like in the paper, $N = 1, \dots, 8$ hidden states are considered.


## 📄 How to use

This tool can be used as a library,
but it also comes with a CLI.

```bash
changepoint-detector -f sp500_daily_price.csv
```

## 📄 Devlog

- 3 statistics: mean, var, and mean + var
- for some reason, `numpy==1.26.1` is required for `hmmlearn` to behave nicely and converge