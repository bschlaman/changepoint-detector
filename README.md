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


## 📄 How to use

This tool can be used as a library,
but it also comes with a CLI.

```bash
changepoint-detector -f sp500_daily_price.csv
```