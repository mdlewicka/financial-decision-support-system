# Final Evaluation Summary

## Financial Decision Support System

This document records the results of the one-time final evaluation on the previously closed market test split.

The final evaluation was executed only after model, portfolio and reinforcement-learning choices had been frozen.

No model selection, hyperparameter tuning, portfolio re-optimization or reinforcement-learning retraining was performed using the final test results.

## Evaluation protocol

The project follows a chronological three-way split:

- `training` — model fitting and portfolio construction;
- `validation` — development, comparison and model-selection decisions;
- `test` — one final evaluation after development decisions were frozen.

Before opening the test split:

- the complete project passed Ruff;
- the non-RL regression suite passed with 499 tests;
- the RL and RL integration suite passed with 59 tests;
- the final-evaluation test suite passed;
- the final-evaluation preflight reported `READY`;
- frozen analytical artifacts were recorded with SHA256 fingerprints;
- a persistent one-time-use guard was prepared.

The frozen manifest was created before final-test access:

```text
benchmark_results/final_evaluation_frozen_manifest.json
```

with:

```text
status: frozen
test_split_used: false
frozen_at_utc: 2026-08-12T14:00:27.611264+00:00
```

The final evaluation was then executed once with the explicit confirmation:

```text
OPEN_FINAL_TEST_ONCE
```

The run completed successfully and created:

```text
benchmark_results/final_test/final_test_usage.json
benchmark_results/final_test/final_evaluation_result.json
```

No `final_evaluation_failure.json` was produced.

## Forecasting results

All frozen forecasting approaches were evaluated on the final test split.

| Model | Observations | MAE | RMSE | R² | Directional accuracy |
|---|---:|---:|---:|---:|---:|
| Random Forest | 4786 | 0.009504 | 0.014632 | 0.007772 | 54.53% |
| ARIMA walk-forward | 4786 | 0.009518 | 0.014706 | -0.002362 | 53.36% |
| Naive zero-return baseline | 4786 | 0.009534 | 0.014707 | -0.002547 | 0.42% |

Random Forest achieved the strongest overall final-test result among the evaluated forecasting models.

Its advantage remains limited. In particular, the test R² of approximately `0.0078` indicates that the model explains only a very small fraction of the variation in next-session returns.

The final result therefore supports the same conservative interpretation established during validation: forecasting short-horizon financial returns remains difficult, and model predictions should be treated as supporting signals rather than stand-alone decision rules.

## Classical portfolio results

The five previously defined portfolio strategies were evaluated using the weights frozen before final-test access.

The optimized portfolios were not re-optimized after the test split was opened.

| Strategy | Annualized return | Annualized volatility | Sharpe | Sortino | VaR | CVaR | Maximum drawdown |
|---|---:|---:|---:|---:|---:|---:|---:|
| Defensive | 16.63% | 10.56% | 1.388 | 2.044 | 1.04% | 1.54% | 8.01% |
| Equal Weight | 19.78% | 15.34% | 1.160 | 1.744 | 1.39% | 2.10% | 11.84% |
| Growth | 19.67% | 15.87% | 1.115 | 1.676 | 1.42% | 2.18% | 14.51% |
| Maximum Sharpe | 15.98% | 14.27% | 0.981 | 1.464 | 1.41% | 1.93% | 11.38% |
| Minimum Volatility | 17.13% | 10.65% | 1.422 | 2.037 | 1.06% | 1.62% | 9.11% |

The main final-test observations are:

- `equal_weight` achieved the highest annualized return among the classical portfolios;
- `minimum_volatility` achieved the highest Sharpe ratio;
- `defensive` achieved the lowest realized volatility;
- `defensive` also achieved the smallest maximum drawdown;
- `maximum_sharpe`, although optimized for Sharpe on the training period, did not achieve the highest Sharpe ratio on the final test period.

These results illustrate the limitations of historical portfolio optimization and the sensitivity of portfolio rankings to changing market conditions.

## Reinforcement Learning result

The previously trained PPO allocation policy was loaded from the frozen model artifact:

```text
benchmark_results/rl/ppo/ppo_allocation_agent.zip
```

The model was not retrained after final-test access.

Final-test PPO metrics were:

| Metric | Result |
|---|---:|
| Observations | 565 |
| Annualized mean return | 28.93% |
| Annualized volatility | 22.25% |
| Sharpe ratio | 1.211 |
| Sortino ratio | 2.649 |
| Value at Risk | 1.38% |
| Conditional Value at Risk | 2.07% |
| Maximum drawdown | 11.63% |
| Cumulative return | 81.67% |
| Final portfolio value | 1.8167 |
| Total transaction cost | 0.005061 |
| Mean turnover | 0.008958 |
| Total reward | -2.569404 |

PPO achieved the highest annualized return among the evaluated portfolio-allocation approaches and also achieved the highest Sortino ratio.

However, PPO did not achieve the highest Sharpe ratio and did not minimize volatility or drawdown.

The result therefore does not establish universal dominance of Reinforcement Learning over classical portfolio methods.

Instead, it shows that the frozen PPO policy performed strongly during the final test period while accepting a higher level of volatility.

The negative `total_reward` does not imply a negative investment return. The RL reward is an environment-specific optimization signal that includes penalties and therefore is not equivalent to portfolio return.

## Final interpretation

The final test confirms several conclusions established during development.

Forecasting performance remains weak in absolute terms. Random Forest provides the strongest result among the tested forecasting approaches, but its explanatory power remains very limited.

Classical portfolio strategies continue to demonstrate meaningful differences in risk-return characteristics. Simple and interpretable strategies remain competitive, and optimization objectives selected during training do not necessarily dominate on unseen data.

The PPO policy produced a strong final-period result, particularly in return and downside-risk-adjusted performance. This supports its role as a promising experimental extension of the system.

It does not justify replacing classical allocation methods solely because PPO performed well on this single final test period.

## Governance conclusion

The final test split has now been used and must be treated as consumed.

The final-test results must not be used for:

- additional model selection;
- hyperparameter tuning;
- ARIMA order selection;
- portfolio re-optimization;
- PPO retraining;
- reward-function redesign;
- strategy selection followed by another claim of unbiased test performance.

Any future methodological changes would require a new evaluation protocol and genuinely unseen data.

The recorded final-test artifacts represent the terminal evaluation of the currently frozen project configuration.
