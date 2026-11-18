# Synthetic classification stream

`stream_sample.csv` contains 2,000 ordered rows from NumPy's `default_rng(2026)`. It is synthetic and illustrates buffered training, missing values, new categories, and a change in the label rule halfway through.

| Column | Generation rule |
| --- | --- |
| `signal`, `context` | Independent standard-normal features. After generating the labels, the generator independently masks them with probabilities 7% and 5%. Empty CSV fields represent missing numeric readings. Observed values are stored to six decimal places. |
| `category_code` | Starts as a uniform draw from 0 and 1. From row 601, each code in the remaining stream has a 25% chance of becoming 2. From row 1401, a separate draw gives each code in the remaining stream a 25% chance of becoming 3. |
| `target` | 1 if the latent score is nonnegative, otherwise 0. The numeric score is `1.10*signal - 0.65*context` for rows 1–1,000 and `0.65*signal + 0.85*context` thereafter. Category offsets are `[-0.35, 0.30, 0.70, -0.60]`, and independent normal noise with standard deviation 0.35 is added before thresholding. |

The notebook and buffer benchmark preserve row order and read the CSV in chunks of 32, with a final chunk of 16. Run the generator from the repository root to reconstruct the file in the existing demo directory.

```sh
conda run -n aiml python -B -m demo.generate_stream_data
```

`sample.csv` remains the small Assignment 2.1 sample.
