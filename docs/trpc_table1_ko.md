# TRPC 결과 — COXNet Table 1 형식

- Dataset: RGBTDronePerson validation 1,225 images
- Training: 12 epochs, deterministic seeds 0/1/2
- Selection: each seed checkpoint with the highest `bbox_mAP_50`; ties select the earlier epoch
- `±` is the sample standard deviation over three seeds

## Table 1 — best checkpoint per seed

| Method | Seed | Best epoch | mAP25 | mAP50 (all) | mAP50 (tiny) | tiny1 | tiny2 | tiny3 | small |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TRPC | 0 | 10 | 59.87 | 46.25 | 47.96 | 9.08 | 36.84 | 53.20 | 29.32 |
| TRPC | 1 | 12 | 58.49 | 45.45 | 47.23 | 8.98 | 35.88 | 52.32 | 27.26 |
| TRPC | 2 | 12 | 58.68 | 45.60 | 47.48 | 26.01 | 34.87 | 53.12 | 27.32 |
| **TRPC mean ± SD** | 0/1/2 | 10/12/12 | 59.01 ± 0.75 | 45.77 ± 0.43 | 47.56 ± 0.37 | 14.69 ± 9.80 | 35.86 ± 0.99 | 52.88 ± 0.49 | 27.97 ± 1.17 |

## Controlled comparison with COXNet

| Method | Seed | Best epoch | mAP25 | mAP50 (all) | mAP50 (tiny) | tiny1 | tiny2 | tiny3 | small |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **COXNet mean ± SD** | 0/1/2 | 10/9/12 | 58.88 ± 0.44 | 45.70 ± 0.49 | 47.28 ± 0.50 | 16.02 ± 7.71 | 35.70 ± 0.54 | 52.58 ± 0.47 | 29.70 ± 0.33 |
| **TRPC mean ± SD** | 0/1/2 | 10/12/12 | 59.01 ± 0.75 | 45.77 ± 0.43 | 47.56 ± 0.37 | 14.69 ± 9.80 | 35.86 ± 0.99 | 52.88 ± 0.49 | 27.97 ± 1.17 |
| **Paired Δ (TRPC−COXNet)** | 0/1/2 | — | +0.13 ± 1.04 | +0.07 ± 0.77 | +0.28 ± 0.78 | -1.33 ± 12.83 | +0.16 ± 0.45 | +0.30 ± 0.96 | -1.73 ± 1.01 |

## Final epoch reference

- TRPC epoch-12 mAP50 (all): **45.58 ± 0.12**

## Interpretation

- COXNet mAP50: **45.70 ± 0.49**; TRPC mAP50: **45.77 ± 0.43**.
- Paired mAP50 difference: **+0.07 ± 0.77**.
- The mAP50 difference is smaller than the seed variation. This table does not support a claim that TRPC improves overall AP.
- `tiny1` has high variance and should not be interpreted from its mean alone.
