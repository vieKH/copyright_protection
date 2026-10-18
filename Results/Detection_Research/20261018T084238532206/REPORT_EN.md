# Watermark Detection Results on 200 Images

Detection of a specific reference watermark with known phases and spectral positions; the original image is not required.
Only images without attacks are evaluated. A negative result means that the watermark being tested was not detected; it does not prove that the image has never been watermarked.

| Experiment | Dataset | N | TP | TN | FP | FN | Accuracy | TPR | FPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1: no watermark | 512 | 100 | 0 | 99 | 1 | 0 | 99.00% | N/A | 1.00% |
| 1: no watermark | 1024 | 100 | 0 | 100 | 0 | 0 | 100.00% | N/A | 0.00% |
| 1: no watermark | all | 200 | 0 | 199 | 1 | 0 | 99.50% | N/A | 0.50% |
| 2: half watermarked | 512 | 100 | 50 | 49 | 1 | 0 | 99.00% | 100.00% | 2.00% |
| 2: half watermarked | 1024 | 100 | 50 | 50 | 0 | 0 | 100.00% | 100.00% | 0.00% |
| 2: half watermarked | all | 200 | 100 | 99 | 1 | 0 | 99.50% | 100.00% | 1.00% |
| 3: all watermarked | 512 | 100 | 100 | 0 | 0 | 0 | 100.00% | 100.00% | N/A |
| 3: all watermarked | 1024 | 100 | 100 | 0 | 0 | 0 | 100.00% | 100.00% | N/A |
| 3: all watermarked | all | 200 | 200 | 0 | 0 | 0 | 100.00% | 100.00% | N/A |

TP: watermarked and detected; TN: unwatermarked and not detected; FP: false positive; FN: false negative.
N/A: the corresponding class is absent from the experiment; do not replace this with 0%.

## Protocol Fixed Before Execution

- 100 images of size 512×512: block size 64, binary watermark size 21×21; 100 images of size 1024×1024: block size 128, watermark size 41×41.
- The watermark is an approximately balanced binary 0/1 matrix, not a QR code whose content is decoded.
- S=300; gap=1; payload seed=42; phase seed=20261011; master seed=20261018.
- Valid spectral positions are selected randomly within [4,12] and are known at both embedding and detection; the extraction start is random and differs from (0,0), with a stride of half the block size.
- Detection score: Pearson correlation between phase-projection scores and the reference watermark. A positive decision is made when score > threshold.
- Five folds are formed by source image and balanced across the two resolutions; images with identical pixel content remain in the same fold. Each image is evaluated using a threshold calibrated only on images outside its fold.
- Target alpha=0.01; the threshold is the order statistic at rank ceil((n+1)(1-alpha)) of the scores of UNWATERMARKED images. Scores of watermarked images are not used to select the threshold.
- Alpha is a target under the assumption that negative samples are exchangeable, not the probability that an individual decision is correct. Calibration pools both datasets and provides no separate guarantee for each resolution.
- Experiment 2 uses a reproducible random selection of exactly 50 watermarked images in each group; the assignments are recorded in decisions.csv.
- Scores are computed for 400 inputs (200 original images and 200 watermarked versions) and reused for the 600 decisions across the three scenarios. The three scenarios are not independent samples.

## Thresholds Calibrated Outside Each Test Fold

| Fold | Unique Calibration Images | Test Images | Threshold |
|---:|---:|---:|---:|
| 0 | 160 | 40 | 0.11563010 |
| 1 | 160 | 40 | 0.09836522 |
| 2 | 160 | 40 | 0.11563010 |
| 3 | 160 | 40 | 0.11563010 |
| 4 | 160 | 40 | 0.11563010 |

## Images with False Positive or False Negative Decisions

- False positive: `512 (64).tif`, experiment 1 and 2; score=0.11563010, threshold=0.09836522, fold=1.

PSNR of watermarked images: mean 49.118 dB; minimum 48.970 dB.

## Scope of the Conclusions

This is a cross-validation evaluation of the protocol, not an independent test of a final deployment threshold. Thresholds are not adjusted based on test results. Related images with nonidentical pixel content (the same scene, cropped or resized versions) may still introduce dependencies between folds.
Other watermarks, incorrect phase keys, geometric attacks, JPEG compression, noise, and searches over multiple keys or positions have not been evaluated. Each protocol change requires recalibration and reevaluation.
Do not infer that the true FPR is zero merely because no false positives are observed in a finite sample. These results must not be interpreted as proof of authorship.
