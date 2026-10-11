# Results from the manuscript

All numbers are taken from *LARA: Labor-market-aware relational attention for technology-to-skill recommendation* (under review). They were produced with the research code on the Korean data described in [reproducing_the_paper.md](reproducing_the_paper.md); this package cannot reproduce them without that data.

## Data and evaluation units

| Item | Value |
| --- | --- |
| Labor-market data | Korean job postings, 2021-2024; 7,934,099 postings |
| Patents | Granted Korean patents filed 2019-2023; 40 per year, 200 in total |
| Candidate skills | Entire skill dictionary, 6,558 skills |
| Graph nodes | 6,558 skills, 450 occupations, 29 / 118 / 464 taxonomy groups |
| Graph relations | occupation-skill, skill co-occurrence, taxonomy membership, aggregated group co-occurrence |
| Train / validation / test | 120 / 40 / 40 patents |
| Known relevant pairs | 1,040 (train), 352 (validation), 367 (test); one explicit negative per split |
| Relevant skill | merged human grade 1 or 2 |
| Evaluation unit | full-dictionary recommendation list per patent |

Learned models were trained with three seeds (42, 43, 44); checkpoints were selected on validation known-positive Recall@5 and evaluated once on the sealed test set. Bold marks the best value per column.

## Full-dictionary retrieval on the test set (%)

Recall@5 is mean ± sample SD across seeds; other columns are means. Skill frequency, which retrieves no known positive at any cutoff, is omitted.

| Model | Recall@5 | Hit@5 | Recall@20 | Recall@100 |
| --- | ---: | ---: | ---: | ---: |
| BGE similarity | 9.48 | 57.50 | 15.59 | 33.14 |
| Relation diffusion | 9.48 | 57.50 | 15.59 | 33.14 |
| Text MLP | 15.94 ± 0.38 | 80.83 | 38.99 | 66.31 |
| GraphSAGE (skill graph) | 16.64 ± 0.94 | 84.17 | 37.26 | 65.88 |
| GraphSAGE (full graph) | 13.29 ± 1.37 | 67.50 | 34.90 | 62.78 |
| R-GCN | 14.70 ± 2.13 | 76.67 | 30.54 | 57.26 |
| PinSage | 16.99 ± 0.82 | 78.33 | 38.51 | 63.63 |
| HGT | 16.94 ± 0.71 | 80.83 | 39.50 | 68.12 |
| Taxonomy GNN | 13.64 ± 1.48 | 71.67 | 35.77 | 64.43 |
| LARA-NA | 18.60 ± 1.10 | 84.17 | 41.11 | 68.26 |
| **LARA** | **20.86 ± 0.90** | **87.50** | **44.26** | **72.82** |

Relation diffusion equals BGE similarity because validation selected a diffusion strength of zero. Within-seed increments of LARA over Text MLP, HGT and LARA-NA: 4.92 ± 1.26, 3.93 ± 0.19 and 2.26 ± 0.45 percentage points.

![RQ1: Recall@5 and Recall@K on the test set](figures/results_rq1.png)

## Recall by semantic group (test Recall@5, %, mean ± SD)

Low, medium and high bins are defined by the BGE cosine of training positives (127, 100 and 140 test pairs); the last column is the overlapping subset of 334 positives outside each patent's BGE top five.

| Model | Low | Medium | High | Outside BGE top-5 |
| --- | ---: | ---: | ---: | ---: |
| BGE similarity | 0.00 | 0.95 | **37.91** | 0.00 |
| Relation diffusion | 0.00 | 0.95 | **37.91** | 0.00 |
| Text MLP | 10.91 ± 0.34 | 19.92 ± 0.89 | 25.21 ± 1.54 | 11.74 ± 0.34 |
| GraphSAGE (skill graph) | 10.03 ± 1.72 | 18.60 ± 2.34 | 30.66 ± 3.89 | 12.80 ± 1.55 |
| GraphSAGE (full graph) | 6.31 ± 2.73 | 17.53 ± 0.73 | 24.01 ± 4.10 | 9.31 ± 1.07 |
| R-GCN | 5.96 ± 2.06 | 18.22 ± 2.73 | 35.28 ± 3.77 | 8.63 ± 1.55 |
| PinSage | **11.80 ± 0.68** | 18.35 ± 1.73 | 30.77 ± 1.59 | 11.58 ± 1.48 |
| HGT | 8.35 ± 2.03 | 19.33 ± 2.00 | 30.23 ± 3.75 | 11.53 ± 1.05 |
| Taxonomy GNN | 7.34 ± 1.52 | 19.65 ± 1.44 | 24.40 ± 3.05 | 9.47 ± 1.38 |
| LARA-NA | 10.79 ± 1.44 | 19.33 ± 2.36 | 34.04 ± 1.13 | 13.57 ± 1.12 |
| LARA | 10.81 ± 0.36 | **22.23 ± 1.17** | 34.37 ± 1.50 | **15.94 ± 0.83** |

Relative to Text MLP, LARA retrieves 24.00 additional positive pairs outside the BGE top five while losing 8.67 (net +15.33); relative to HGT the counts are 21.33 and 6.67, relative to LARA-NA 16.33 and 7.33. BGE's higher macro recall in the high bin does not contradict LARA recovering more high-bin pairs in absolute terms (41 versus 32 on average): the macro metric divides by each patent's number of positives in the bin first.

![RQ2: recall by semantic group and pairwise gains](figures/results_rq2.png)

## Structural ablations

Increments are LARA minus the control in known-positive Recall@5, in percentage points, computed within matching seeds before averaging; positive values favor LARA. Controls: removal of occupation relations, removal of skill co-occurrence, removal of all posting relations, degree-preserving rewiring of the market graph, and fixed (not patent-conditioned) occupation routing.

| Control | Test Recall@5 | Increment | Low | Medium | High | Outside BGE top-5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LARA | 20.86 ± 0.90 | – | | | | |
| without occupation relations | 19.95 ± 1.11 | +0.91 ± 1.45 | −1.99 | +0.98 | +4.23 | +0.94 |
| without skill co-occurrence | 21.10 ± 0.83 | −0.24 ± 1.09 | −0.50 | −0.18 | −1.94 | +0.55 |
| without all posting relations | 19.46 ± 1.01 | +1.41 ± 1.46 | −0.76 | +0.06 | +6.49 | +1.25 |
| rewired market graph | 21.65 ± 0.71 | −0.79 ± 0.20 | −1.13 | −0.27 | −2.90 | −0.37 |
| fixed occupation routing | 20.43 ± 0.59 | +0.43 ± 1.44 | −0.11 | −1.60 | +1.32 | +0.40 |

The removal and fixed-routing controls lower recall, which quantifies the contribution of those components. Removing co-occurrence and rewiring do not, so the experiments support the complete architecture while not establishing that the observed edge arrangement itself is necessary. The proposed model and protocol were not changed after these test results were inspected.

![Ablations: increments overall and by group](figures/results_ablation.png)

## Sensitivity to training-set size (validation Recall@5, %)

Separate fits on nested training subsets; at 30-90 patents the SD combines three subset draws and three seeds, at 120 patents three seeds. The last two columns are matched increments.

| Training patents | Runs per model | Text MLP | LARA-NA | LARA | LARA − Text MLP | LARA − LARA-NA |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 30 | 9 | 17.64 ± 0.64 | 19.75 ± 0.58 | **20.95 ± 0.92** | +3.32 ± 0.98 | +1.21 ± 1.18 |
| 60 | 9 | 18.97 ± 0.58 | 20.97 ± 0.54 | **21.98 ± 1.21** | +3.01 ± 1.11 | +1.01 ± 1.38 |
| 90 | 9 | 19.34 ± 0.55 | 21.42 ± 0.88 | **22.53 ± 0.79** | +3.19 ± 0.94 | +1.11 ± 0.94 |
| 120 | 3 | 19.17 ± 0.84 | 22.00 ± 0.88 | **23.50 ± 1.10** | +4.33 ± 0.67 | +1.50 ± 1.86 |

## What the method does and does not claim

The method retrieves candidate skills for a new description in an observed labor-market context. It does not forecast which skills will be adopted, and the Korean experiment does not show that trained weights transfer unchanged to another national labor market. Reported recall covers known positives only; unjudged recommendations are neither correct nor incorrect under these data.
