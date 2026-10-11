# Models

## Variants

| CLI name | Internal identifier | Class | Description |
| --- | --- | --- | --- |
| `lara` (default) | `hgt_conditional` | `conditional_routing_ranker.ConditionalRoutingRanker` | Full model: content-preserving query adapter, description-conditioned attention over skill, occupation and taxonomy nodes, three relation routes (co-occurrence, occupation, taxonomy) with description-specific occupation routing, a learned gate over the routes, and a bounded residual on top of the frozen text similarity |
| `lara-na` | `hgt_network` | `network_attention_ranker.NetworkAttentionRanker` | Description-conditioned node attention with fixed (market-weighted) routing to skills and equal mass per node type; same encoder, decoder and bounded residual |

The internal identifiers are kept because configurations and checkpoints record them. The paper calls the two variants LARA and LARA-NA.

## Components and where they live

| Paper section | Component | Module |
| --- | --- | --- |
| Labor-market graph and node features | frozen text features for skills, occupations, taxonomy groups; typed graph attention (HGT-style, static, with a log edge-weight prior) | `encoders`, `attention_ranker.TypedAttentionLayer`, `attention_ranker.AttentionRanker` |
| Patent-conditioned node attention | per-type softmax attention of the query over actual nodes | `network_attention_ranker.NetworkAttentionRanker.node_attention`, `conditional_routing_ranker.ConditionalRoutingRanker.content_attention` |
| Relation routes | co-occurrence route (direct plus propagated), occupation route with query-conditioned edge probabilities, taxonomy route, gate over the three families, uniform-attention reference mass | `ConditionalRoutingRanker.encode_queries` and helpers |
| Skill score | decoder over query and skill states plus log-ratio of routed mass to its reference, bounded by `residual_bound * tanh(.)` around the text cosine | `ConditionalRoutingRanker.score_pairs`, `ranking.SkillRanker.combine_scores` |
| Training objective for partial labels | soft targets from the frozen cosine distribution, positives mixed in with `positive_target_mass`, explicit zeros removed, KL retrieval loss plus auxiliary pairwise loss and residual penalty | `retrieval_objective` |
| Inference | all-candidate scoring in chunks; descriptions encoded in memory | `recommender.RecommendationEngine` |

Non-learned references are computed during training for the validation split: `popularity` (log posting count), `semantic` (text cosine), `diffusion` (cosine smoothed over the co-occurrence graph with a validation-selected alpha). They are reported in `validation_comparison.csv` next to the trained model and can be used as engines (`model="semantic"`), but they are not the proposed method.

## Hyperparameters (`model` section of `lara_app.json`)

| Key | Default | Meaning |
| --- | --- | --- |
| `hidden_dim`, `decoder_dim`, `layers`, `attention_heads` | 128, 64, 2, 4 | widths and depth of the graph attention and decoder |
| `edge_weight_prior` | 1.0 | weight of the log edge-weight prior in attention logits |
| `dropout` | 0.1 | dropout in encoder and decoder |
| `query_neighbors`, `query_temperature` | 20, 0.1 | semantic attachment of a query to its nearest skills (text only) |
| `node_attention_temperature`, `node_attention_mode` | 0.2, `learned` | node attention of `lara-na`; `uniform` gives a reference without learned attention |
| `content_query_bound` | 0.1 | maximum change of the query vector by the content adapter |
| `semantic_attention_temperature`, `learned_attention_bound` | 0.05, 0.1 | `lara` attention: cosine term temperature and bound on the learned term |
| `occupation_route_temperature`, `routing_rank`, `routing_query_chunk` | 0.1, 16, 8 | description-conditioned occupation routing |
| `residual_bound` | 0.1 | bound of the correction around the text cosine |
| `retrieval_temperature`, `positive_target_mass` | 0.03, 0.5 | soft target construction |
| `auxiliary_rank_weight`, `rank_temperature` | 0.05, 0.03 | auxiliary pairwise loss on judged pairs |
| `residual_regularization` | 0.01 | penalty on deviations from the text cosine |
| `epochs`, `learning_rate`, `weight_decay`, `gradient_clip` | 60, 0.001, 0.0001, 5.0 | AdamW training; full-batch over the training queries |
| `checkpoint_scoring` | true | gradient checkpointing of the scoring chunks to save memory |
| `diffusion_alphas` | [0, 0.25, 0.5, 0.75] | grid for the diffusion reference |
| `retrieval_ks`, `score_chunk_size` | [5, 10, 20, 50, 100], 16384 | evaluation cutoffs and scoring chunk |
| `seed` (top level) | 42 | initialization seed |

`top_k = 5` and `score_mode = bounded_residual` are fixed, as in the paper. Model selection: highest validation known-positive Recall@5, then Hit@5, then the earliest epoch; epoch 0 equals the semantic baseline by construction.

## Metrics

Evaluation reports, for grade thresholds >= 1 and >= 2 and every K in `ks`: macro recall and hit rate of known positives over queries that have at least one known positive, micro recall, how many of the top-K slots were judged at all, and how many judged-irrelevant skills were retrieved. Unjudged skills remain in the ranking and are not counted as errors. Within-pool NDCG@5, precision and pairwise accuracy over judged candidates are reported as secondary numbers.
