# Phase 2A-6A Retail Amazon Reviews Exploration

Phase 2A-6A creates the Retail / E-commerce Support exploration foundation for
Amazon Reviews 2023. The goal is to inspect schema shape, sample quality, review
text characteristics, and product metadata coverage before creating curated
Retail seed prompts, KB/context records, and gold/eval records.

This phase has no RAG, retrieval, embeddings, prompt assembly, model
calls, GPU runs, or benchmark inference. It also does not create the final
Retail prompt dataset.

## Data Source

The planned source is `McAuley-Lab/Amazon-Reviews-2023` on Hugging Face. The
dataset is very large and includes review records plus item metadata records.
It must not be downloaded wholesale. Use controlled sampling by category and
row limit, then review the EDA outputs before seed generation.

Initial category priorities:

- `All_Beauty`
- `Home_and_Kitchen`
- `Electronics`

Additional candidate categories include `Clothing_Shoes_and_Jewelry`,
`Sports_and_Outdoors`, and `Toys_and_Games`.

## Expected Fields

Expected review fields:

- `rating`
- `title`
- `text`
- `images`
- `asin`
- `parent_asin`
- `user_id`
- `timestamp`
- `verified_purchase`
- `helpful_vote`

Expected metadata fields:

- `main_category`
- `title`
- `average_rating`
- `rating_number`
- `features`
- `description`
- `price`
- `images`
- `videos`
- `store`
- `categories`
- `details`
- `parent_asin`
- `bought_together`

## Exploration Outputs

The exploration script works against local JSONL samples and writes generated
outputs under `data/generated/retail/`. These generated files remain local and
ignored.

- `amazon_reviews_exploration_report.json`
- `amazon_reviews_field_profile.json`
- `amazon_reviews_text_profile.json`
- `amazon_reviews_quality_report.json`
- `plots/`
- `word_views/`

If `matplotlib` is available, the script writes lightweight PNG plots for rating
distribution, review length, helpful votes, verified purchase distribution, and
top terms. If plotting is unavailable, it writes text summaries instead.

Word views are always created:

- `top_unigrams.txt`
- `top_bigrams.txt`
- `issue_terms.txt`
- `longest_reviews_preview.txt`
- `shortest_reviews_preview.txt`
- `low_rating_issue_preview.txt`
- `high_rating_positive_preview.txt`

Previews omit `user_id` and redact simple email, number, and local path patterns.

## Quality Checks

The quality profile counts data issues without storing sensitive patterns:

- duplicate review keys when `asin`, `user_id`, and `timestamp` are present
- missing `parent_asin`
- missing or invalid ratings
- very short and very long reviews
- possible spam or low-quality text
- PII-like pattern counts
- HTML-like text counts
- non-ASCII ratio summary

The field profile records fields seen, missing counts, type counts, example
values, and selected unique counts. The text profile records title and review
length summaries, rating and verified-purchase distributions, helpful vote
summary, frequent terms, and common issue terms.

## Commands

Dry-run the plan without loading data:

```text
python scripts/phase2/explore_retail_amazon_reviews.py --dry-run
```

Write committed schema/example samples with fake data:

```text
python scripts/phase2/explore_retail_amazon_reviews.py --write-schema-samples
```

Explore local controlled samples:

```text
python scripts/phase2/explore_retail_amazon_reviews.py --explore-local --reviews-input data/generated/retail/amazon_reviews_sample.jsonl --metadata-input data/generated/retail/amazon_metadata_sample.jsonl
```

Rebuild reports from existing local generated sample files:

```text
python scripts/phase2/explore_retail_amazon_reviews.py --summarize-local
```

## Before Phase 2A-6B

Review the EDA report, field profile, text profile, quality report, plots, and
word views before creating Retail seed prompts. The seed strategy should use
product metadata plus review-derived summaries as KB/context, include answer,
insufficient evidence, escalation, out-of-scope, and spam-or-fraud behaviors,
and avoid committing raw user IDs or bulk raw reviews.

## Phase 2A-6B Controlled Real-Data Loading

Phase 2A-6B adds optional controlled real-data loading from
`McAuley-Lab/Amazon-Reviews-2023` on Hugging Face. Use
`--load-from-huggingface` only for small samples. Do not download the full Amazon
Reviews 2023 dataset.

Generated real samples are written under `data/generated/retail/` and remain
local and ignored. The loader does not write raw `user_id`; it writes a
deterministic `user_id_hash` instead. After loading, the script automatically
runs the same local EDA pipeline and produces the report, field profile, text
profile, quality report, plots, and word views.

Default config names are inferred from the category:

- reviews: `raw_review_<category>`
- metadata: `raw_meta_<category>`

If Hugging Face config names differ, inspect the dataset page and rerun with
explicit `--reviews-config` and `--metadata-config` values.

Controlled load command:

```text
python scripts/phase2/explore_retail_amazon_reviews.py --load-from-huggingface --category All_Beauty --sample-limit 1000 --metadata-limit 1000
```

Explore existing local generated samples:

```text
python scripts/phase2/explore_retail_amazon_reviews.py --explore-local --reviews-input data/generated/retail/amazon_reviews_sample.jsonl --metadata-input data/generated/retail/amazon_metadata_sample.jsonl
```

Inspect the generated report:

```text
python -m json.tool data/generated/retail/amazon_reviews_exploration_report.json
```

Review checklist before Retail seed creation:

- rating distribution
- review length distribution
- top terms
- issue terms
- low-rating examples
- metadata product coverage
- missing fields
- duplicate and quality flags
- whether the sample is good enough for Phase 2A-6C seed creation

## Next Step

After controlled EDA review, proceed to Phase 2A-6C Retail curated seed
creation.
