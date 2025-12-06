# RFT Example Projects

Each folder below contains a curated example dataset and configuration tailored to a domain. Run `rft demo` from the project root to open the main app, then point to any of these folders when uploading data.

## Available examples

| Example | Description |
| --- | --- |
| `sentiment_analysis/` | IMDB-style movie reviews with binary sentiment labels (\'Positive\' / \'Negative\'). Ideal for exploring how RFT handles balanced sentiment data. |
| `medical_document_classification/` | De-identified clinical notes with labels like `Diagnosis` / `Treatment`. Shows how to profile longer, structured medical text. |
| `legal_document_review/` | Contracts & clauses annotated with review statuses. Use this to see how human corrections shift model risk decisions and confidence plots. |
| `customer_feedback/` | Short-form support tickets, each marked for urgency and sentiment. Demonstrates multi-label or multi-metric evaluation while keeping the UI responsive. |

Each example folder should include a README with dataset guidance plus a `sample.csv` / `sample.txt` file. Feel free to swap in your own data for experiments.