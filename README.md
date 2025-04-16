# Evaluation Framework for AI-Generated Veterinary Medical Record Summaries

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) **Repository for the development and validation of a structured framework to evaluate and compare AI-generated summaries against clinician-generated summaries of veterinary medical records.**

This work was developed by researchers at ANI.ML Health Inc and the University of Guelph, and forms the basis of a presentation at the SAVY Conference (May 2025).

## Abstract

**Background:** Medical record summarization is a critical but time-consuming task in veterinary practice. While artificial intelligence (AI) language models show promise in medical text processing, their performance compared to veterinary professionals in summarizing clinical records remains largely unexplored, and standardized evaluation frameworks for such comparisons are lacking.

**Objective:** To develop and validate a structured evaluation framework for assessing veterinary medical record summarizations, and to apply this framework in comparing the performance of AI language models against practicing veterinarians.

*(For full abstract details, please refer to the associated publication/presentation.)*

## Framework Overview

We developed a structured evaluation framework based on five key criteria, weighted according to clinical significance:

1.  **Factual Accuracy** (Weight: 2.5): Consistency and correctness of information compared to the source record.
2.  **Completeness** (Weight: 1.2): Inclusion of all critical information (diagnoses, treatments, events).
3.  **Chronological Ordering** (Weight: 1.0): Clarity and correctness of the timeline of events.
4.  **Clinical Relevance** (Weight: 1.5): Focus on important clinical information, avoiding excessive detail.
5.  **Organization** (Weight: 0.8): Clarity, structure, and readability of the summary.

Summaries were scored on a 1-5 scale for each criterion by an AI-based evaluator.

## Methodology

* **Dataset:** 41 medical record summaries from a diverse set of veterinary cases.
* **Summarizers Compared:**
    * Practicing Veterinarians
    * AI Models (gpt-4o-2024-08-06, claude-3-5-sonnet-20241022, gemini-2.0-flash-exp)
    * Custom Summarization Framework Implementation (developed in this project)
* **Evaluation Approach:** An **LLM-as-judge** approach was used, employing `claude-3-5-sonnet-20241022` as the evaluator to apply the 5-criteria framework consistently.
* **Implementation:** The evaluation process is implemented in this repository using [`promptfoo`](https://www.promptfoo.dev/) to systematically run the evaluator against all summaries.

## Key Results

* The evaluation framework successfully differentiated performance across summarization approaches.
* Baseline AI models and clinicians showed similar performance (Weighted Score ~4.3).
* Our custom summarization framework implementation significantly outperformed clinicians (Weighted Score: **4.801** vs 4.315).
    * Notable improvements were seen in Completeness (4.68 vs 3.56) and Factual Accuracy (4.71 vs 4.07).
* The custom implementation achieved a 100% pass rate (score >= 4.0), compared to 82.9% for clinicians.

These results demonstrate the utility of the framework and the potential for optimized AI implementations to enhance clinical documentation quality.
