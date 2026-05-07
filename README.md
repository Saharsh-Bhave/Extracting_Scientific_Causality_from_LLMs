# Extracting_Scientific_Causality_from_LLMs

## Overview

This project evaluates the ability of Large Language Models (LLMs) to identify and extract causal relationships from complex scientific literature across domains such as Biology, Chemistry, Physics, and Healthcare. The goal is to understand how effectively modern LLMs can interpret scientific findings and distinguish causal relationships from other forms of associations in research papers.

The rapid growth of scientific publications has made manual synthesis of research increasingly difficult. According to the National Library of Medicine, PubMed alone contains over 35 million biomedical citations, with thousands of new papers added every week. Extracting meaningful causal insights from this massive volume of literature is a major challenge for researchers and healthcare professionals.

Traditional information extraction systems often struggle with implicit and domain-specific causal reasoning commonly found in scientific writing. While transformer-based models such as GPT and BERT have shown strong performance in general NLP tasks, their ability to accurately extract structured causal relationships from scientific literature remains underexplored.

This project aims to bridge that gap by systematically evaluating multiple LLMs on human-annotated scientific datasets and analyzing their strengths, weaknesses, and limitations in causal reasoning.

---

## Objectives

* Evaluate the performance of multiple LLMs in extracting causal relationships from scientific literature
* Compare LLM-generated annotations with human annotations
* Analyze model performance across different scientific domains
* Identify limitations in current LLM-based causal extraction approaches
* Explore hypothesis generation capabilities in a selected scientific domain

---

## Methodology

### 1. Human Annotation

* Annotate approximately **600 scientific papers**
* Papers span multiple domains:

  * Biology
  * Chemistry
  * Physics
  * Healthcare
* Human annotators identify:

  * Causal relationships
  * Non-causal relationships
  * Relation types

### 2. LLM-Based Annotation

* Multiple LLMs are prompted using detailed annotation guidelines
* Models generate:

  * Conclusions
  * Causality labels
  * Relation classifications

### 3. Evaluation

* Compare LLM outputs with human annotations
* Measure:

  * Accuracy
  * Precision
  * Recall
  * F1-score
  * Weighted F1-score
* Analyze model behavior across domains and relation types

### 4. Hypothesis Generation

* Select one scientific domain for deeper evaluation
* Use LLMs to generate scientific hypotheses based on extracted relationships
* Evaluate the quality and relevance of generated hypotheses

---

## Technologies & Tools

* Python
* Pandas
* Scikit-learn
* Large Language Models (GPT, Claude, Gemini, etc.)
* Prompt Engineering
* NLP Evaluation Metrics
* CSV/Excel-based Annotation Pipelines

---

## Expected Impact

This research can improve AI-driven literature review systems and knowledge synthesis workflows across multiple scientific fields. In healthcare, for example, fine-tuned models capable of identifying causal relationships may help uncover disease mechanisms by learning from symptoms, treatments, and biomedical research findings.

The project also contributes to understanding the limitations of modern LLMs in scientific reasoning and structured information extraction.

---

## Timeline

**Duration:** February 24, 2026 – May 10, 2026
**Mentorship:** Dr. Xuan Lu

---

## Contributors

* Saharsh Bhave
* Tirth Shah
* Siddharth Bhattacharjee
* Research Team under Dr. Xuan Lu
