# Business Understanding (BU)

**Phase:** [BU] Business Understanding  
**Date:** 2024  
**Team:** Team 16  

## Problem Statement

The Zambian National Assembly generates extensive debate transcripts during parliamentary sittings, but currently lacks an automated system to identify which speaker utterances are directly relevant to the motions being discussed. This creates challenges for:

- **Parliamentary staff** who need to index and search through Hansard records efficiently
- **Researchers and journalists** who want to analyze parliamentary discourse and voting patterns
- **Citizens** who seek to understand how their representatives engage with specific policy issues

The problem is to **classify each speaker turn in parliamentary debates as either "Relevant" or "NotRelevant" to the motion under discussion**, where:

- **Relevant**: utterances that argue for/against the motion, provide supporting evidence, propose amendments, or discuss implementation
- **NotRelevant**: procedural points, greetings, tangential discussions, jokes, or administrative matters

## Intended Users

1. **Parliamentary Information Services** - for automated indexing and search
2. **Political researchers** - for discourse analysis and policy tracking  
3. **Journalists** - for identifying key arguments and positions on specific motions
4. **Civic organizations** - for monitoring representative engagement with issues

## Key Performance Indicators (KPIs)

### Primary Metrics

- **Macro-F1 Score** ≥ 0.75 (balanced performance across both classes)
- **Relevant Class Recall** ≥ 0.80 (capture most relevant utterances)
- **Area Under Precision-Recall Curve (AUPRC)** ≥ 0.70 (handle class imbalance)

### Secondary Metrics

- Balanced Accuracy ≥ 0.75
- Per-sitting performance consistency
- Per-speaker performance analysis

## Scope and Constraints

### In Scope

- Parliamentary debates and proceedings from Zambian National Assembly
- English language utterances
- Motions from Order Papers (substantive motions, not procedural)
- Speaker turns with clear attribution and timestamps

### Out of Scope

- Committee proceedings (different format and context)
- Question Time sessions (different interaction patterns)
- Languages other than English
- Real-time classification (batch processing acceptable)

### Data Coverage

- Target: 6-10 parliamentary sittings from 2023
- Minimum: 1,000 manually labeled utterances for training
- Time range: Representative sample across different motion types

## Risks and Assumptions

### Technical Risks

- **Low inter-annotator agreement** - Complex cases may be subjectively labeled
- **Class imbalance** - Most utterances may be relevant, creating skewed training data
- **Context dependency** - Relevance may require understanding previous utterances
- **Motion complexity** - Compound motions may have multiple relevant topics

### Business Risks

- **Annotation quality** - Inconsistent labeling could hurt model performance
- **Generalizability** - Model may not work well on different parliamentary systems
- **Deployment complexity** - Integration with existing parliamentary systems

### Key Assumptions

- Parliamentary transcripts are accurately transcribed with speaker attribution
- Order Papers correctly identify the motions being debated
- Manual annotation can achieve reasonable consistency (κ ≥ 0.75)
- TF-IDF and transformer features will capture relevance patterns effectively

## Ethical Considerations

- **Transparency**: Classification decisions should be explainable to users
- **Bias**: Ensure model doesn't discriminate based on speaker identity or political affiliation  
- **Privacy**: No personal information beyond public parliamentary records
- **Accuracy**: False classifications could misrepresent parliamentary discourse

### Extended Ethical Considerations

- **Transparency and Explainability**: Classification decisions should be explainable to users, enabling parliamentary staff and researchers to understand why utterances are classified as relevant or not relevant. The model should provide confidence scores and feature importance to support decision transparency (Westminster Foundation for Democracy, 2024).

- **Political Neutrality and Non-Discrimination**: Ensure the model doesn't discriminate based on speaker identity, political affiliation, or political orientation, as algorithmic bias against political viewpoints can arise in AI systems (Leerssen, 2022). The model should perform consistently across all political parties and individual representatives.

- **Fairness Across Demographics**: Minimize bias and ensure fairness across different speaker demographics, including gender, age, constituency, and years of service to prevent systematic disadvantaging of any group (Inter-Parliamentary Union, 2024).

- **Data Representativeness**: Address potential correlations that may overlap with protected categories or political viewpoints by ensuring training data represents diverse speakers, motion types, and parliamentary contexts to prevent accidental bias (Wikipedia, 2025).

- **Contextual Sensitivity**: Respect the cultural and institutional context of Zambian parliamentary discourse, ensuring the model doesn't impose external definitions of relevance that may not align with local parliamentary traditions and practices.

- **Privacy and Consent**: Maintain appropriate handling of public parliamentary records while respecting speaker attribution and ensuring no personal information beyond publicly available Hansard records is used in the classification system.

- **Accountability and Human Oversight**: Promote human autonomy and decision-making by designing the system as a decision-support tool rather than a replacement for human judgment, with clear protocols for human review of classifications (Westminster Foundation for Democracy, 2024).

- **Impact Assessment**: Monitor for unintended consequences such as potential chilling effects on parliamentary speech or systematic misrepresentation of certain speakers' contributions to debates.

**References:**
- Inter-Parliamentary Union. (2024). Ethical principles: Fairness and non-discrimination. AI Guidelines for Parliaments.
- Leerssen, P. (2022). Algorithmic Political Bias in Artificial Intelligence Systems. Philosophy & Technology, 35(2).
- Westminster Foundation for Democracy. (2024). AI guidelines for parliaments.
- Wikipedia. (2025). Algorithmic bias.

## Success Criteria

The project will be considered successful if:

1. Achieves target KPI thresholds on held-out test data
2. Demonstrates consistent performance across different sittings and speakers
3. Provides interpretable predictions with confidence scores
4. Delivers a working CLI tool for batch classification
5. Completes all CRISP-DM phases with proper documentation

---

**Next Phase:** [DU] Data Understanding - Scraping and exploratory data analysis
