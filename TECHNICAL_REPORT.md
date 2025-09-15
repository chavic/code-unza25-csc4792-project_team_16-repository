# Automated Classification of Parliamentary Utterances: A Machine Learning Approach to Motion Relevance Detection

**Victor Chabunda**, **Constance Chilamo**, **Francis Kalunga**
University of Zambia, School of Engineering  
{francis.kalunga, victor.chabunda, chilamo.constance}@cs.unza.zm

## Abstract

This paper presents an automated system for classifying parliamentary utterances as relevant or not relevant to their associated motions. We implement a complete end-to-end pipeline from data collection to model deployment, utilizing both traditional machine learning approaches (TF-IDF with Logistic Regression, SVM) and advanced natural language processing techniques (Sentence Transformers). Our system achieves 98% accuracy on a dataset of 500 labeled utterances from the Zambian National Assembly, though analysis reveals significant class imbalance challenges (97.6% vs 2.4% distribution). We demonstrate the effectiveness of automated annotation using Large Language Models (Gemma 3:270M) and provide a comprehensive evaluation framework. The system successfully identifies relevant utterances with 100% recall but struggles with minority class detection, highlighting the critical importance of addressing class imbalance in real-world classification tasks.

**Keywords:** Parliamentary Analysis, Text Classification, Machine Learning, Natural Language Processing, Class Imbalance

## 1. Introduction

Parliamentary debates generate extensive textual data that requires systematic analysis for effective governance, research, and civic engagement. The Zambian National Assembly produces detailed transcripts of parliamentary proceedings, but currently lacks automated systems to identify which speaker utterances are directly relevant to specific motions under discussion. This creates significant challenges for parliamentary staff, researchers, and citizens who need to efficiently navigate and analyze parliamentary discourse.

### 1.1 Problem Statement

Given a parliamentary motion and the corresponding debate transcript, the challenge is to automatically classify each Member of Parliament (MP) utterance as either "Relevant" or "Not Relevant" to that motion. Relevant utterances include arguments for/against the motion, supporting evidence, proposed amendments, or implementation discussions. Not relevant utterances encompass procedural points, greetings, tangential discussions, jokes, or administrative matters.

### 1.2 Research Objectives

Our primary objectives are to:
1. Develop an automated pipeline for collecting and processing parliamentary data
2. Implement multiple machine learning approaches for utterance classification
3. Evaluate model performance across different architectures and feature representations
4. Identify and address challenges in real-world classification scenarios
5. Create a deployable system for practical parliamentary analysis

### 1.3 Contributions

This work makes several key contributions:
- **Complete Pipeline**: End-to-end system from data scraping to model deployment
- **Automated Annotation**: LLM-based labeling system using Gemma 3:270M
- **Multi-Model Comparison**: Comprehensive evaluation of traditional and advanced NLP approaches
- **Real-World Analysis**: Identification and analysis of class imbalance challenges
- **Deployable System**: Web-based demonstration application for practical use

## 2. Related Work

### 2.1 Parliamentary Text Analysis

Previous work in parliamentary text analysis has focused on various aspects including sentiment analysis [1], topic modeling [2], and speaker behavior analysis [3]. However, limited research exists on automated relevance classification for parliamentary utterances, particularly in African parliamentary contexts.

### 2.2 Text Classification Approaches

Text classification has been extensively studied using both traditional machine learning methods and deep learning approaches. TF-IDF with linear classifiers remains effective for many tasks [4], while transformer-based models have shown superior performance on complex language understanding tasks [5].

### 2.3 Class Imbalance in NLP

Class imbalance is a well-documented challenge in machine learning, particularly in text classification where minority classes may be underrepresented [6]. Various techniques including SMOTE [7], cost-sensitive learning [8], and ensemble methods [9] have been proposed to address this issue.

## 3. Methodology

### 3.1 Data Collection Pipeline

Our system implements a comprehensive data collection pipeline targeting the Zambian Parliament website (parliament.gov.zm). The pipeline consists of two main components:

**Debate Scraping**: Automated collection of parliamentary debate transcripts in HTML format, extracting speaker information, timestamps, and utterance text.

**Order Paper Scraping**: Collection of parliamentary order papers containing motion texts and procedural information in JSON format.

The scraping system includes polite crawling practices with request throttling, content hashing for duplicate detection, and robust error handling for website structure variations.

### 3.2 Data Processing and Segmentation

We developed a sophisticated HTML parser using BeautifulSoup to extract structured utterances from parliamentary debate pages. The parser:

- Identifies speaker turns using heading patterns and stage markers
- Extracts timestamps and session metadata
- Links utterances to relevant motions using Jaccard similarity
- Generates structured output in JSONL format

**Motion-Utterance Linkage**: Utterances are linked to motions based on date matching and semantic similarity using Jaccard coefficient between motion and utterance text.

### 3.3 Automated Annotation System

To address the challenge of manual annotation, we implemented an automated labeling system using Large Language Models:

**Model**: Gemma 3:270M via local Ollama installation
**Prompt Engineering**: Carefully crafted prompts with strict relevance criteria and examples
**Quality Control**: Confidence scoring and explanation generation for each annotation
**Validation**: Systematic evaluation of annotation consistency

The annotation system processes utterances in batches, providing detailed explanations for classification decisions and confidence scores for quality assessment.

### 3.4 Model Architectures

We implemented and compared multiple machine learning approaches:

#### 3.4.1 Baseline Models (TF-IDF Features)
- **Features**: 7,148 TF-IDF features (1-2 grams, English stopwords)
- **Classifiers**: Logistic Regression, Support Vector Machine
- **Preprocessing**: Text normalization, stopword removal, motion-utterance concatenation

#### 3.4.2 Advanced Models (Sentence Transformers)
- **Base Model**: all-MiniLM-L6-v2 (384-dimensional embeddings)
- **Classifiers**: Logistic Regression, SVM, Random Forest
- **Text Processing**: Motion-utterance concatenation with [SEP] token
- **Feature Engineering**: Sentence-level semantic embeddings

### 3.5 Evaluation Framework

We implemented a comprehensive evaluation framework including:

**Metrics**: Accuracy, Macro F1, Per-class Recall/Precision, Confusion Matrix
**Data Splitting**: Sitting-wise train/validation/test splits (70/15/15) to prevent data leakage
**Cross-Validation**: Stratified sampling to maintain class distribution
**Visualization**: Confusion matrices, performance comparison charts

## 4. Experimental Setup

### 4.1 Dataset

Our dataset consists of:
- **Source**: Zambian National Assembly parliamentary proceedings
- **Time Period**: 2024 parliamentary sessions
- **Scale**: 500 labeled utterances across multiple sittings
- **Class Distribution**: 488 RELEVANT (97.6%), 12 NOT_RELEVANT (2.4%)
- **Imbalance Ratio**: 40.7:1

### 4.2 Implementation Details

**Environment**: Python 3.11, scikit-learn, transformers, sentence-transformers
**Hardware**: Standard development machine with GPU acceleration for transformer models
**Reproducibility**: Fixed random seeds (42) across all experiments
**Validation**: 80/20 train-test split with stratification

### 4.3 Training Configuration

**Baseline Models**:
- Logistic Regression: max_iter=1000, default regularization
- SVM: RBF kernel, default parameters

**Advanced Models**:
- Sentence Transformer: all-MiniLM-L6-v2 pre-trained model
- Classifiers: Default parameters for fair comparison

## 5. Results and Analysis

### 5.1 Model Performance Comparison

| Model Type | Model | Accuracy | Macro F1 | Relevant Recall | Relevant Precision |
|------------|-------|----------|----------|-----------------|-------------------|
| **Baseline** | Logistic Regression | 98.0% | 49.5% | 100.0% | 98.0% |
| **Baseline** | SVM | 98.0% | 49.5% | 100.0% | 98.0% |
| **Advanced** | Sentence Transformer + Logistic | 98.0% | 49.5% | 100.0% | 98.0% |
| **Advanced** | Sentence Transformer + SVM | 98.0% | 49.5% | 100.0% | 98.0% |
| **Advanced** | Sentence Transformer + Random Forest | 98.0% | 49.5% | 100.0% | 98.0% |

### 5.2 Confusion Matrix Analysis

All models exhibit identical confusion matrices:
```
                Predicted
Actual    NOT_RELEVANT  RELEVANT
NOT_RELEVANT      0        2
RELEVANT          0       98
```

**Key Observations**:
- **Perfect RELEVANT Recall**: 100% - models successfully identify all relevant utterances
- **Zero NOT_RELEVANT Recall**: 0% - models fail to identify any non-relevant utterances
- **High Accuracy**: 98% achieved through majority class prediction strategy

### 5.3 Class Imbalance Impact

The severe class imbalance (97.6% vs 2.4%) fundamentally affects model behavior:

**Model Convergence**: All models converge to identical prediction strategies, defaulting to majority class prediction
**Feature Engineering Impact**: Advanced sentence transformers show no improvement over traditional TF-IDF features
**Architecture Impact**: Different classifiers (Logistic, SVM, Random Forest) produce identical results

### 5.4 Automated Annotation Quality

The LLM-based annotation system demonstrates:
- **High Confidence**: Average confidence score of 0.89 across all annotations
- **Consistent Decisions**: Detailed explanations for each classification
- **Scalable Processing**: Efficient batch processing of large datasets

## 6. Discussion

### 6.1 Key Findings

**Technical Success**: The system successfully implements a complete end-to-end pipeline with multiple model architectures and comprehensive evaluation.

**Performance Limitations**: Despite high accuracy (98%), the system fails to address the core challenge of minority class detection, with zero recall for NOT_RELEVANT utterances.

**Class Imbalance Dominance**: The 40.7:1 imbalance ratio overwhelms model differences, causing all architectures to converge to identical prediction strategies.

### 6.2 Implications

**Practical Utility**: While the system correctly identifies relevant utterances, its inability to detect non-relevant content limits practical applications in parliamentary analysis.

**Feature Engineering**: The equivalence of TF-IDF and sentence transformer performance suggests that feature quality is not the limiting factor in this domain.

**Model Architecture**: The identical performance across different classifiers confirms that the problem lies in data distribution rather than model choice.

### 6.3 Limitations

**Data Scale**: Limited to 500 labeled examples may not be sufficient for robust model training
**Annotation Quality**: LLM-based annotations require validation against human judgments
**Domain Specificity**: Results may not generalize to other parliamentary systems or languages

## 7. Future Work

### 7.1 Immediate Improvements

**Address Class Imbalance**:
- Implement SMOTE or other oversampling techniques
- Apply cost-sensitive learning with class weights
- Use threshold tuning for better minority class recall

**Data Expansion**:
- Collect more parliamentary sessions
- Increase NOT_RELEVANT examples through targeted sampling
- Validate annotation quality through inter-annotator agreement studies

### 7.2 Advanced Techniques

**Ensemble Methods**: Combine models with different sampling strategies
**Active Learning**: Identify most informative examples for annotation
**Advanced Architectures**: Fine-tune transformer models on parliamentary data

### 7.3 System Enhancements

**Real-time Processing**: Stream processing for live parliamentary debates
**Multi-language Support**: Extend to other languages and parliamentary systems
**Advanced Analytics**: Speaker behavior analysis and topic modeling

## 8. Conclusion

This work presents a comprehensive system for automated parliamentary utterance classification, successfully implementing a complete pipeline from data collection to model deployment. While achieving high accuracy (98%), the system reveals critical challenges in real-world classification scenarios, particularly the dominance of class imbalance over model architecture choices.

The key contribution is not just the technical implementation, but the identification and analysis of class imbalance as the primary limiting factor in parliamentary text classification. This finding has important implications for future work in this domain and highlights the need for specialized techniques to address minority class detection in parliamentary analysis.

The system provides a solid foundation for parliamentary discourse analysis and demonstrates the potential for automated systems in government transparency and civic engagement applications.

## Acknowledgments

We thank the University of Zambia for providing the computational resources and academic support for this research. We also acknowledge the Zambian National Assembly for making parliamentary proceedings publicly available, enabling this research.

## References

[1] Proksch, S. O., & Slapin, J. B. (2015). The politics of parliamentary debate: Parties, rebels and representation. Cambridge University Press.

[2] Laver, M., Benoit, K., & Garry, J. (2003). Extracting policy positions from political texts using words as data. American Political Science Review, 97(2), 311-331.

[3] Slapin, J. B., & Proksch, S. O. (2008). A scaling model for estimating time-series party positions from texts. American Journal of Political Science, 52(3), 705-722.

[4] Joachims, T. (1998). Text categorization with support vector machines: Learning with many relevant features. European conference on machine learning, 137-142.

[5] Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. arXiv preprint arXiv:1810.04805.

[6] He, H., & Garcia, E. A. (2009). Learning from imbalanced data. IEEE Transactions on knowledge and data engineering, 21(9), 1263-1284.

[7] Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: synthetic minority oversampling technique. Journal of artificial intelligence research, 16, 321-357.

[8] Elkan, C. (2001). The foundations of cost-sensitive learning. International joint conference on artificial intelligence, 17(1), 973-978.

[9] Galar, M., Fernandez, A., Barrenechea, E., Bustince, H., & Herrera, F. (2012). A review on ensembles for the class imbalance problem: bagging-, boosting-, and hybrid-based approaches. IEEE Transactions on Systems, Man, and Cybernetics, Part C, 42(4), 463-484.

---

**Technical Report** - CSC 4792 Group Project - Team 16  
**Automated Classification of Parliamentary Utterances**  
**University of Zambia, School of Engineering**  
**2025**
