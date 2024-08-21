# Introduction to Artificial Intelligence and Data Science

## Siri Query Processing

### Query Example
"Hey, Siri. Show me a good breakfast restaurant near me."

### Processing Steps

1. **Speech-to-Text Conversion**
   - Convert spoken query to text format

2. **Semantic Analysis**
   - Identify key terms: breakfast, restaurant
   - Formulate structured query:
     - Place type: restaurant
     - Meal type: breakfast
     - Rating: 3-5 stars
     - Distance: < 3 km

3. **Location-Based Search**
   - Utilize the user's current location
   - Search for restaurants meeting the criteria
   - Filter and rank results based on ratings and other metrics

4. **Natural Language Processing (NLP)**
   - Analyze written reviews for sentiment
   - Contribute to overall rating assessment

5. **Language Translation**
   - Potential on-site translation of menu (e.g., Kannada to English)

## Types of Data

### 1. Tabular Data
- Most common form of data.
- Widely applicable in various business use cases.
- Structured as rows representing data points and columns representing features.

### 2. Time-series Data
- A form of tabular data that is ordered by time.
- Represents data points collected at different time intervals.

### 3. Image Data
- Increasingly popular in recent years.
- Typically structured as a number of data points × height × width × sensor channels.
- Primarily used in vision-related tasks.

### 4. Video Data
- A sequence of image data over time.
- A form of time series data was considered in the context of vision tasks.

### 5. Text Data
- Utilized in language processing tasks.
- Represented as a text corpus that needs conversion to numerical form for further processing.

### 6. Audio Data
- Involved in language and speech processing tasks.
- Represented as an audio recording corpus requiring signal processing techniques.

## Continuous vs Categorical Data

### 1. Continuous Data
- Data that can take any numerical value within a range.
- Often associated with regression tasks in predictive modelling.
- Examples include temperature, height, and sales figures.

### 2. Categorical Data
- Data that is divided into distinct groups or categories.
- Commonly used in classification tasks.
- Examples include gender, colours, and product types.

## Predictive AI vs Generative AI

### 1. Predictive AI
- Focuses on making predictions based on existing data.
- Utilizes supervised learning models, like regression and classification.
- Examples include forecasting sales, predicting stock prices, and diagnosing diseases.

### 2. Generative AI
- Aims to create new data instances similar to the training data.
- Involves models that learn the distribution of data to generate new content.
- Examples include generating images, text, audio or video.

## Predictive AI

### 1. Supervised Learning
- **Do you know the targets/labels in your dataset?** Yes.
- **Is the target a continuous variable or class?**
  - **Value**: Regression
    - Example Models:
      1. Linear Regression
      2. Tree-Based Models
      3. Neural Networks
  - **Class**: Classification
    - Example Models:
      1. Logistic Regression
      2. Tree-Based Models
      3. Neural Networks

### 2. Unsupervised Learning
- **Do you know the targets/labels in your dataset?** No.
- **Need to find patterns/clusters?**
  - **Clustering**:
    - Example Models:
      1. K-Means
      2. DBSCAN
      3. Agglomerative Clustering
      4. Gaussian Mixture Model (GMM)
- **Need to find anomaly?**
  - **Anomaly Detection**:
    - Example Models:
      1. Gaussian Mixture Model (GMM)
      2. Isolation Forest
      3. Principal Component Analysis (PCA)

## Generative AI

### 1. Types of Generative Tasks
- **Text to Text**: Generates new text based on a text prompt.
- **Text to Image/Video**: Converts textual descriptions into images or videos.
- **Image/Video to Text**: Generates captions or descriptions for images or videos.
- **Image/Video to Image/Video**: Transforms images or videos into other images or videos (e.g., style transfer).
- **Text to Audio**: Converts text into spoken audio or music.
- **Audio to Text**: Converts speech or audio into text.
- **Text/Image to Code**: Generates code from natural language or images (e.g., diagrams).

### 2. Key Elements
- **Input**: The input to a generative AI model is often called a "prompt."
- **Model**: These tasks are typically handled by large language or vision models.
- **Output**: The model generates outputs like images, videos, text, or speech based on the prompt.

## Foundational Model (Base Model)

### 1. Overview
- A large-scale AI model trained on vast amounts of diverse data.
- Acts as a foundational base for various downstream tasks and applications.

### 2. Key Characteristics
- **Broad Knowledge**: Possesses extensive knowledge and general capabilities across multiple domains.
- **Prompt Engineering**: Requires carefully crafted prompts to perform specific tasks effectively.
- **Retrieval-Augmented Generation (RAG)**: Can access specific data through retrieval methods to enhance performance.
- **Adaptability**: Easily adaptable to new tasks through fine-tuning.
- **Generalization**: Capable of generalizing to new tasks with minimal additional training.

### 3. Applications
- **Summarization**
- **Question-Answering**
- **Instruction Following**
- **Rewriting in Different Styles**

### 4. Examples
- GPT
- BERT
- T5

## AI/ML Workflow

### 1. Frame the AI Problem
- **Align with Business Needs**: Understand the core business requirements.
  - Identify subproblems related to one or more of the five tasks a computer can perform.
  - Establish the current baseline: What is currently being done?
  - Define success criteria: What does success look like?

### 2. Gather Data & Data Preparation
- **Data Exploration**: Analyze and explore the available data.
- **Data Munging/Wrangling**: Clean, preprocess, and prepare the data for use in downstream machine learning models.
- **Establish Baselines**: Set baselines for the data, domain knowledge, and the state-of-the-art (SOTA).

### 3. Model Exploration & Improvement
- Experiment with different models.
- Enhance performance using Cross-Validation.
- Consider designing new models to address specific challenges.

### 4. Ensemble Models
- Combine multiple models and solutions to improve performance and robustness.

### 5. Present the Solution
- **Tell a Data Story**: Effectively communicate the insights and solutions using data storytelling.

### 6. Deploy
- Implement the solution into production or operational environments.

## Data Analytics

### 1. Descriptive Analytics
- **Question**: What happened?
- **Techniques**: Summary statistics, histograms.
- **Foundations**: Statistics, data visualization, matrix algebra.

### 2. Diagnostic Analytics
- **Question**: Why did it happen? What is the cause?
- **Techniques**: Correlation, covariance, entropy, mutual information.
- **Foundations**: Probability, statistics.

### 3. Predictive Analytics
- **Question**: What will happen next?
- **Techniques**: Machine learning (ML), deep learning (DL), and time series models.
- **Foundations**: Probability, calculus, matrix algebra.

### 4. Prescriptive Analytics
- **Question**: What action should be taken to achieve the desired result? (If...then scenarios)
- **Techniques**: ML/DL models, data-driven optimization.
- **Foundations**: Probability, calculus, matrix algebra, optimization.
