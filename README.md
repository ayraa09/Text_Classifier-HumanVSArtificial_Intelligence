# **Human vs AI: Text Classification**

## **Overview**
In recent years, artificial intelligence has advanced significantly, leading to the widespread use of AI-generated text in domains such as journalism, research, and creative writing. While AI-generated text can be highly sophisticated, distinguishing between human-written and machine-generated content remains a crucial challenge.  

This project focuses on building a text classification model to differentiate between AI-generated and human-written text. The insights gained from this analysis can have practical applications in academic integrity, misinformation detection, and AI content validation.

## **Dataset**
The dataset used for this project is sourced from Hugging Face:  
[NicolaiSivesind/human-vs-machine (research_abstracts_labeled)](https://huggingface.co/datasets/NicolaiSivesind/human-vs-machine).  

The dataset is split into training, validation, and test sets. For this project, we use the training and test sets.

## **Project Workflow**
1. **Data Loading and Exploration**  
   - Importing the dataset  
   - Analyzing data distribution  
   - Visualizing text characteristics (word clouds, histogram, box plots)

2. **Preprocessing & Feature Engineering**  
   - Text cleaning (removing stopwords, special characters, and lowercasing)  
   - Converting text into numerical representations using **TF-IDF Vectorization**  

3. **Model Training & Evaluation**  
   The following machine learning models were trained and compared:
   - **Logistic Regression** (Chosen model)
   - **Random Forest Classifier**
   - **Naïve Bayes**
   - **Support Vector Machine (SVM)**

   **Performance Comparison:**
   | Model               | Accuracy |
   |---------------------|----------|
   | Logistic Regression | 96%      |
   | Random Forest       | 95%      |
   | Naïve Bayes        | 90%      |
   | SVM                | 96%      |

   **Key Observations:**  
   - Logistic Regression and SVM performed the best with **96% accuracy**.  
   - SVM had issues with recall and computational efficiency, making **Logistic Regression the preferred choice**.  

4. **Deployment with Streamlit**  
   - The trained model is deployed using **Streamlit** for real-time text classification.  
   - Users can input text, and the model will classify it as either "Human-Written" or "AI-Generated" with a confidence score.

## **Installation & Usage**
### **Requirements**
- Python 3.x
- Required libraries:
  ```bash
  pip install numpy pandas matplotlib seaborn nltk textblob sc
