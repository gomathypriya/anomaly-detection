# Anomaly-detection

      - Project Title: Rare Class Prediction through Anomalay Detection
      - Objective: To create a uniform analytical framework for anomaly detection that can provide consistent performance across domains without customization.
      
 Project Details: Final Year Capstone Project
 ============================================
      - University & School : National University of Singapore (Institute of Systems Science)
      - Course Name : Master of Technology in Enterprise Business Analytics
      - Team Member : Dr. Acebedo, Cleta Milagros Libre, Yu Jun, Gomathypriya Dhanapal, Sougata Deb 
      
Prerequisites:
==============
      - Install RStudio (Tool used: RStudio Version 0.98.1091 – © 2009-2014 RStudio, Inc.)
      - Install Python 3.6 (Tool used: Anaconda Navigator: Spyder 3.2.3) 

Data Resources:
==============
      - For framework development: 4 datasets from 4 different domains: 
            - Dataset 1 (proprietary) from an institute of higher studies for student failure prediction.  
            - Dataset 2 (proprietary) from an IT company for poor VOIP call quality prediction. 
            - Dataset 3 (public) from BayesiaLab for severe/fatal accident prediction. 
            - Dataset 4 (public) from Kaggle for credit card fraud prediction.

      - For framework validation: 12 public datasets from 4 different sources: 
            - Stony Brook University ODDS datasets, 
            - Harvard Dataverse Anomaly Detection Benchmark datasets, 
            - Kaggle PaySim dataset and 
            - University of California Irvine ML repository.  

Code Short Desctiption:
=======================
  Python Code
  -----------
      - First Python code was created based on the BayesiaLab data but can easily be customized for other datasets. 
      - It demonstrates step-by-step process for basic data cleaning, followed by building different supervised models.
        
      Supervised models:
                  - Logistic Regression
                  - Naïve Bayes

      - Second Python code is for benchmarking performance of the prominent anomaly detection techniques. 
      - This code takes the cleaned datasets as inputs. Hence the cleaned datasets from first code can be used as inputs here.

      Anomaly detection models:
                  - Local Outlier Factor
                  - One class SVM
                  - Isolation Forest
                  
      In both cases, the model performance is evaluated using ROC-AUC and F1 Score for the rare class.

  R code
  ------
  
Generate Multiple Solutions R code:

      - Used to generate multiple (6,480) predictions from the same dataset by using different preprocessing, variable selection and anomaly detection models. 
      - This code was used on the 4 development dataset to assess impact of each preprocessing, variable selection or model building option on final prediction performance.
      - These results were subsequently analyzed to come up with the proposed uniform analytical framework.
      
Apply Proposed Framework R code:

      - Implementation of our proposed uniform analytical framework. 
      - Apart from the basic data features, it also needs an additional input on the predictor data type: H(uman) or M(achine).
      - Behavioral and other human generated / interpreted / captured predictors should be categorized as “H”, 
            e.g. age, purchase volume, examination score, amount transferred.       
      - On the other hand, data generated / interpreted / captured or related to machines should be categorized as “M”, 
            e.g. image pixels, network delays, blood test results.            
      - If your data has a mix of both, choose the option based on majority type.

This R code will automatically perform the following steps:

      - Impute missing values for both numeric and categorical predictors.
      - Convert categorical predictors to numeric using Relative Frequency Encoding
      - Apply a z-score standardization on numeric predictors
      - Apply PCA for variable selection (variance explained: 0.90)
      - Build the following predictive models
      - Semi-supervised Autoencoder Neural Network (AEN)
      - Semi-supervised One Class SVM
      - Unsupervised KNN (uses average distance to nearest neighbours as anomaly score)
      - Normalize the predicted anomaly scores using a Min-Max transformation
      - Create a dynamic ensemble prediction based on the data type (specified) and event rate (observed in training sample)
      - Output the best F1 score for the final ensemble prediction

      This code does not require any parameter tuning or customization apart from the data specifications. 
      It adjusts the implicit parameters, such as K for KNN or no. of hidden nodes for AEN based on the training data characteristics.

    
 # Note:
    - Run the scripts as per the step number mentioned infront of each python and R script
