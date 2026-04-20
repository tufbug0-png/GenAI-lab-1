# Human Activity Recognition (HAR) using WISDM Dataset

##  Overview
This project builds a Deep Learning model to classify human activities using tri-axial accelerometer data from the WISDM dataset.  
The model learns patterns from time-series sensor data to predict activities like walking, jogging, sitting, etc.

---

##  Objective
- Process accelerometer data (x, y, z)
- Apply sliding window segmentation
- Train a deep learning model (1D CNN)
- Classify human activities accurately

---

## Dataset
- File: time_series_data_human_activities.csv
- Features:
  - X-axis acceleration
  - Y-axis acceleration
  - Z-axis acceleration
- Target:
  - Activity labels

---

## Steps Performed

### 1. Data Loading
- Loaded dataset using pandas
- Checked structure and null values

### 2. Preprocessing
- Label encoding for activities
- Feature scaling using StandardScaler

### 3. Sliding Window
- Window size: 128
- Step size: 64
- Converts raw data into sequences

### 4. Train-Test Split
- Stratified split for balanced classes

### 5. Model
1D CNN Architecture:
- Conv1D (64 filters)
- MaxPooling
- Conv1D (128 filters)
- MaxPooling
- Flatten
- Dense layers

### 6. Training
- Optimizer: Adam
- Loss: categorical_crossentropy
- Epochs: 10
- Batch size: 64

### 7. Evaluation
- Accuracy on test data
- Confusion matrix visualization

---

## Results
- Model performs well on major activities
- Some confusion between similar activities

---

## Libraries Used
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- tensorflow / keras

---

## Run Instructions

  Install dependencies:
   pip install numpy pandas matplotlib seaborn scikit-learn tensorflow

---

## Concepts Used
- Time-series classification
- Sliding window technique
- 1D CNN
- Feature scaling

---

## Author
Anurag Yadav
