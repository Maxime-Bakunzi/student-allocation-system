
---

# AI-Powered Student Subject Allocation System (For Rwanda's Secondary Schools)

### **Project Overview**
This project aims to develop a machine learning model that automatically allocates secondary school students to appropriate advanced course combinations based on their academic performance. While the objective of the project was initially aligned with Rwanda’s education system, the publicly available dataset used for this project is closely related, though not from Rwanda. The dataset was modified by engineering a new feature called **Adv_course_combination**, which categorizes students into specific combinations based on their performance in subjects and career aspirations.

### **Objective**
1. **Model 1**: Implement a simple neural network without any optimization techniques.
2. **Model 2**: Implement a neural network with at least three optimization techniques, including L1/L2 regularization and different optimizers.
3. **Evaluate**: Compare the performance of both models in terms of error metrics (confusion matrix, precision, F1 score, specificity).

### **Dataset Used**
Though the original intent was to use data specific to Rwanda, a similar dataset on student performance was acquired, which includes grades in subjects such as Mathematics, Physics, Chemistry, Biology, History, English, and Geography. The feature **Adv_course_combination** was engineered based on student performance and career aspirations, and it categorizes students into the following combinations:
- **PCB (Physics, Chemistry, Biology)** for aspiring doctors.
- **PCM (Physics, Chemistry, Mathematics)** for aspiring engineers or game developers.
- **MPG (Mathematics, Physics, Geography)** for aspiring bankers, accountants, stock investors, or engineers.
- **HEG (History, English, Geography)** for aspiring teachers or business owners.

If a student did not achieve at least 50% in core subjects like Mathematics or Physics, they were assigned the HEG combination, regardless of their aspirations.

### **Feature Engineering**
The `Adv_course_combination` feature was engineered using logical rules to classify students based on subject scores and aspirations. This feature became the target for classification in the machine learning model.

### **Model Implementations**
1. **Model 1: Simple Neural Network (Vanilla Model)**
    - A basic feedforward neural network was implemented without any optimization techniques.
    - The model consisted of two hidden layers and used the Adam optimizer for training.

2. **Model 2: Neural Network with Optimization Techniques**
    - Applied **L1 Regularization** with both Adam and RMSProp optimizers.
    - Applied **L2 Regularization** with both Adam and RMSProp optimizers.
    - Used **Early Stopping** and **Learning Rate Schedulers** to optimize model performance and avoid overfitting.

### **Model Evaluation Metrics**
The performance of the models was evaluated using the following metrics:
- **Confusion Matrix**: Shows the model's ability to correctly classify each combination.
- **F1 Score**: A balance between precision and recall, useful in this multi-class classification problem.
- **Precision**: The accuracy of the positive predictions.
- **Specificity**: The ability of the model to correctly identify true negatives.

### **Results and Comparison**

| Model                                         | Optimizer | Regularization  | LR Scheduler | Weighted Precision | F1 Score | Test Accuracy |
|-----------------------------------------------|-----------|-----------------|--------------|--------------------|----------|---------------|
| **Simple Neural Network (Baseline)**          | Adam      | None            | No           | 0.46               | 0.4501   | 45.25%        |
| **L1 Regularization with Adam**               | Adam      | L1 (0.01)       | No           | 0.50               | 0.4612   | 49.50%        |
| **L1 Regularization with RMSProp**            | RMSProp   | L1 (0.01)       | No           | 0.34               | 0.3674   | 44.50%        |
| **L2 Regularization with Adam**               | Adam      | L2 (0.01)       | No           | 0.55               | 0.5078   | 52.50%        |
| **L2 Regularization with Adam & LR Scheduler**| Adam      | L2 (0.01)       | Yes          | 0.44               | 0.4150   | 44.25%        |
| **L2 Regularization with RMSProp**            | RMSProp   | L2 (0.01)       | No           | 0.34               | 0.3796   | 46.50%        |

**Notes:**

- **Weighted Precision** is taken from the "weighted avg" precision in the classification reports.
- **F1 Score** is the overall weighted average F1 score you provided.
- **Test Accuracy** is the overall accuracy percentage on the test dataset.
- **LR Scheduler** indicates whether a Learning Rate Scheduler was used during training.

### **Observations:**

- **Best Performing Model:** The **L2 Regularization with Adam optimizer** (without learning rate scheduler) achieved the highest test accuracy (52.50%) and F1 score (0.5078). It also had the highest weighted precision (0.55), indicating better overall performance compared to other models.
  
- **Effect of Regularization:**
  - **L2 Regularization** generally outperformed **L1 Regularization** in this context, especially when used with the Adam optimizer.
  - Regularization helped in slightly improving the model's ability to generalize, as seen by the incremental increases in accuracy and F1 scores.

- **Impact of Optimizers:**
  - The **Adam optimizer** consistently performed better than **RMSProp** across both L1 and L2 regularizations.
  - Models using RMSProp had lower weighted precision and F1 scores, indicating that Adam might be more suitable for this dataset.

- **Learning Rate Scheduler:**
  - Introducing a **Learning Rate Scheduler** with L2 Regularization and Adam did not improve the model's performance in this case. In fact, it slightly decreased both the accuracy and F1 score compared to using L2 Regularization with Adam alone.
  - This suggests that while learning rate schedulers can be beneficial, they may not always lead to performance gains and should be carefully tuned.

### **Conclusion:**

Applying optimization techniques led to **slight improvements** in model performance over the baseline model. The best results were achieved with **L2 Regularization using the Adam optimizer**, without a learning rate scheduler. This indicates that regularization can help enhance model generalization, but the choice of optimizer and careful tuning of hyperparameters are crucial for achieving better performance for different datasets.

### **How to Run the Project**
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Maxime-Bakunzi/student-allocation-system.git
   cd student-allocation-system
   ```
2. **Install Dependencies**:
   Install the required Python libraries using:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Notebook**:
   Open the `notebook.ipynb` file using Jupyter Notebook or any compatible environment:
   ```bash
   jupyter notebook notebook.ipynb
   ```
4. **View the Saved Models**:
   The trained models can be found in the `saved_models` directory. These models can be loaded and evaluated on new datasets.

### **Directory Structure**
```
student-allocation-system/
├── notebook.ipynb               # Project code and analysis
├── student-scores.csv           # Dataset
├── requirements.txt             # Required packages
├── saved_models/                # Directory containing saved models
│   ├── best_model.h5
│   ├── l1_with_adam.h5
│   ├── l1_with_rmsprop.h5
│   ├── l2_with_adam.h5
│   ├── l2_with_adam_and_lr_scheduler.h5
│   ├── l2_with_rmsprop.h5
│   ├── simple_model.h5
└── README.md                    # Project overview and instructions
```

---