# **Protest Dynamics in the Israel-Palestine Conflict**  

This repository contains code, data processing scripts, and visualization tools for analyzing global protest activity following the events of October 7, using data from the **Armed Conflict Location & Event Data Project (ACLED)**. The study explores the correlation between conflict intensity and public mobilization, leveraging **natural language processing (NLP)** techniques for sentiment classification.  


## **Usage**  

### **1. Obtain Data Access**  
The dataset used in this study comes from **ACLED** and **cannot be publicly shared** due to licensing restrictions. To run the analysis, request access from [ACLED](https://acleddata.com) and download the relevant datasets.  

### **2. Api keys**
In access_keys.toml update your OpenAI API key.  

### **3. Prepare Data**  
Once access is granted, place the dataset in the `data/` directory.  

### **4. Run Primarily Annotations (optional)**  
If you want to new classification model, run `additional_code/zero_shot_classification.py` with *path* of your subset data. 

### **5. Train Classifyer (optional)**
Execute `additional_code/fine_tune_roberta.py` with *path_to_annotated_data* of your Run Primarily Annotations data.

### **6. Related Classification**
Execute `additional_code/fine_tuned_roberta_classify.py`. Due to model size download from [here](https://acleddata.com). for costume classification model change the model_dir to your model path.

### **7. Annotations**
Execute `additional_code/zero_shot_classification.py` only on your relevent dataset.

### **8. Data Exploration**
Explore the data using the `main.ipynb`.





