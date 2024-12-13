# Bridging Resource Gaps in Cross-Lingual Sentiment Analysis  
**Adaptive Self-Alignment with Data Augmentation and Transfer Learning**  

This repository contains the code and resources for the paper:  
**"Bridging Resource Gaps in Cross-Lingual Sentiment Analysis: Adaptive Self-Alignment with Data Augmentation and Transfer Learning"**  

## Abstract  
Cross-lingual sentiment analysis is essential for accurately interpreting emotions across diverse linguistic contexts, yet significant challenges persist due to uneven performance in low- and medium-resource languages. This study introduces an adaptive self-alignment technique for large language models, integrating innovative data augmentation and transfer learning strategies to address resource disparities. Extensive experiments on 11 languages demonstrate that our approach consistently outperforms state-of-the-art methods, achieving an average improvement of 7.35 F1 points. Notably, the method excels in medium-resource languages, bridging performance gaps with high-resource counterparts. With robust domain adaptation and practical industry adoption, this research sets a new benchmark for multilingual sentiment analysis, paving the way for more inclusive and equitable natural language processing applications.


---

## Repository Structure  
```
/root  
│-- data/                       # Datasets used for the experiments
│   │-- raw/split_data.py        # Constructs training and testing datasets
│   │-- semantic_similarity_search.py        # Semantic similarity search module
│   │-- raw/sentiment_instruction_generator.py        # Sentiment instruction generator
│-- src/                        # Source code for the framework
│-- examples/                       # Project configuration files
│   │-- train_lora/llama3_lora_sft.yaml        # LoRA fine-tuning configuration
│   │-- merge_lora/llama3_lora_sft.yaml        # LoRA merging configuration
│   │-- inference/llama3_lora_sft.yaml        # Inference configuration
│-- models/                     # Pretrained models and checkpoints  
│-- results/                    # Output results and metrics
│-- test.py                    # Testing script 
│-- README.md                   # Project documentation  
```


---

## Requirements  
The project requires the following dependencies:  
- Python >= 3.10
- Transformers >= 4.41.2   

Install the dependencies using:  
```bash
pip install -r requirements.txt
```

---


## Usage  

### 1. Data Preparation  
Place your datasets in the `data/` directory. Preprocess the data using:  
```bash
python data/raw/split_data.py
python data/semantic_similarity_search.py 
python data/raw/sentiment_instruction_generator.py
```

### 2. Model Training  
Train the cross-lingual sentiment analysis model:  
```bash
llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml
```

### 3. Model Merging  
Merging for the Llama3-8B-Instruct model:  
```bash
llamafactory-cli export examples/merge_lora/llama3_lora_sft.yaml
```

### 4. Evaluation  
Evaluate the model on test data:  
```bash
python test.py
```

---

## Results  
Our proposed framework demonstrates consistent improvements over traditional approaches:  

| Dataset         | Baseline Accuracy | Our Method Accuracy |  
|-----------------|-------------------|---------------------|  
| MultiEmo (PL)   | 72.1%             | 81.5%               |  
| Arabic Tweets   | 68.4%             | 76.8%               |  

---



## Contact  
For questions, please contact:  
- Email: wangyawen@mail.tsinghua.edu.cn 

---

## License  
This project is licensed under the MIT License.  

