# CTRL for Wind Power Forecasting

### CTRL: Collaborative Temporal Representation Learning for Wind Power Forecasting [[Paper](https://dl.acm.org/doi/abs/10.1145/3711129.3711336)]

## News

🎉🎉🎉 **CTRL** model has been integrated
into [pyFAST: Flexible, Advanced Framework for Multi-source and Sparse Time Series Analysis in PyTorch](https://github.com/freepose/pyFAST).
The implementation details can be found [here](https://github.com/freepose/pyFAST/blob/main/fast/model/mts/coat.py).  
We sincerely thank [Zhijin Wang](https://github.com/freepose) and colleagues for their valuable support during the
integration process.

## Abstract

Accurate wind power forecasting is crucial for grid stability and renewable energy integration, but existing methods struggle to capture complex temporal dependencies in wind data. This paper introduces Collaborative Temporal Representation Learning (CTRL), a novel deep learning model that leverages collaborative representation learning to enhance forecasting accuracy and robustness. CTRL integrates Reversible Instance Normalization (RevIN), RNN-based hidden state learning, and a specialized collaborative representation unit to capture multi-directional temporal dynamics across different time scales and variables. Experimental results on two real-world wind power datasets demonstrate that CTRL significantly outperforms 20 existing methods, including state-of-the-art deep learning approaches, achieving up to 9.67% and 10.42% improvement in forecasting accuracy, respectively. These findings highlight the potential of collaborative representation learning for advancing wind power forecasting and facilitating the effective integration of renewable energy resources.

## Model Architecture

![Model Architecture](model_architecture.png)

## Requirements

- Python 3.10+
- PyTorch 2.3.1+

## Dataset

We utilize two publicly available datasets in this study.

## Experimental Setup

We divide the data into training and testing sets using an 80/20 split.

## Contact

If you have any questions or suggestions, please feel free to contact me at [yuehu.xm@gmail.com](yuehu.xm@gmail.com).

## Citation

If you find this work useful in your research, please use the following citation formats:

**BibTeX:**

```bibtex
@inproceedings{pro/eitce2024/Hu,
	author     = {Yue Hu and Senzhen Wu and Yu Chen and Xinhao He and Zihao Xie and Zhijin Wang and Xiufeng Liu and Yonggang Fu},
	title      = {{CTRL}: Collaborative Temporal Representation Learning for Day-ahead Wind Power Forecasting},
	booktitle  = {Proceedings of the 8th International Conference on Electronic Information Technology and Computer Engineering},
    doi        = {10.1145/3711129.3711336},
	publisher  = {{ACM}},
	year       = {2024},
	month      = {10},
	address    = {Haikou, China},
}
```

**APA/Plain Text:**

Yue Hu and Senzhen Wu and Yu Chen and Xinhao He and Zihao Xie and Zhijin Wang and Xiufeng Liu and Yonggang Fu. CTRL: Collaborative Temporal Representation Learning for Wind Power Forecasting. ACM EITCE 2024, https://doi.org/10.1145/3711129.3711336
