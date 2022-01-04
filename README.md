## 基于浏览记录的个性化节点推荐算法
### 实验环境
- python=3.6.13
- tensorflow==1.10

### 目录结构

```
├─ gcn-master
│  ├─ experiment
│  │  ├─ cora_rank(保存各类算法在某数据集上的排序结果)
│  │  ├─ ...
│  │  ├─ draw(画结果图的py文件)
│  │  ├─ exp.py(对比各类排序算法的排序结果)
│  │  └─ run.py(加载预测模型并进行节点排序，将结果保存)
│  │
│  ├─ gcn
│  │  ├─ data(数据集)
│  │  ├─ ...
│  │  └─ train.py(加载数据集训练模型并保存模型)
│  │  
│  model
│  │  ├─ ...
│  │  └─ cora(在某数据集上的模型)
│  │ 
│  └─exp_result.docx(实验结果) 
```
- 首先运行`gcn/train.py`，训练模型并保存至`model`文件夹。
- 运行`experiment/run.py`加载模型进行节点排序并保存排序结果。
- 运行`experiment/exp.py`对比算法的排序结果



