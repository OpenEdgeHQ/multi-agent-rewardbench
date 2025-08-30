## verl 使用指南

### 运行环境
如果使用的是root@f4f1721b6352这个机器，那可以直接
```
conda activate verl
```
如果是其他机器，请参考https://verl.readthedocs.io/en/latest/start/install.html进行环境安装。

### 数据
我们优化后的数据已经上传到`data/Rewardbench_Optimizeddata_answer.parquet`, 可以直接使用

### reward 计算
verl中会根据data_source进行reward function的选用，见`verl/utils/reward_score/__init__.py`.
现在的实现中当data_source为`Rewardbench`则会调用`verl/utils/reward_score/rewardbench.py`中的算分逻辑：
```
def compute_score_ours(model_output: str, ground_truth: str):
    """
    Aggregate function that combines all reward functions with equal weights.
    Returns the average of all individual rewards.
    """
    # Initialize reward functions
    cosine_scaled_reward = get_cosine_scaled_reward()
    repetition_penalty_reward = get_repetition_penalty_reward()
    
    # Calculate individual rewards
    acc_reward = accuracy_reward(model_output, ground_truth)
    format_reward_val = format_reward(model_output)
    reasoning_reward = reasoning_steps_reward(model_output)
    cosine_reward = cosine_scaled_reward(model_output, ground_truth, acc_reward)
    repetition_reward = repetition_penalty_reward(model_output)
    
    # Calculate equal-weighted average
    total_reward = (acc_reward + format_reward_val + reasoning_reward + 
                   cosine_reward + repetition_reward) / 5.0
    
    return total_reward 
```

### 运行
> 现在是基于recipe/dapo中的逻辑进行运行，因此运行脚本也放在这下面

一个可以跑的脚本是`recipe/dapo/run_rewardbench.sh`.

```
bash recipe/dapo/run_rewardbench.sh
```

