## (Model Based) Semantic Reward 使用指南

**请先阅读FIRE.md**

将本Folder里的rewardbench.py替换jiachen的verl/utils/reward_score/rewardbench.py文件即可。
本实现使用vllm部署qwen3-reranker-0.6B或者qwen3-embedding-0.6B，来为训练提供api接口。

因此，请先
```
bash deploy.sh
```

再：
```
bash recipe/dapo/run_rewardbench.sh
```

默认使用FSDP Backend：
```
bash conv.sh
```
将权重转化为Huggingface格式。


关于数据，可以check data.py通过原版数据构造verl格式的数据，然后简单写个脚本把对应增强数据的增强答案，加到原版数据的enhanced_answer这个key里(记得check system prompt，之前jiachen构造的system prompt不一样)

示例数据：4agent.parquet
