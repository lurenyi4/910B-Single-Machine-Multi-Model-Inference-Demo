910B使用官方的MindIE框架推理时，无法实现单卡多个模型的推理，为尝试解决这个问题，借用了llamafactory的chat方式进行模型推理并使用fastapi构建成API。
使用方式：
1.下载llamafactory-NPU的代码，并安装依赖。具体可参考：https://github.com/hiyouga/LLaMA-Factory/blob/main/README_zh.md
2.将chat_service.py放入LLaMA-Factory/src/llamafactory/api/目录下.
3.在LLaMA-Factory/src/llamafactory/目录下执行代码
example:python api/chat_service.py   --model_name_or_path /Qwen/Qwen2.5-7B-Instruct   --infer_backend huggingface   --template qwen
