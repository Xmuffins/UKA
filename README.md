# UKA Dialogue Evaluation

This repository contains two automated dialogue-evaluation flows:

- Chinese Sentient-Eval style flow: `code/test_sentient_eval.py`
- English ESConv flow: `code/test_extes.py`

The profile and benchmark data are stored under `code/profile/`.

## Setup

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

The code expects OpenAI-compatible chat/completions endpoints for both the policy model and the simulator model.

Common environment variables:

```bash
export POLICY_API_KEY="your-policy-api-key-or-EMPTY-for-a-local-server"
export LOCAL_LLM_CHAT_URL="http://0.0.0.0:8000/v1/chat/completions"
export LOCAL_LLM_COMPLETION_URL="http://0.0.0.0:8000/v1/completions"
export REMOTE_LLM_CHAT_URL="http://0.0.0.0:8000/v1/chat/completions"

export UKA_KB_ROOT="./code/kb"
export UKA_EMBEDDING_MODEL="sentence-transformers/embeddinggemma-300m"
```

Simulator environment variables:

```bash
export SIMULATOR_API_KEY="your-simulator-api-key"
export SIMULATOR_BASE_URL="https://your-openai-compatible-endpoint/v1"
export SIMULATOR_MODEL="your-simulator-model"
```

You can also override the simulator per flow:

```bash
export CHINESE_SIMULATOR_API_KEY="your-key"
export CHINESE_SIMULATOR_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
export CHINESE_SIMULATOR_MODEL="deepseek-v3"

export EXTES_SIMULATOR_API_KEY="your-key"
export EXTES_SIMULATOR_BASE_URL="https://your-openai-compatible-endpoint/v1"
export EXTES_SIMULATOR_MODEL="gpt-4o"
```

## Run The Chinese Flow

From the repository root:

```bash
cd code
python test_sentient_eval.py <policy_model> <training|testing>
```

Example:

```bash
python test_sentient_eval.py qwen-32b testing
```

This flow loads `code/profile/simulator_profile_withfirsttalk.jsonl`, initializes the Chinese simulator, and writes testing results to:

```text
code/<policy_model>-uka-normal.jsonl
```

In `training` mode, the script updates the Chroma knowledge base but does not append session results to the output JSONL.

## Run The English Flow

From the repository root:

```bash
cd code
python test_extes.py <policy_model> <training|testing> <uka|principle>
```

Examples:

```bash
python test_extes.py qwen-32b testing uka
python test_extes.py qwen-32b training principle
```

This flow currently loads ESConv data:

- `code/profile/ESConv/train.jsonl` for `training`
- `code/profile/ESConv/test.jsonl` for `testing`

Testing results are written to:

```text
code/<policy_model>-<mode>-rebuttal-esconv.jsonl
```

The `uka` mode uses user modeling plus knowledge retrieval. The `principle` mode runs a critic-free retry loop during training and stores reusable dialogue principles in the knowledge base.

## Analyze Results

After a testing run:

```bash
cd code
python analyze_score.py <result_file.jsonl>
```

Example:

```bash
python analyze_score.py qwen-32b-uka-esconv.jsonl
```
