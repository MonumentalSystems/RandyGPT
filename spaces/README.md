---
title: randyGPT
emoji: 📖
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# randyGPT — OpenAI-compatible API

A GPT trained from scratch in Rust on 114 Project Gutenberg books.
Model weights load from [MonumentalSystems/randygpt-s](https://huggingface.co/MonumentalSystems/randygpt-s).

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/v1/models` | List available models |
| POST | `/v1/chat/completions` | Generate text (OpenAI-compatible) |

## Usage

```bash
curl https://monumentalsystems-randygpt-space.hf.space/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "randygpt-s",
    "messages": [{"role": "user", "content": "Once upon a time"}],
    "max_tokens": 200,
    "temperature": 0.8
  }'
```

### OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://monumentalsystems-randygpt-space.hf.space/v1",
    api_key="none",
)

response = client.chat.completions.create(
    model="randygpt-s",
    messages=[{"role": "user", "content": "Once upon a time"}],
    max_tokens=200,
)
print(response.choices[0].message.content)
```
