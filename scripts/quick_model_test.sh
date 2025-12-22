#!/bin/bash
# Quick test of new models via Ollama

echo "Testing DeepSeek-R1 8B (reasoning model)..."
echo ""

curl -s http://YOUR_GPU_HOST:11434/api/generate -d '{
  "model": "deepseek-r1:8b",
  "prompt": "Explain this 65816 assembly in one sentence: LDA #$80 STA $2100 (hide screen during setup)",
  "stream": false,
  "options": {
    "temperature": 0.7,
    "num_predict": 200
  }
}' | python3 -c "import sys, json; print(json.load(sys.stdin)['response'])"

echo ""
echo "Testing Qwen3 14B (latest version)..."
echo ""

curl -s http://YOUR_GPU_HOST:11434/api/generate -d '{
  "model": "qwen3:14b",
  "prompt": "Explain this 65816 assembly in one sentence: LDA #$80 STA $2100 (hide screen during setup)",
  "stream": false,
  "options": {
    "temperature": 0.7,
    "num_predict": 200
  }
}' | python3 -c "import sys, json; print(json.load(sys.stdin)['response'])"
