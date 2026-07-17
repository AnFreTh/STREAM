"""
LLM-as-a-Judge evaluation for topic models.

Three evaluation tasks:
1. Intruder Detection: insert a word from another topic, ask the LLM to identify it
2. Topic Rating:       ask the LLM to rate coherence on a 1-5 scale (+ label)
3. Word Fit:           ask the LLM to score each top word as fitting the topic (0/1)

Workflow:
  1. generate_prompts.py    -> reads topics from S3, generates JSONL prompt files
  2. submit_bedrock.py      -> submits JSONL to Bedrock Batch Inference (chunked)
  3. parse_results.py       -> parses Bedrock output, computes scores, saves to S3
"""
