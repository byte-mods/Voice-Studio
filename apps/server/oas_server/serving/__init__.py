"""Inference serving for registered ModelVersions.

Each modality (ASR / LLM / TTS) gets a lightweight wrapper plus an LRU model
cache so repeated requests don't re-load the model. Heavy deps are imported
lazily.
"""
