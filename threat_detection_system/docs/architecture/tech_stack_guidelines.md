# Tech Stack Guidelines
## Recommended Tech Stack Summary
| Category	| Recommended Technologies |
|-----------|------------------------|
| Language	| Python 3.12+, TypeScript |
| Backend	| FastAPI, Uvicorn, Pydantic, SQLAlchemy |
| Agent Orchestration	| LangGraph |
| Agent Development	| PydanticAI |
| LLM Abstraction	| LiteLLM |
| STT	| faster-whisper, WhisperX |
| Diarization	| pyannote.audio |
| Audio Processing	| librosa, torchaudio, pydub, FFmpeg |
| Voice Activity Detection	| Silero VAD |
| Emotion Recognition	| SpeechBrain, Hugging Face audio models |
| NLP	| spaCy, transformers, sentence-transformers |
| Embeddings	| BGE, multilingual-e5 |
| Database	| PostgreSQL |
| Cache	| Redis |
| Object Storage	| MinIO |
| Vector Store (optional)	| pgvector |
| Configuration	| pydantic-settings, PyYAML |
| Testing	| pytest, hypothesis |
| Dashboard	| React, TypeScript, Material UI |
| Background Jobs	| Celery + Redis |
| Monitoring	| OpenTelemetry, Prometheus, Grafana |
| Documentation	| MkDocs Material, Mermaid |
| Packaging	| Poetry |
| Quality	| Ruff, Black, MyPy |
| Deployment	| Docker, Docker Compose |

## A few architectural suggestions
- Keep agents stateless. Persist state, artifacts, and evaluation results outside the agents so they can be scaled horizontally. 
- Use typed contracts everywhere. Every agent should accept a Pydantic model and return a Pydantic model. This makes orchestration, testing, and future replacement of individual agents much easier. 
- Separate orchestration from intelligence. LangGraph should coordinate agent execution, while each agent focuses on one responsibility (audio analysis, transcript analysis, fraud detection, etc.). 
- Abstract external providers. Define interfaces for LLMs, TTS, STT, and storage so you can switch between open-source and commercial implementations without changing business logic.