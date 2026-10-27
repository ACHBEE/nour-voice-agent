# STT benchmark (8 phrases, one speaker)

| Model | WER Darja | WER Darja+FR | WER French | WER English | WER all | CER all | p50 latency (s) | errors |
|---|---|---|---|---|---|---|---|---|
| speechmatics/linden-1:ar | 0.67 | 0.75 | 1.00 | 1.00 | 0.78 | 0.66 | 0.85 | 1 |
| openai/gpt-live-transcribe:ar | 0.88 | 0.94 | 0.17 | 0.71 | 0.78 | 0.57 | 1.50 | 0 |
| google/gemini-3.5-transcribe-live:ar | 0.85 | 0.94 | 0.17 | 1.00 | 0.81 | 0.66 | 2.13 | 0 |
| assemblyai/universal-3-6-pro:ar | 0.90 | 0.94 | 0.33 | 1.00 | 0.85 | 0.59 | 1.00 | 0 |
| deepgram/nova-3:multi | 1.00 | 0.94 | 0.33 | 0.71 | 0.87 | 0.74 | 0.57 | 0 |
| deepgram/nova-3:ar-DZ | 0.80 | 0.94 | 1.00 | 1.00 | 0.88 | 0.74 | 0.56 | 0 |
| deepgram/nova-3:ar-DZ +keyterms | 0.85 | 0.94 | 1.00 | 1.00 | 0.91 | 0.72 | 0.66 | 0 |
