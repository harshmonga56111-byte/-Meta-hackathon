---
title: Digital Twin Env
emoji: 🏛️
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# 🏛️ Smart Campus Energy & Space Optimization Environment

An OpenEnv-compliant reinforcement learning environment where AI agents optimize energy consumption, HVAC systems, lighting, and occupant comfort across a simulated university campus.

## Problem Statement

Modern campuses waste energy through non-adaptive HVAC, poor space utilization, and static lighting policies. This environment simulates realistic campus dynamics—occupancy patterns, weather, solar generation, grid pricing, and demand response events—so agents can learn optimal building management strategies.

## Action Space

The agent controls the campus infrastructure via a structured JSON/Pydantic action payload:

```python
CampusAction:
  zone_actions: List[ZoneAction]  # Per-building controls
    - zone_id: str
    - set_target_temp: float (16-30°C)
    - set_hvac_mode: "off" | "heating" | "cooling" | "auto"
    - set_lighting: "off" | "low" | "medium" | "high"
    - redirect_occupants: int (+/- people)
  set_battery_mode: "charge" | "discharge" | "auto"
Observation SpaceAt each step, the environment returns a detailed snapshot of the campus state:Zone Metrics: Per-zone temperature, occupancy, HVAC state, lighting, energy draw, comfort score, CO₂, humidity.Weather Data: Outdoor temp, solar irradiance, wind, clouds, rain.Energy Metrics: Solar generation, grid draw, battery level, current grid price.Global Metrics: Alerts, cumulative energy usage, and current step count.Tasks & Expected DifficultyThe environment features 3 deterministic tasks with programmatic graders to test the agent's scalability and adaptability:Task IDZonesStepsDifficultyDescriptioneasy_single_building124EasyStable weather, single zone management.medium_multi_building348MediumWeather variation, solar generation, and battery management.hard_full_campus672HardDemand response events, 6 interconnected buildings, full renewable integration.Reward Function & Grading LogicThe reward function provides a continuous signal over the full trajectory, preventing sparse reward issues.Per-step reward combining: * Comfort score (weighted 0.4)Energy efficiency (0.3)Cost penalty (negative weight)Demand response bonusFinal Grader Score (0.0 – 1.0):Final score = (0.4 × avg_comfort) + (0.3 × avg_efficiency) + (0.3 × cost_savings_vs_baseline)Setup and Usage InstructionsRunning Locally with PythonBashpip install fastapi uvicorn pydantic openai requests
uvicorn server.app:app --port 7860
Running Locally with Docker (Recommended)Bashdocker build -t smart-campus-env .
docker run -p 7860:7860 smart-campus-env
Running Baseline InferenceTo run the baseline evaluation script using an OpenAI-compatible LLM:Bashexport API_BASE_URL=[https://router.huggingface.co/v1](https://router.huggingface.co/v1)
export MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
export HF_TOKEN=your_token
export ENV_URL=http://localhost:7860

python inference.py
Baseline ScoresExpected baseline scores using standard LLM agents (e.g., Llama-3.1-8B-Instruct):TaskExpected Scoreeasy_single_building~0.65medium_multi_building~0.55hard_full_campus~0.42