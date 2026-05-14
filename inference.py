"""
Inference Script — Smart Campus Energy Optimizer
Emits required [START]/[STEP]/[END] structured output to stdout.
"""

import os
import sys
import json
import time
import requests
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY      = os.getenv("API_KEY") or os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY", "placeholder")
MODEL_NAME   = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
ENV_URL      = os.getenv("ENV_URL", "http://localhost:7860")


# ── Mandatory structured logging ────────────────────────────────────────────

def log(tag: str, **kwargs) -> None:
    """Print [TAG] key=value ... to stdout, flushed immediately."""
    parts = [f"[{tag}]"]
    for k, v in kwargs.items():
        # Stringify values so the log line is always parseable
        if isinstance(v, float):
            parts.append(f"{k}={v:.6f}")
        else:
            parts.append(f"{k}={v}")
    print(" ".join(parts), flush=True)


# ── LLM system prompt ────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an AI agent optimizing energy usage and occupant comfort on a smart campus.

You receive observations about buildings (temperature, occupancy, HVAC, lighting, energy) and weather.
You must output a JSON action to optimize the campus. The action format is:

{
  "zone_actions": [
    {
      "zone_id": "<zone_id>",
      "set_target_temp": <float 18-26>,
      "set_hvac_mode": "<off|heating|cooling|auto>",
      "set_lighting": "<off|low|medium|high>"
    }
  ],
  "set_battery_mode": "<charge|discharge|auto>"
}

Goals (in priority order):
1. Keep comfort scores above 0.7 for all occupied zones
2. Minimize energy cost — use solar/battery during peak grid prices
3. Turn off or reduce HVAC/lighting in unoccupied zones
4. During demand events, reduce grid draw aggressively

Respond with ONLY valid JSON. No explanation."""


def _get_llm_action(client: OpenAI, obs: dict) -> dict:
    obs_summary = json.dumps({
        "step":        obs["step"],
        "time_of_day": obs["time_of_day"],
        "weather":     obs["weather"],
        "energy":      obs["energy"],
        "alerts":      obs["alerts"],
        "buildings": [
            {
                "zone_id":   b["zone_id"],
                "temp":      b["current_temp"],
                "target":    b["target_temp"],
                "occupancy": f"{b['occupancy']}/{b['max_occupancy']}",
                "hvac":      b["hvac_mode"],
                "lighting":  b["lighting"],
                "comfort":   b["comfort_score"],
            }
            for b in obs["buildings"]
        ],
    }, indent=2)

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": f"Campus state:\n```json\n{obs_summary}\n```\nProvide your action as JSON."},
            ],
            temperature=0.2,
            max_tokens=500,
        )
        text = (completion.choices[0].message.content or "{}").strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        return json.loads(text)
    except Exception as e:
        print(f"[WARN] LLM error: {e}", file=sys.stderr, flush=True)
        return {"zone_actions": [], "set_battery_mode": "auto"}


def run_task(client: OpenAI, task_id: str) -> float:
    # ── [START] ──────────────────────────────────────────────────────────────
    log("START", task=task_id, model=MODEL_NAME)

    # Reset environment
    try:
        resp = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id}, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        log("END", task=task_id, score=0.001, steps=0, error=str(e))
        return 0.001

    session_id   = data["session_id"]
    obs          = data["observation"]
    done         = False
    total_reward = 0.0
    step         = 0

    while not done:
        step += 1

        # Get action from LLM
        action = _get_llm_action(client, obs)

        # Step the environment
        try:
            step_resp = requests.post(
                f"{ENV_URL}/step",
                json=action,
                params={"session_id": session_id},
                timeout=30,
            )
            step_resp.raise_for_status()
            result = step_resp.json()
        except Exception as e:
            print(f"[WARN] Step error: {e}", file=sys.stderr, flush=True)
            break

        obs          = result["observation"]
        reward       = float(result["reward"])
        done         = result["done"]
        total_reward += reward

        # ── [STEP] ───────────────────────────────────────────────────────────
        log("STEP",
            task=task_id,
            step=step,
            reward=reward,
            total_reward=total_reward,
            comfort=obs["avg_comfort_score"],
            efficiency=obs["energy_efficiency_score"],
            cost=obs["total_energy_cost"],
            done=done,
        )

    # Grade
    try:
        grade_resp = requests.post(
            f"{ENV_URL}/grade",
            params={"session_id": session_id},
            timeout=30,
        )
        grade_resp.raise_for_status()
        score_data = grade_resp.json()
        raw_score  = float(score_data.get("score", 0.0))
    except Exception as e:
        print(f"[WARN] Grade error: {e}", file=sys.stderr, flush=True)
        raw_score = 0.0

    # Clamp score to strictly (0, 1) — validator rejects 0.0 and 1.0 exactly
    score = max(0.001, min(0.999, raw_score))

    # ── [END] ────────────────────────────────────────────────────────────────
    log("END", task=task_id, score=score, steps=step, total_reward=total_reward)

    return score


def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    tasks = [
        "easy_single_building",
        "medium_multi_building",
        "hard_full_campus",
    ]

    results = {}
    for task_id in tasks:
        try:
            score = run_task(client, task_id)
        except Exception as e:
            print(f"[WARN] Task {task_id} failed: {e}", file=sys.stderr, flush=True)
            log("END", task=task_id, score=0.001, steps=0, error=str(e))
            score = 0.001
        results[task_id] = score
        time.sleep(1)

    avg = sum(results.values()) / len(results)
    log("SUMMARY", avg_score=avg, tasks=len(results))


if __name__ == "__main__":
    main()
