"""Smart Campus Energy & Space Optimization Environment."""

from __future__ import annotations

import math
import random
import time
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# ── Models ──────────────────────────────────────────────────────────────────

class HVACMode(str, Enum):
    OFF = "off"
    HEATING = "heating"
    COOLING = "cooling"
    AUTO = "auto"

class LightingLevel(str, Enum):
    OFF = "off"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class BuildingZone(BaseModel):
    zone_id: str
    name: str
    current_temp: float
    target_temp: float
    occupancy: int
    max_occupancy: int
    hvac_mode: HVACMode
    lighting: LightingLevel
    energy_consumption_kw: float
    comfort_score: float  # 0-1
    co2_ppm: float
    humidity_percent: float

class WeatherState(BaseModel):
    outdoor_temp_c: float
    solar_irradiance_w_m2: float
    wind_speed_ms: float
    cloud_cover_pct: float
    is_raining: bool

class EnergySource(BaseModel):
    solar_generation_kw: float
    grid_draw_kw: float
    battery_level_pct: float
    battery_capacity_kwh: float
    total_consumption_kw: float
    grid_price_per_kwh: float

class CampusObservation(BaseModel):
    task_id: str
    step: int
    max_steps: int
    time_of_day: float  # 0-24
    day_of_week: int  # 0-6
    buildings: List[BuildingZone]
    weather: WeatherState
    energy: EnergySource
    alerts: List[str]
    total_energy_cost: float
    avg_comfort_score: float
    energy_efficiency_score: float

class ZoneAction(BaseModel):
    zone_id: str
    set_target_temp: Optional[float] = None
    set_hvac_mode: Optional[HVACMode] = None
    set_lighting: Optional[LightingLevel] = None
    redirect_occupants: Optional[int] = None  # +/- people to redirect

class CampusAction(BaseModel):
    zone_actions: List[ZoneAction] = Field(default_factory=list)
    set_battery_mode: Optional[str] = None  # "charge", "discharge", "auto"
    curtail_grid: Optional[bool] = None

class StepResult(BaseModel):
    observation: CampusObservation
    reward: float
    done: bool
    info: Dict[str, Any]

class ResetRequest(BaseModel):
    task_id: Optional[str] = None

class TaskScore(BaseModel):
    task_id: str
    score: float
    details: Dict[str, Any]


# ── Simulation Engine ───────────────────────────────────────────────────────

TASK_CONFIGS = {
    "easy_single_building": {
        "num_buildings": 1,
        "max_steps": 24,
        "zones": [
            {"zone_id": "main_hall", "name": "Main Hall", "max_occ": 200},
        ],
        "weather_var": 0.1,
        "demand_events": False,
        "renewable": False,
    },
    "medium_multi_building": {
        "num_buildings": 3,
        "max_steps": 48,
        "zones": [
            {"zone_id": "lecture_hall_a", "name": "Lecture Hall A", "max_occ": 150},
            {"zone_id": "library", "name": "Library", "max_occ": 100},
            {"zone_id": "lab_building", "name": "Lab Building", "max_occ": 80},
        ],
        "weather_var": 0.4,
        "demand_events": False,
        "renewable": True,
    },
    "hard_full_campus": {
        "num_buildings": 6,
        "max_steps": 72,
        "zones": [
            {"zone_id": "lecture_hall_a", "name": "Lecture Hall A", "max_occ": 150},
            {"zone_id": "lecture_hall_b", "name": "Lecture Hall B", "max_occ": 200},
            {"zone_id": "library", "name": "Library", "max_occ": 100},
            {"zone_id": "lab_building", "name": "Lab Building", "max_occ": 80},
            {"zone_id": "admin_block", "name": "Admin Block", "max_occ": 60},
            {"zone_id": "cafeteria", "name": "Cafeteria", "max_occ": 250},
        ],
        "weather_var": 0.8,
        "demand_events": True,
        "renewable": True,
    },
}


class CampusSimulator:
    def __init__(self, task_id: str, seed: int = 42):
        self.task_id = task_id
        self.rng = random.Random(seed)
        cfg = TASK_CONFIGS[task_id]
        self.max_steps = cfg["max_steps"]
        self.weather_var = cfg["weather_var"]
        self.has_demand_events = cfg["demand_events"]
        self.has_renewable = cfg["renewable"]
        self.step_count = 0
        self.time_of_day = 6.0  # start at 6 AM
        self.day_of_week = 0
        self.total_cost = 0.0
        self.cumulative_comfort = 0.0
        self.cumulative_efficiency = 0.0
        self.demand_event_active = False

        # Initialize zones
        self.zones: Dict[str, Dict[str, Any]] = {}
        for z in cfg["zones"]:
            self.zones[z["zone_id"]] = {
                "name": z["name"],
                "current_temp": 22.0 + self.rng.uniform(-2, 2),
                "target_temp": 22.0,
                "occupancy": 0,
                "max_occupancy": z["max_occ"],
                "hvac_mode": HVACMode.AUTO,
                "lighting": LightingLevel.MEDIUM,
                "energy_kw": 0.0,
                "comfort": 0.8,
                "co2": 400.0,
                "humidity": 45.0,
            }

        # Weather
        self.base_outdoor_temp = 30.0 + self.rng.uniform(-5, 5)
        self.weather = {
            "outdoor_temp": self.base_outdoor_temp,
            "solar": 0.0,
            "wind": 3.0,
            "clouds": 0.3,
            "rain": False,
        }

        # Energy
        self.battery_level = 50.0
        self.battery_cap = 100.0
        self.battery_mode = "auto"
        self.grid_price = 0.12

    def _occupancy_pattern(self, hour: float, zone_id: str, max_occ: int) -> int:
        """Realistic occupancy based on time of day."""
        if hour < 7 or hour > 22:
            return int(max_occ * 0.05)
        if "cafeteria" in zone_id:
            # Meal peaks
            if 11.5 < hour < 13.5 or 18 < hour < 20:
                return int(max_occ * self.rng.uniform(0.7, 1.0))
            return int(max_occ * self.rng.uniform(0.1, 0.3))
        if "lecture" in zone_id:
            # Class hours
            if 8 < hour < 12 or 14 < hour < 17:
                return int(max_occ * self.rng.uniform(0.5, 0.95))
            return int(max_occ * self.rng.uniform(0.05, 0.2))
        if "library" in zone_id:
            if 9 < hour < 21:
                return int(max_occ * self.rng.uniform(0.3, 0.8))
            return int(max_occ * self.rng.uniform(0.05, 0.15))
        if "lab" in zone_id:
            if 9 < hour < 18:
                return int(max_occ * self.rng.uniform(0.4, 0.85))
            return int(max_occ * self.rng.uniform(0.05, 0.15))
        # Default
        if 8 < hour < 18:
            return int(max_occ * self.rng.uniform(0.3, 0.7))
        return int(max_occ * self.rng.uniform(0.05, 0.2))

    def _update_weather(self):
        hour = self.time_of_day
        # Temperature varies with time of day
        diurnal = 5 * math.sin(math.pi * (hour - 6) / 12) if 6 < hour < 18 else -3
        noise = self.rng.gauss(0, self.weather_var * 2)
        self.weather["outdoor_temp"] = self.base_outdoor_temp + diurnal + noise
        # Solar
        if 6 < hour < 18:
            solar_angle = math.sin(math.pi * (hour - 6) / 12)
            cloud_factor = 1 - self.weather["clouds"] * 0.7
            self.weather["solar"] = max(0, 800 * solar_angle * cloud_factor + self.rng.gauss(0, 30))
        else:
            self.weather["solar"] = 0
        # Cloud / rain
        self.weather["clouds"] = max(0, min(1, self.weather["clouds"] + self.rng.gauss(0, 0.05 * self.weather_var)))
        self.weather["rain"] = self.weather["clouds"] > 0.75 and self.rng.random() < 0.3
        self.weather["wind"] = max(0, self.weather["wind"] + self.rng.gauss(0, 0.5))
        # Grid price varies
        if 10 < hour < 14 or 18 < hour < 21:
            self.grid_price = 0.18 + self.rng.uniform(0, 0.08)
        else:
            self.grid_price = 0.08 + self.rng.uniform(0, 0.04)

    def _compute_zone_energy(self, z: Dict[str, Any]) -> float:
        """Compute energy for a zone based on HVAC and lighting."""
        hvac_kw = 0.0
        temp_diff = abs(z["current_temp"] - z["target_temp"])
        if z["hvac_mode"] == HVACMode.COOLING:
            hvac_kw = 2.0 + temp_diff * 1.5 + z["occupancy"] * 0.02
        elif z["hvac_mode"] == HVACMode.HEATING:
            hvac_kw = 1.5 + temp_diff * 1.2 + z["occupancy"] * 0.015
        elif z["hvac_mode"] == HVACMode.AUTO:
            if z["current_temp"] > z["target_temp"] + 1:
                hvac_kw = 1.5 + temp_diff * 1.0 + z["occupancy"] * 0.015
            elif z["current_temp"] < z["target_temp"] - 1:
                hvac_kw = 1.0 + temp_diff * 0.8 + z["occupancy"] * 0.01

        light_kw = {"off": 0, "low": 0.5, "medium": 1.5, "high": 3.0}.get(z["lighting"], 1.5)
        return hvac_kw + light_kw

    def _compute_comfort(self, z: Dict[str, Any]) -> float:
        """Comfort score 0-1 based on temp, CO2, humidity."""
        temp_diff = abs(z["current_temp"] - z["target_temp"])
        temp_score = max(0, 1 - temp_diff / 5)
        co2_score = max(0, 1 - max(0, z["co2"] - 600) / 800)
        hum_score = 1 - abs(z["humidity"] - 50) / 50
        occ_ratio = z["occupancy"] / max(1, z["max_occupancy"])
        crowd_score = 1 if occ_ratio < 0.85 else max(0, 1 - (occ_ratio - 0.85) * 5)
        # Lighting comfort
        if z["occupancy"] > 0 and z["lighting"] == "off":
            light_score = 0.3
        else:
            light_score = 1.0
        return (temp_score * 0.35 + co2_score * 0.2 + hum_score * 0.15 +
                crowd_score * 0.15 + light_score * 0.15)

    def _simulate_temp(self, z: Dict[str, Any]):
        outdoor = self.weather["outdoor_temp"]
        heat_from_people = z["occupancy"] * 0.1
        drift = (outdoor - z["current_temp"]) * 0.03 + heat_from_people * 0.01

        if z["hvac_mode"] == HVACMode.COOLING:
            drift -= 0.5
        elif z["hvac_mode"] == HVACMode.HEATING:
            drift += 0.5
        elif z["hvac_mode"] == HVACMode.AUTO:
            if z["current_temp"] > z["target_temp"] + 0.5:
                drift -= 0.3
            elif z["current_temp"] < z["target_temp"] - 0.5:
                drift += 0.3

        z["current_temp"] += drift + self.rng.gauss(0, 0.1)
        z["co2"] = 400 + z["occupancy"] * 2.5 + self.rng.gauss(0, 10)
        z["humidity"] = 45 + (z["occupancy"] * 0.1) + self.rng.gauss(0, 2)

    def apply_action(self, action: CampusAction):
        for za in action.zone_actions:
            if za.zone_id not in self.zones:
                continue
            z = self.zones[za.zone_id]
            if za.set_target_temp is not None:
                z["target_temp"] = max(16, min(30, za.set_target_temp))
            if za.set_hvac_mode is not None:
                z["hvac_mode"] = za.set_hvac_mode
            if za.set_lighting is not None:
                z["lighting"] = za.set_lighting
            if za.redirect_occupants is not None:
                change = max(-z["occupancy"], min(20, za.redirect_occupants))
                z["occupancy"] = max(0, z["occupancy"] + change)
        if action.set_battery_mode in ("charge", "discharge", "auto"):
            self.battery_mode = action.set_battery_mode

    def tick(self) -> StepResult:
        self.step_count += 1
        self.time_of_day = (self.time_of_day + 1) % 24
        if self.time_of_day == 0:
            self.day_of_week = (self.day_of_week + 1) % 7

        self._update_weather()

        # Demand response events (hard mode)
        if self.has_demand_events and self.rng.random() < 0.08:
            self.demand_event_active = True
        elif self.demand_event_active and self.rng.random() < 0.4:
            self.demand_event_active = False

        total_consumption = 0.0
        alerts = []
        for zid, z in self.zones.items():
            z["occupancy"] = self._occupancy_pattern(self.time_of_day, zid, z["max_occupancy"])
            self._simulate_temp(z)
            z["energy_kw"] = self._compute_zone_energy(z)
            z["comfort"] = self._compute_comfort(z)
            total_consumption += z["energy_kw"]

            if z["current_temp"] > 30:
                alerts.append(f"{z['name']}: Temperature critically high ({z['current_temp']:.1f}°C)")
            if z["co2"] > 1000:
                alerts.append(f"{z['name']}: CO2 levels high ({z['co2']:.0f} ppm)")
            if z["occupancy"] > z["max_occupancy"] * 0.9:
                alerts.append(f"{z['name']}: Near max occupancy")

        # Solar generation
        solar_gen = 0.0
        if self.has_renewable:
            solar_gen = self.weather["solar"] * 0.02  # 2% efficiency panels
        
        # Battery logic
        if self.battery_mode == "charge" and solar_gen > 0:
            charge = min(solar_gen * 0.5, (100 - self.battery_level) * self.battery_cap / 100)
            self.battery_level = min(100, self.battery_level + charge / self.battery_cap * 100)
            solar_gen -= charge
        elif self.battery_mode == "discharge":
            discharge = min(self.battery_level * self.battery_cap / 100 * 0.1, total_consumption * 0.3)
            self.battery_level = max(0, self.battery_level - discharge / self.battery_cap * 100)
            total_consumption -= discharge

        grid_draw = max(0, total_consumption - solar_gen)
        hour_cost = grid_draw * self.grid_price
        if self.demand_event_active:
            hour_cost *= 2.5
            alerts.append("⚡ DEMAND RESPONSE EVENT: Grid prices surging!")

        self.total_cost += hour_cost

        # Build observation
        buildings = []
        comfort_scores = []
        for zid, z in self.zones.items():
            buildings.append(BuildingZone(
                zone_id=zid, name=z["name"],
                current_temp=round(z["current_temp"], 1),
                target_temp=round(z["target_temp"], 1),
                occupancy=z["occupancy"],
                max_occupancy=z["max_occupancy"],
                hvac_mode=z["hvac_mode"],
                lighting=z["lighting"],
                energy_consumption_kw=round(z["energy_kw"], 2),
                comfort_score=round(z["comfort"], 3),
                co2_ppm=round(z["co2"], 1),
                humidity_percent=round(z["humidity"], 1),
            ))
            comfort_scores.append(z["comfort"])

        avg_comfort = sum(comfort_scores) / len(comfort_scores) if comfort_scores else 0
        # Efficiency = how well energy tracks occupancy needs
        total_occ = sum(z["occupancy"] for z in self.zones.values())
        total_max = sum(z["max_occupancy"] for z in self.zones.values())
        occ_ratio = total_occ / max(1, total_max)
        energy_per_person = total_consumption / max(1, total_occ) if total_occ > 0 else total_consumption
        efficiency = max(0, min(1, 1 - (energy_per_person - 0.5) / 5))

        self.cumulative_comfort += avg_comfort
        self.cumulative_efficiency += efficiency

        obs = CampusObservation(
            task_id=self.task_id, step=self.step_count, max_steps=self.max_steps,
            time_of_day=round(self.time_of_day, 1), day_of_week=self.day_of_week,
            buildings=buildings, weather=WeatherState(
                outdoor_temp_c=round(self.weather["outdoor_temp"], 1),
                solar_irradiance_w_m2=round(self.weather["solar"], 1),
                wind_speed_ms=round(self.weather["wind"], 1),
                cloud_cover_pct=round(self.weather["clouds"] * 100, 1),
                is_raining=self.weather["rain"],
            ),
            energy=EnergySource(
                solar_generation_kw=round(solar_gen, 2),
                grid_draw_kw=round(grid_draw, 2),
                battery_level_pct=round(self.battery_level, 1),
                battery_capacity_kwh=self.battery_cap,
                total_consumption_kw=round(total_consumption, 2),
                grid_price_per_kwh=round(self.grid_price, 3),
            ),
            alerts=alerts,
            total_energy_cost=round(self.total_cost, 2),
            avg_comfort_score=round(avg_comfort, 3),
            energy_efficiency_score=round(efficiency, 3),
        )

        # Reward: balance cost, comfort, efficiency
        cost_penalty = -hour_cost * 2
        comfort_reward = avg_comfort * 3
        efficiency_reward = efficiency * 2
        demand_bonus = 1.0 if self.demand_event_active and grid_draw < total_consumption * 0.5 else 0
        reward = comfort_reward + efficiency_reward + cost_penalty + demand_bonus

        done = self.step_count >= self.max_steps

        info = {
            "hour_cost": round(hour_cost, 3),
            "solar_gen": round(solar_gen, 2),
            "grid_draw": round(grid_draw, 2),
            "demand_event": self.demand_event_active,
        }

        return StepResult(observation=obs, reward=round(reward, 4), done=done, info=info)

    def get_final_score(self) -> float:
        """Grader: score 0.0 - 1.0."""
        if self.step_count == 0:
            return 0.0
        avg_comfort = self.cumulative_comfort / self.step_count
        avg_efficiency = self.cumulative_efficiency / self.step_count
        # Baseline cost comparison (naive constant HVAC)
        baseline_cost = self.step_count * 0.5 * len(self.zones)
        cost_ratio = min(1, baseline_cost / max(0.01, self.total_cost))
        score = avg_comfort * 0.4 + avg_efficiency * 0.3 + cost_ratio * 0.3
        return round(max(0, min(1, score)), 4)

    def get_observation(self) -> CampusObservation:
        buildings = []
        comfort_scores = []
        for zid, z in self.zones.items():
            buildings.append(BuildingZone(
                zone_id=zid, name=z["name"],
                current_temp=round(z["current_temp"], 1),
                target_temp=round(z["target_temp"], 1),
                occupancy=z["occupancy"],
                max_occupancy=z["max_occupancy"],
                hvac_mode=z["hvac_mode"],
                lighting=z["lighting"],
                energy_consumption_kw=round(z["energy_kw"], 2),
                comfort_score=round(z["comfort"], 3),
                co2_ppm=round(z["co2"], 1),
                humidity_percent=round(z["humidity"], 1),
            ))
            comfort_scores.append(z["comfort"])
        avg_comfort = sum(comfort_scores) / len(comfort_scores) if comfort_scores else 0
        total_consumption = sum(z["energy_kw"] for z in self.zones.values())
        total_occ = sum(z["occupancy"] for z in self.zones.values())
        energy_per_person = total_consumption / max(1, total_occ) if total_occ > 0 else total_consumption
        efficiency = max(0, min(1, 1 - (energy_per_person - 0.5) / 5))

        return CampusObservation(
            task_id=self.task_id, step=self.step_count, max_steps=self.max_steps,
            time_of_day=round(self.time_of_day, 1), day_of_week=self.day_of_week,
            buildings=buildings,
            weather=WeatherState(
                outdoor_temp_c=round(self.weather["outdoor_temp"], 1),
                solar_irradiance_w_m2=round(self.weather["solar"], 1),
                wind_speed_ms=round(self.weather["wind"], 1),
                cloud_cover_pct=round(self.weather["clouds"] * 100, 1),
                is_raining=self.weather["rain"],
            ),
            energy=EnergySource(
                solar_generation_kw=0, grid_draw_kw=0,
                battery_level_pct=round(self.battery_level, 1),
                battery_capacity_kwh=self.battery_cap,
                total_consumption_kw=round(total_consumption, 2),
                grid_price_per_kwh=round(self.grid_price, 3),
            ),
            alerts=[], total_energy_cost=round(self.total_cost, 2),
            avg_comfort_score=round(avg_comfort, 3),
            energy_efficiency_score=round(efficiency, 3),
        )


# ── FastAPI App ─────────────────────────────────────────────────────────────

app = FastAPI(title="Smart Campus Energy Optimizer")

# Session storage
sessions: Dict[str, CampusSimulator] = {}
current_session_id: Optional[str] = None


@app.post("/reset")
def reset(req: ResetRequest = ResetRequest()):
    global current_session_id
    task_id = req.task_id or "easy_single_building"
    if task_id not in TASK_CONFIGS:
        raise HTTPException(400, f"Unknown task: {task_id}. Available: {list(TASK_CONFIGS.keys())}")
    sid = str(uuid.uuid4())
    sim = CampusSimulator(task_id, seed=int(time.time()) % 10000)
    sessions[sid] = sim
    current_session_id = sid
    obs = sim.get_observation()
    return {"session_id": sid, "observation": obs.model_dump()}


@app.post("/step")
def step(action: CampusAction, session_id: Optional[str] = None):
    sid = session_id or current_session_id
    if not sid or sid not in sessions:
        raise HTTPException(400, "No active session. Call /reset first.")
    sim = sessions[sid]
    if sim.step_count >= sim.max_steps:
        raise HTTPException(400, "Episode already done.")
    sim.apply_action(action)
    result = sim.tick()
    return result.model_dump()


@app.get("/state")
def state(session_id: Optional[str] = None):
    sid = session_id or current_session_id
    if not sid or sid not in sessions:
        raise HTTPException(400, "No active session.")
    sim = sessions[sid]
    obs = sim.get_observation()
    return {"session_id": sid, "observation": obs.model_dump(), "step": sim.step_count, "done": sim.step_count >= sim.max_steps}


@app.post("/grade")
def grade(session_id: Optional[str] = None):
    sid = session_id or current_session_id
    if not sid or sid not in sessions:
        raise HTTPException(400, "No active session.")
    sim = sessions[sid]
    score = sim.get_final_score()
    return TaskScore(task_id=sim.task_id, score=score, details={
        "steps_taken": sim.step_count,
        "total_cost": round(sim.total_cost, 2),
        "avg_comfort": round(sim.cumulative_comfort / max(1, sim.step_count), 3),
        "avg_efficiency": round(sim.cumulative_efficiency / max(1, sim.step_count), 3),
    }).model_dump()


@app.get("/tasks")
def list_tasks():
    return [{"id": k, "max_steps": v["max_steps"], "num_zones": len(v["zones"])} for k, v in TASK_CONFIGS.items()]


# Serve frontend
# Serve frontend safely
import os

# Mount static only if exists
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def index():
    file_path = "static/index.html"
    
    if os.path.exists(file_path):
        return FileResponse(file_path)
    
    return {
        "status": "Backend is running ✅",
        "message": "Frontend not found",
        "expected_path": file_path
    }


def main():
    """ASGI app factory — called by the OpenEnv/uvicorn runner as: app = main()"""
    return app


if __name__ == "__main__":
    import uvicorn, os
    uvicorn.run(app, host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", "7860")))
