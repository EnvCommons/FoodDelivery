"""
Food Delivery Dispatch Optimization Environment for OpenReward.

A hyper-realistic food delivery platform simulation where an AI agent
manages courier-order matching, order batching, surge pricing, and
fleet repositioning across a procedurally generated city.
"""

import json
from typing import Any, List

from pydantic import BaseModel

from openreward.environments import Environment, JSONObject, ToolOutput, tool, TextBlock
from simulation import (
    Simulation,
    ScenarioConfig,
    WeatherConfig,
    DispatchActions,
    GetInfoParams,
)


# ── Task Spec ──────────────────────────────────────────────────────────────────

class TaskSpec(BaseModel):
    id: str


# ── Scenario Registry ─────────────────────────────────────────────────────────

SCENARIOS: dict[str, dict[str, Any]] = {
    "weekday_calm": {
        "demand_multiplier": 1.0,
        "weather": WeatherConfig(name="clear"),
        "num_couriers": 20,
    },
    "weekday_busy": {
        "demand_multiplier": 1.3,
        "weather": WeatherConfig(name="clear"),
        "num_couriers": 20,
    },
    "rainy_evening": {
        "demand_multiplier": 1.3,
        "weather": WeatherConfig(name="rain", demand_boost=0.25, travel_time_boost=0.30),
        "num_couriers": 18,
    },
    "weekend_rush": {
        "demand_multiplier": 1.6,
        "weather": WeatherConfig(name="clear"),
        "num_couriers": 22,
    },
    "friday_surge": {
        "demand_multiplier": 1.0,
        "weather": WeatherConfig(name="clear"),
        "num_couriers": 20,
        "demand_spike": (120, 2.0),
    },
    "understaffed": {
        "demand_multiplier": 1.3,
        "weather": WeatherConfig(name="clear"),
        "num_couriers": 14,
    },
    "holiday_peak": {
        "demand_multiplier": 2.0,
        "weather": WeatherConfig(name="clear"),
        "num_couriers": 25,
    },
    "late_night": {
        "demand_multiplier": 0.8,
        "weather": WeatherConfig(name="clear"),
        "num_couriers": 12,
        "duration_minutes": 180,
        "demand_decline_start": 60,
        "demand_decline_rate": 0.005,
    },
}

TRAIN_SCENARIOS = [
    "weekday_calm", "weekday_busy", "rainy_evening",
    "weekend_rush", "friday_surge", "understaffed",
]
TRAIN_SEEDS = [42, 123, 777]

TEST_SCENARIOS = ["holiday_peak", "late_night"]
TEST_SEEDS = [42, 123, 777]


# ── Environment Class ─────────────────────────────────────────────────────────

class FoodDelivery(Environment):
    """Food delivery dispatch optimization environment."""

    def __init__(self, task_spec: JSONObject, secrets: dict[str, str] = {}) -> None:
        super().__init__(task_spec)
        self.validated = TaskSpec.model_validate(task_spec)

        # Parse task ID: "{scenario}_seed{N}"
        parts = self.validated.id.rsplit("_seed", 1)
        if len(parts) != 2:
            raise ValueError(
                f"Invalid task ID format: {self.validated.id}. "
                f"Expected format: '{{scenario}}_seed{{N}}'"
            )
        scenario_name = parts[0]
        try:
            seed = int(parts[1])
        except ValueError:
            raise ValueError(f"Invalid seed in task ID: {parts[1]}")

        if scenario_name not in SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario_name}")

        sc = SCENARIOS[scenario_name]
        config = ScenarioConfig(
            scenario_name=scenario_name,
            seed=seed,
            demand_multiplier=sc["demand_multiplier"],
            num_couriers=sc["num_couriers"],
            weather=sc["weather"],
            duration_minutes=sc.get("duration_minutes", 240),
            demand_spike=sc.get("demand_spike"),
            demand_decline_start=sc.get("demand_decline_start"),
            demand_decline_rate=sc.get("demand_decline_rate", 0.0),
        )

        self.sim = Simulation(config)

    async def get_prompt(self) -> List[TextBlock]:
        """Return the initial prompt for the agent."""
        sc = self.sim.config
        weather_info = f"{sc.weather.name}"
        if sc.weather.demand_boost > 0:
            weather_info += f" (demand +{sc.weather.demand_boost:.0%}, travel +{sc.weather.travel_time_boost:.0%})"

        special_notes = ""
        if sc.demand_spike:
            special_notes += (
                f"\n- DEMAND SPIKE: At step {sc.demand_spike[0]} "
                f"({self.sim._step_to_time_str(sc.demand_spike[0])}), "
                f"demand will surge to {sc.demand_spike[1]}x normal levels."
            )
        if sc.demand_decline_start is not None:
            special_notes += (
                f"\n- DEMAND DECLINE: Starting at step {sc.demand_decline_start}, "
                f"demand will gradually decrease. Repositioning becomes critical."
            )

        prompt = f"""You are a food delivery dispatch optimizer managing a fleet of couriers in a simulated city.

## Scenario: {sc.scenario_name.replace('_', ' ').title()}

You manage a 6km x 6km city divided into 9 zones (3x3 grid). There are {len(self.sim.restaurants)} restaurants and {len(self.sim.couriers)} e-bike couriers.

The simulation runs for {sc.duration_minutes} minutes (5:00 PM to {self.sim._step_to_time_str(sc.duration_minutes)}). Each step = 1 minute. At each step you make dispatch decisions.

## Your Tools

1. **get_info()** - Get static city layout: zones, restaurants, courier starting positions. Does NOT advance time. Call this first.

2. **step(actions)** - Advance simulation by 1 minute with optional dispatch actions:
   - `assignments`: List of {{"order_ids": [int, ...], "courier_id": int}} -- assign 1-2 orders to an idle courier
   - `surge_multipliers`: Dict of zone_id (as string) -> float (1.0-3.0) -- set surge pricing per zone
   - `repositions`: List of {{"courier_id": int, "zone_id": int}} -- send idle couriers to a target zone
   All fields optional. Empty step() advances time with no actions.

Weather: {weather_info}
Demand multiplier: {sc.demand_multiplier}x{special_notes}

Start by calling get_info(), then call step() repeatedly to run the simulation.
"""
        return [TextBlock(text=prompt)]

    @tool
    async def get_info(self, params: GetInfoParams) -> ToolOutput:
        """Get static city layout: zones, restaurants, courier positions. Does not advance time."""
        info = self.sim.get_city_info()
        text = json.dumps(info, indent=2, default=str)
        return ToolOutput(
            metadata=info,
            blocks=[TextBlock(text=text)],
            reward=0.0,
            finished=False,
        )

    @tool
    async def step(self, params: DispatchActions) -> ToolOutput:
        """Advance the simulation by one minute with optional dispatch actions."""
        result = self.sim.tick(params)

        # Build display text
        lines = [
            f"=== Step {result.step} ({result.time_str}) ===",
            f"New orders: {len(result.new_orders)} | "
            f"Deliveries: {len(result.deliveries_completed)} | "
            f"Expired: {len(result.orders_expired)}",
            f"Pending: {len(result.pending_orders)} | "
            f"Active: {len(result.active_orders)}",
            f"Step reward: {result.step_reward:+.2f} | "
            f"Cumulative: {result.cumulative_reward:+.2f}",
            f"Stats: {result.total_orders_seen} total | "
            f"{result.total_delivered} delivered | "
            f"{result.total_expired} expired | "
            f"{result.on_time_count} on-time | "
            f"{result.late_count} late",
        ]

        if result.avg_delivery_time > 0:
            lines.append(f"Avg delivery time: {result.avg_delivery_time:.1f} min")

        # Show pending orders (agent needs these to make decisions)
        if result.pending_orders:
            lines.append(f"\n--- Pending Orders ({len(result.pending_orders)}) ---")
            for o in result.pending_orders[:15]:
                lines.append(
                    f"  Order {o['order_id']}: rest={o['restaurant_id']} "
                    f"({o['restaurant_name']}) "
                    f"r@({o['restaurant_position']['x']:.1f},{o['restaurant_position']['y']:.1f}) "
                    f"c@({o['customer_position']['x']:.1f},{o['customer_position']['y']:.1f}) "
                    f"${o['value']:.0f}+${o['tip']:.0f} "
                    f"wait={o['minutes_waiting']}min "
                    f"promise=step{o['promised_delivery_time']}"
                )
            if len(result.pending_orders) > 15:
                lines.append(f"  ... and {len(result.pending_orders) - 15} more")

        # Show courier states
        idle = [c for c in result.couriers if c["state"] == "idle"]
        busy = [c for c in result.couriers if c["state"] != "idle"]
        lines.append(
            f"\n--- Couriers: {len(idle)} idle, {len(busy)} busy ---"
        )
        for c in idle[:8]:
            lines.append(
                f"  Courier {c['courier_id']}: IDLE "
                f"at ({c['position']['x']:.1f},{c['position']['y']:.1f}) "
                f"zone {c['zone_id']} "
                f"speed={c['speed']}km/h"
            )
        if len(idle) > 8:
            lines.append(f"  ... and {len(idle) - 8} more idle")

        text = "\n".join(lines)

        if result.finished:
            final_reward = self.sim.get_normalized_reward()
            text += (
                f"\n\n{'='*40}\n"
                f"SIMULATION COMPLETE\n"
                f"{'='*40}\n"
                f"Final normalized reward: {final_reward:.4f}\n"
                f"Total orders: {result.total_orders_seen}\n"
                f"Delivered: {result.total_delivered} "
                f"({result.total_delivered/max(1,result.total_orders_seen)*100:.1f}%)\n"
                f"On-time: {result.on_time_count}\n"
                f"Late: {result.late_count}\n"
                f"Expired: {result.total_expired}\n"
                f"Avg delivery time: {result.avg_delivery_time:.1f} min\n"
            )
            return ToolOutput(
                metadata=result.model_dump(),
                blocks=[TextBlock(text=text)],
                reward=final_reward,
                finished=True,
            )

        return ToolOutput(
            metadata=result.model_dump(),
            blocks=[TextBlock(text=text)],
            reward=result.step_reward,
            finished=False,
        )

    @classmethod
    def list_tasks(cls, split: str) -> list[JSONObject]:
        """Return task list for the given split."""
        if split == "train":
            return [
                {"id": f"{scenario}_seed{seed}"}
                for scenario in TRAIN_SCENARIOS
                for seed in TRAIN_SEEDS
            ]
        elif split == "test":
            return [
                {"id": f"{scenario}_seed{seed}"}
                for scenario in TEST_SCENARIOS
                for seed in TEST_SEEDS
            ]
        raise ValueError(f"Unknown split: {split}. Expected 'train' or 'test'.")

    @classmethod
    def list_splits(cls) -> list[str]:
        return ["train", "test"]
