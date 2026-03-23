# FoodDelivery

[![OpenReward Environment](https://img.shields.io/badge/%E2%AD%90%20OpenReward-Environment-f7e6cc)](https://openreward.ai/GeneralReasoning/FoodDelivery)

## Description

FoodDelivery is a food delivery dispatch optimization environment. An agent manages a fleet of e-bike couriers across a procedurally generated city, making real-time decisions about courier-order matching, order batching, surge pricing, and fleet repositioning under stochastic conditions.

The simulation models a 4-hour dinner service period (5:00 PM -- 9:00 PM) with non-homogeneous Poisson order arrivals, Gamma-distributed restaurant preparation times, and lognormal stochastic travel times with time-of-day traffic effects. Scenarios range from calm weekday evenings to holiday peak demand, rainy weather, sudden demand spikes, and understaffed conditions.

## Capabilities

- Real-time courier-order matching and assignment under time pressure
- Order batching (up to 2 orders per courier) for delivery efficiency
- Dynamic surge pricing per zone to manage supply-demand imbalances
- Idle courier repositioning to anticipate demand shifts
- Multi-step sequential decision-making (240 steps per task)
- Reasoning about stochastic travel times, restaurant prep delays, and demand patterns

## Compute Requirements

No additional compute required beyond the environment server.

## License

MIT

## Tasks

There are 24 tasks across 8 scenarios, each run with 3 random seeds.

**Train split (18 tasks):**

| Scenario | Demand | Weather | Couriers | Duration | Description |
|----------|--------|---------|----------|----------|-------------|
| weekday_calm | 1.0x | Clear | 20 | 240 min | Baseline scenario |
| weekday_busy | 1.3x | Clear | 20 | 240 min | Higher demand, same supply |
| rainy_evening | 1.3x | Rain (+25% demand, +30% travel) | 18 | 240 min | Bad weather, fewer couriers |
| weekend_rush | 1.6x | Clear | 22 | 240 min | Weekend dinner peak |
| friday_surge | 1.0x -> 2.0x at step 120 | Clear | 20 | 240 min | Sudden demand spike mid-service |
| understaffed | 1.3x | Clear | 14 | 240 min | Severe courier shortage |

**Test split (6 tasks):**

| Scenario | Demand | Weather | Couriers | Duration | Description |
|----------|--------|---------|----------|----------|-------------|
| holiday_peak | 2.0x | Clear | 25 | 240 min | Maximum demand |
| late_night | 0.8x -> declining | Clear | 12 | 180 min | Declining demand, repositioning critical |

Each task is identified as `{scenario}_seed{N}` (e.g., `weekday_calm_seed42`). Seeds used are 42, 123, and 777.

The city model is a 6km x 6km grid divided into 9 zones (3x3), with 30 restaurants and 12--25 couriers depending on scenario. Zone types (downtown core, commercial, residential, suburban) determine local demand intensity.

## Reward Structure

This is a dense, verifiable reward environment. Rewards are computed deterministically after each 1-minute simulation step based on delivery outcomes:

| Event | Reward |
|-------|--------|
| On-time delivery (within promised window) | +1.0 |
| Slightly late delivery (within 10 min of promise) | +0.3 |
| Very late delivery (>10 min past promise) | -0.5 |
| Expired order (unassigned >20 min or undelivered >60 min) | -1.5 |
| Surge pricing revenue | +0.01 per unit revenue |

The final reward returned at task completion is the normalized cumulative reward:

$$\text{reward} = \frac{\sum_{t=1}^{T} r_t}{N_{\text{orders}}}$$

where $r_t$ is the step reward and $N_{\text{orders}}$ is the total number of orders that arrived during the simulation.

We do not use LLM graders for this task.

## Data

All data is procedurally generated from the scenario configuration and random seed. No external data files are required. The simulation generates:

- **City layout**: 9 zones with demand multipliers, 30 restaurants with cuisine types and prep time distributions
- **Order arrivals**: Non-homogeneous Poisson process with dinner rush profile
- **Prep times**: Gamma-distributed per restaurant type (fast food ~8 min, standard ~18 min, premium ~25 min)
- **Travel times**: Manhattan distance with lognormal stochastic noise and time-of-day traffic factors
- **Courier speeds**: Base ~18 km/h (e-bike) with lognormal perturbation per courier

Identical seeds produce identical simulations for reproducibility.

## Tools

Agents have access to 2 tools:

| Tool | Description |
|------|-------------|
| `get_info()` | Returns static city layout: zone boundaries, restaurant positions/types, courier starting positions. Does not advance simulation time. |
| `step(actions)` | Advances the simulation by 1 minute. Accepts optional dispatch actions: order assignments, surge multipliers per zone, and courier repositioning commands. Returns current state, step reward, and completion status. |

The `step` tool accepts three optional action types:
- **assignments**: Assign 1--2 orders to an idle courier (batching supported)
- **surge_multipliers**: Set per-zone price multipliers (1.0--3.0x, reduces demand via elasticity)
- **repositions**: Send idle couriers to target zones

## Time Horizon

FoodDelivery is a multi-step environment with 240 simulation steps per task (180 for late_night). Each step represents 1 minute of simulated time. A typical run requires 240+ tool calls (one `get_info` call plus ~240 `step` calls).

## Other Environment Requirements

There are no external API requirements. FoodDelivery works out of the box with the OpenReward endpoint without any secrets.

## Safety

Agents in FoodDelivery optimize delivery logistics within a fully simulated environment. The surge pricing mechanism introduces an economic optimization component where agents must balance revenue extraction against service quality. While the environment rewards efficient dispatch, the objectives are bounded and the simulation is self-contained.

## Citations

```bibtex
@dataset{GRFoodDelivery,
  author    = {General Reasoning Inc. Team},
  title     = {FoodDelivery},
  year      = {2026},
  publisher = {OpenReward},
  url       = {https://openreward.ai/GeneralReasoning/fooddelivery}
}
```
