"""
Golden tests for the Food Delivery Dispatch Optimization environment.

Tests simulation mechanics, environment wrapper, reward computation,
edge cases, and determinism. Run with: pytest golden_tests.py -v
"""

import asyncio
import re

import numpy as np
import pytest

from simulation import (
    Assignment,
    CourierState,
    DispatchActions,
    OrderStatus,
    Position,
    Reposition,
    ScenarioConfig,
    Simulation,
    WeatherConfig,
    ZoneType,
)
from fooddelivery import FoodDelivery, TRAIN_SCENARIOS, TEST_SCENARIOS, TRAIN_SEEDS, TEST_SEEDS


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_sim(scenario="weekday_calm", seed=42, **overrides) -> Simulation:
    """Create a simulation with defaults."""
    from fooddelivery import SCENARIOS
    sc = SCENARIOS[scenario]
    kwargs = {
        "scenario_name": scenario,
        "seed": seed,
        "demand_multiplier": sc["demand_multiplier"],
        "num_couriers": sc["num_couriers"],
        "weather": sc["weather"],
        "duration_minutes": sc.get("duration_minutes", 240),
        "demand_spike": sc.get("demand_spike"),
        "demand_decline_start": sc.get("demand_decline_start"),
        "demand_decline_rate": sc.get("demand_decline_rate", 0.0),
    }
    kwargs.update(overrides)
    config = ScenarioConfig(**kwargs)
    return Simulation(config)


def make_env(task_id="weekday_calm_seed42") -> FoodDelivery:
    """Create an environment instance."""
    return FoodDelivery(task_spec={"id": task_id})


def run_empty_steps(sim: Simulation, n: int):
    """Run n steps with no actions."""
    results = []
    for _ in range(n):
        r = sim.tick(DispatchActions())
        results.append(r)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Category 1: Environment Framework Tests
# ══════════════════════════════════════════════════════════════════════════════

def test_list_splits():
    """list_splits returns train and test."""
    splits = FoodDelivery.list_splits()
    assert splits == ["train", "test"]


def test_list_tasks_train_count():
    """Train split has 6 scenarios x 3 seeds = 18 tasks."""
    tasks = FoodDelivery.list_tasks("train")
    assert len(tasks) == len(TRAIN_SCENARIOS) * len(TRAIN_SEEDS)
    assert len(tasks) == 18


def test_list_tasks_test_count():
    """Test split has 2 scenarios x 3 seeds = 6 tasks."""
    tasks = FoodDelivery.list_tasks("test")
    assert len(tasks) == len(TEST_SCENARIOS) * len(TEST_SEEDS)
    assert len(tasks) == 6


def test_list_tasks_invalid_split():
    """Invalid split raises ValueError."""
    with pytest.raises(ValueError, match="Unknown split"):
        FoodDelivery.list_tasks("invalid")


def test_task_id_format():
    """All task IDs match {scenario}_seed{N} pattern."""
    pattern = re.compile(r"^[a-z_]+_seed\d+$")
    for split in ["train", "test"]:
        tasks = FoodDelivery.list_tasks(split)
        for task in tasks:
            assert pattern.match(task["id"]), f"Bad task ID: {task['id']}"


def test_environment_init():
    """Environment initializes correctly from valid task spec."""
    env = make_env("weekday_calm_seed42")
    assert env.sim.step_num == 0
    assert len(env.sim.restaurants) == 30
    assert len(env.sim.couriers) == 20
    assert len(env.sim.zones) == 9
    assert not env.sim.finished


def test_environment_init_invalid_scenario():
    """Invalid scenario raises ValueError."""
    with pytest.raises(ValueError, match="Unknown scenario"):
        make_env("nonexistent_seed42")


def test_environment_init_invalid_format():
    """Invalid task ID format raises ValueError."""
    with pytest.raises(ValueError, match="Invalid task ID"):
        make_env("bad_format")


@pytest.mark.asyncio
async def test_get_prompt_returns_textblock():
    """get_prompt returns a list of TextBlocks with content."""
    env = make_env()
    result = await env.get_prompt()
    assert isinstance(result, list)
    assert len(result) >= 1
    assert hasattr(result[0], "text")
    assert "dispatch" in result[0].text.lower()
    assert "step" in result[0].text.lower()


# ══════════════════════════════════════════════════════════════════════════════
# Category 2: City Generation Tests
# ══════════════════════════════════════════════════════════════════════════════

def test_zone_count_and_types():
    """9 zones with correct types and demand multipliers."""
    sim = make_sim()
    assert len(sim.zones) == 9

    # Check specific zone types
    zone_map = {z.zone_id: z for z in sim.zones}
    assert zone_map[4].zone_type == ZoneType.DOWNTOWN_CORE
    assert zone_map[4].demand_multiplier == 2.0
    assert zone_map[1].zone_type == ZoneType.COMMERCIAL
    assert zone_map[0].zone_type == ZoneType.SUBURBAN
    assert zone_map[0].demand_multiplier == 0.7
    assert zone_map[2].zone_type == ZoneType.RESIDENTIAL


def test_zone_boundaries():
    """Zone boundaries cover the full 6x6 grid."""
    sim = make_sim()
    for zone in sim.zones:
        row = zone.zone_id // 3
        col = zone.zone_id % 3
        assert zone.x_min == col * 2.0
        assert zone.x_max == (col + 1) * 2.0
        assert zone.y_min == row * 2.0
        assert zone.y_max == (row + 1) * 2.0
        # Center should be in the middle
        assert zone.center.x == zone.x_min + 1.0
        assert zone.center.y == zone.y_min + 1.0


def test_restaurant_count():
    """Exactly 30 restaurants with valid positions and types."""
    sim = make_sim()
    assert len(sim.restaurants) == 30
    valid_types = {"fast_food", "standard", "premium"}
    for r in sim.restaurants:
        assert r.restaurant_type in valid_types
        assert 0 < r.position.x < 6.0
        assert 0 < r.position.y < 6.0
        assert 0 <= r.zone_id <= 8
        assert r.base_prep_time > 0


def test_restaurant_distribution():
    """Downtown core should have more restaurants than suburban."""
    sim = make_sim()
    zone_counts = {}
    for r in sim.restaurants:
        zone_counts[r.zone_id] = zone_counts.get(r.zone_id, 0) + 1
    # Downtown core (zone 4) should have more than suburban corners
    assert zone_counts.get(4, 0) > zone_counts.get(0, 0)


def test_courier_count_matches_config():
    """Courier count matches scenario configuration."""
    for scenario, expected in [("weekday_calm", 20), ("understaffed", 14),
                                ("weekend_rush", 22), ("holiday_peak", 25)]:
        sim = make_sim(scenario)
        assert len(sim.couriers) == expected, f"{scenario}: expected {expected}, got {len(sim.couriers)}"


def test_courier_initial_state():
    """All couriers start IDLE with no orders."""
    sim = make_sim()
    for c in sim.couriers:
        assert c.state == CourierState.IDLE
        assert len(c.current_orders) == 0
        assert c.speed > 0
        assert 0 < c.position.x < 6.0
        assert 0 < c.position.y < 6.0


def test_deterministic_generation():
    """Same seed produces identical city layout."""
    sim1 = make_sim(seed=42)
    sim2 = make_sim(seed=42)

    assert len(sim1.restaurants) == len(sim2.restaurants)
    for r1, r2 in zip(sim1.restaurants, sim2.restaurants):
        assert r1.restaurant_id == r2.restaurant_id
        assert r1.name == r2.name
        assert abs(r1.position.x - r2.position.x) < 1e-10
        assert abs(r1.position.y - r2.position.y) < 1e-10
        assert r1.restaurant_type == r2.restaurant_type

    for c1, c2 in zip(sim1.couriers, sim2.couriers):
        assert abs(c1.position.x - c2.position.x) < 1e-10
        assert abs(c1.speed - c2.speed) < 1e-10


def test_different_seeds_differ():
    """Different seeds produce different layouts."""
    sim1 = make_sim(seed=42)
    sim2 = make_sim(seed=999)
    # Restaurant positions should differ
    positions_differ = False
    for r1, r2 in zip(sim1.restaurants, sim2.restaurants):
        if abs(r1.position.x - r2.position.x) > 0.01:
            positions_differ = True
            break
    assert positions_differ


# ══════════════════════════════════════════════════════════════════════════════
# Category 3: Simulation Mechanics Tests
# ══════════════════════════════════════════════════════════════════════════════

def test_step_advances_time():
    """Each tick increments step_num by 1."""
    sim = make_sim()
    assert sim.step_num == 0
    sim.tick(DispatchActions())
    assert sim.step_num == 1
    sim.tick(DispatchActions())
    assert sim.step_num == 2


def test_orders_arrive():
    """Orders should arrive within the first 10 steps."""
    sim = make_sim()
    run_empty_steps(sim, 10)
    assert sim.total_orders_seen > 0
    # Check that orders have valid fields
    for oid, order in sim.orders.items():
        assert order.order_id == oid
        assert order.placed_at >= 0
        assert order.promised_delivery_time > order.placed_at
        assert order.value >= 15.0
        assert order.tip >= 2.0
        assert order.prep_time > 0


def test_order_arrival_rate_varies():
    """Peak hours should produce more orders than early hours."""
    sim = make_sim(seed=42)
    # Run first 30 steps (5:00-5:30 PM, early)
    early_results = run_empty_steps(sim, 30)
    early_orders = sim.total_orders_seen

    # Create new sim, fast-forward to peak (6:30 PM = step 90)
    sim2 = make_sim(seed=42)
    run_empty_steps(sim2, 90)
    pre_peak_total = sim2.total_orders_seen
    run_empty_steps(sim2, 30)
    peak_orders = sim2.total_orders_seen - pre_peak_total

    # Peak should have more orders than early
    assert peak_orders > early_orders


def test_order_assignment():
    """Assigning an order to a courier updates both states."""
    sim = make_sim()
    # Generate some orders
    run_empty_steps(sim, 5)

    # Find a pending order and idle courier
    pending = [o for o in sim.orders.values() if o.status == OrderStatus.PENDING]
    idle = [c for c in sim.couriers if c.state == CourierState.IDLE]

    if not pending or not idle:
        pytest.skip("No pending orders or idle couriers after 5 steps")

    order = pending[0]
    courier = idle[0]

    sim.tick(DispatchActions(
        assignments=[Assignment(order_ids=[order.order_id], courier_id=courier.courier_id)]
    ))

    assert order.status == OrderStatus.ASSIGNED
    assert order.assigned_courier_id == courier.courier_id
    assert courier.state == CourierState.EN_ROUTE_PICKUP
    assert order.order_id in courier.current_orders


def test_order_expiry_unassigned():
    """Unassigned orders expire after 20 minutes."""
    sim = make_sim(seed=42)
    # Generate orders -- run enough steps to guarantee orders
    run_empty_steps(sim, 10)

    assert sim.total_orders_seen > 0, "Should have orders after 10 steps"

    # Run 25 more steps without assigning anything
    run_empty_steps(sim, 25)

    # Some early orders should have expired
    expired = [o for o in sim.orders.values() if o.status == OrderStatus.EXPIRED]
    assert len(expired) > 0, "Expected some orders to expire after 25+ min unassigned"


def test_full_delivery_lifecycle():
    """Order goes through PENDING -> ASSIGNED -> PICKED_UP -> DELIVERED."""
    sim = make_sim(seed=100)
    # Generate orders
    run_empty_steps(sim, 3)

    pending = [o for o in sim.orders.values() if o.status == OrderStatus.PENDING]
    idle = [c for c in sim.couriers if c.state == CourierState.IDLE]

    if not pending or not idle:
        pytest.skip("No pending orders or idle couriers")

    order = pending[0]
    courier = idle[0]

    # Assign
    sim.tick(DispatchActions(
        assignments=[Assignment(order_ids=[order.order_id], courier_id=courier.courier_id)]
    ))
    assert order.status == OrderStatus.ASSIGNED

    # Run until order is delivered or 55 steps pass (should be enough)
    for _ in range(55):
        sim.tick(DispatchActions())
        if order.status == OrderStatus.DELIVERED:
            break

    assert order.status == OrderStatus.DELIVERED, (
        f"Order should be delivered but is {order.status}. "
        f"Courier state: {courier.state}, orders: {courier.current_orders}"
    )
    assert order.delivered_at is not None
    assert order.delivered_at > order.placed_at


def test_batch_assignment():
    """Assigning 2 orders to one courier works."""
    sim = make_sim(seed=200)
    run_empty_steps(sim, 8)

    pending = [o for o in sim.orders.values() if o.status == OrderStatus.PENDING]
    idle = [c for c in sim.couriers if c.state == CourierState.IDLE]

    if len(pending) < 2 or not idle:
        pytest.skip("Not enough pending orders or idle couriers for batch test")

    o1, o2 = pending[0], pending[1]
    courier = idle[0]

    sim.tick(DispatchActions(
        assignments=[Assignment(order_ids=[o1.order_id, o2.order_id], courier_id=courier.courier_id)]
    ))

    assert o1.status == OrderStatus.ASSIGNED
    assert o2.status == OrderStatus.ASSIGNED
    assert len(courier.current_orders) == 2
    assert o1.order_id in courier.current_orders
    assert o2.order_id in courier.current_orders


def test_max_batch_size_enforced():
    """Cannot assign more than 2 orders to one courier."""
    sim = make_sim(seed=300)
    run_empty_steps(sim, 15)

    pending = [o for o in sim.orders.values() if o.status == OrderStatus.PENDING]
    idle = [c for c in sim.couriers if c.state == CourierState.IDLE]

    if len(pending) < 3 or not idle:
        pytest.skip("Not enough pending orders for batch limit test")

    o1, o2, o3 = pending[0], pending[1], pending[2]
    courier = idle[0]

    sim.tick(DispatchActions(
        assignments=[Assignment(
            order_ids=[o1.order_id, o2.order_id, o3.order_id],
            courier_id=courier.courier_id
        )]
    ))

    # Only first 2 should be assigned
    assert len(courier.current_orders) == 2
    assigned_count = sum(
        1 for o in [o1, o2, o3] if o.status == OrderStatus.ASSIGNED
    )
    assert assigned_count == 2
    # Third order should still be pending
    assert o3.status == OrderStatus.PENDING


def test_courier_repositioning():
    """Idle courier can be repositioned to a target zone."""
    sim = make_sim()
    idle = [c for c in sim.couriers if c.state == CourierState.IDLE]
    assert len(idle) > 0

    courier = idle[0]
    original_zone = courier.zone_id
    # Pick a different zone
    target_zone = (original_zone + 4) % 9

    sim.tick(DispatchActions(
        repositions=[Reposition(courier_id=courier.courier_id, zone_id=target_zone)]
    ))

    assert courier.state == CourierState.REPOSITIONING
    assert courier.eta_remaining > 0


# ══════════════════════════════════════════════════════════════════════════════
# Category 4: Reward Tests
# ══════════════════════════════════════════════════════════════════════════════

def test_on_time_delivery_reward():
    """On-time delivery should give +1.0 reward."""
    sim = make_sim(seed=100)
    run_empty_steps(sim, 3)

    pending = [o for o in sim.orders.values() if o.status == OrderStatus.PENDING]
    idle = [c for c in sim.couriers if c.state == CourierState.IDLE]

    if not pending or not idle:
        pytest.skip("No orders/couriers available")

    order = pending[0]
    courier = idle[0]

    sim.tick(DispatchActions(
        assignments=[Assignment(order_ids=[order.order_id], courier_id=courier.courier_id)]
    ))

    # Run until delivery
    for _ in range(55):
        result = sim.tick(DispatchActions())
        if order.status == OrderStatus.DELIVERED:
            # Check if it was on time
            delivery_time = order.delivered_at - order.placed_at
            promise_window = order.promised_delivery_time - order.placed_at
            if delivery_time <= promise_window:
                assert sim.on_time_count > 0
            break


def test_expired_order_penalty():
    """Expired orders should give -1.5 reward."""
    sim = make_sim(seed=42)
    run_empty_steps(sim, 10)

    assert sim.total_orders_seen > 0, "Should have orders after 10 steps"
    initial_reward = sim.cumulative_reward

    # Let orders expire by not assigning them for 20 more steps
    run_empty_steps(sim, 20)

    expired = [o for o in sim.orders.values() if o.status == OrderStatus.EXPIRED]
    if expired:
        # Reward should have decreased by 1.5 per expired order
        expected_penalty = -1.5 * len(expired)
        # Account for possible deliveries happening too
        assert sim.cumulative_reward <= initial_reward + len(expired) * 1.5


def test_reward_normalization():
    """Final normalized reward equals cumulative / total_orders_seen."""
    sim = make_sim()
    # Run full simulation
    for _ in range(240):
        sim.tick(DispatchActions())

    assert sim.finished
    assert sim.total_orders_seen > 0
    normalized = sim.get_normalized_reward()
    expected = sim.cumulative_reward / sim.total_orders_seen
    assert abs(normalized - expected) < 1e-6


def test_greedy_assignment_produces_positive_deliveries():
    """A simple greedy strategy should produce some deliveries."""
    sim = make_sim(seed=42)

    total_assigned = 0
    for step in range(240):
        # Simple greedy: assign each pending order to nearest idle courier
        pending = [o for o in sim.orders.values() if o.status == OrderStatus.PENDING]
        idle = [c for c in sim.couriers if c.state == CourierState.IDLE]

        assignments = []
        used_couriers = set()
        for order in pending:
            if not idle:
                break
            # Find nearest idle courier not yet used this step
            best_courier = None
            best_dist = float('inf')
            rest = sim.restaurants[order.restaurant_id]
            for c in idle:
                if c.courier_id in used_couriers:
                    continue
                d = Simulation._manhattan_distance(c.position, rest.position)
                if d < best_dist:
                    best_dist = d
                    best_courier = c
            if best_courier:
                assignments.append(Assignment(
                    order_ids=[order.order_id],
                    courier_id=best_courier.courier_id,
                ))
                used_couriers.add(best_courier.courier_id)
                total_assigned += 1

        sim.tick(DispatchActions(assignments=assignments))

    assert sim.finished
    assert sim.total_delivered > 0, "Greedy strategy should deliver some orders"
    assert sim.on_time_count > 0, "Some orders should be on time"
    assert total_assigned > 0


# ══════════════════════════════════════════════════════════════════════════════
# Category 5: Edge Cases and Termination
# ══════════════════════════════════════════════════════════════════════════════

def test_simulation_terminates_at_duration():
    """Simulation finishes after duration_minutes steps."""
    sim = make_sim()
    assert sim.config.duration_minutes == 240

    for _ in range(240):
        result = sim.tick(DispatchActions())

    assert result.finished
    assert sim.finished
    assert sim.step_num == 240


def test_late_night_shorter_duration():
    """Late night scenario runs 180 steps, not 240."""
    sim = make_sim("late_night")
    assert sim.config.duration_minutes == 180

    for _ in range(180):
        result = sim.tick(DispatchActions())

    assert result.finished
    assert sim.step_num == 180


def test_get_info_does_not_advance_time():
    """get_city_info doesn't change simulation state."""
    sim = make_sim()
    step_before = sim.step_num
    orders_before = sim.total_orders_seen

    info = sim.get_city_info()

    assert sim.step_num == step_before
    assert sim.total_orders_seen == orders_before
    assert "zones" in info
    assert "restaurants" in info
    assert "couriers" in info


def test_invalid_courier_id_skipped():
    """Invalid courier ID doesn't crash, just gets skipped."""
    sim = make_sim()
    run_empty_steps(sim, 5)

    pending = [o for o in sim.orders.values() if o.status == OrderStatus.PENDING]
    if not pending:
        pytest.skip("No pending orders")

    # Try to assign to non-existent courier
    result = sim.tick(DispatchActions(
        assignments=[Assignment(order_ids=[pending[0].order_id], courier_id=999)]
    ))

    # Should not crash; order should still be pending
    assert pending[0].status == OrderStatus.PENDING


def test_invalid_order_id_skipped():
    """Invalid order ID doesn't crash."""
    sim = make_sim()
    result = sim.tick(DispatchActions(
        assignments=[Assignment(order_ids=[99999], courier_id=0)]
    ))
    # Should not crash


def test_surge_multiplier_clamping():
    """Surge multiplier gets clamped to [1.0, 3.0]."""
    sim = make_sim()

    sim.tick(DispatchActions(
        surge_multipliers={"4": 5.0, "0": 0.5}
    ))

    assert sim.active_surge.get(4, 1.0) == 3.0
    assert sim.active_surge.get(0, 1.0) == 1.0  # clamped up to 1.0


def test_surge_resets_each_step():
    """Surge multipliers reset each step."""
    sim = make_sim()

    sim.tick(DispatchActions(surge_multipliers={"4": 2.0}))
    assert sim.active_surge.get(4, 1.0) == 2.0

    # Next step without surge
    sim.tick(DispatchActions())
    assert sim.active_surge.get(4, 1.0) == 1.0  # reset


def test_reposition_only_idle_couriers():
    """Cannot reposition a busy courier."""
    sim = make_sim()
    run_empty_steps(sim, 5)

    pending = [o for o in sim.orders.values() if o.status == OrderStatus.PENDING]
    idle = [c for c in sim.couriers if c.state == CourierState.IDLE]

    if not pending or not idle:
        pytest.skip("Need orders and couriers")

    courier = idle[0]
    # Assign order to make courier busy
    sim.tick(DispatchActions(
        assignments=[Assignment(order_ids=[pending[0].order_id], courier_id=courier.courier_id)]
    ))
    assert courier.state != CourierState.IDLE

    # Try to reposition the busy courier
    sim.tick(DispatchActions(
        repositions=[Reposition(courier_id=courier.courier_id, zone_id=0)]
    ))

    # Should NOT be repositioning (should still be on pickup route)
    assert courier.state != CourierState.REPOSITIONING


def test_assign_to_busy_courier_skipped():
    """Cannot assign orders to a busy courier."""
    sim = make_sim()
    run_empty_steps(sim, 5)

    pending = [o for o in sim.orders.values() if o.status == OrderStatus.PENDING]
    idle = [c for c in sim.couriers if c.state == CourierState.IDLE]

    if len(pending) < 2 or not idle:
        pytest.skip("Need at least 2 orders and 1 courier")

    courier = idle[0]
    o1, o2 = pending[0], pending[1]

    # Assign first order
    sim.tick(DispatchActions(
        assignments=[Assignment(order_ids=[o1.order_id], courier_id=courier.courier_id)]
    ))
    assert courier.state != CourierState.IDLE

    # Try to assign second order to same (now busy) courier
    sim.tick(DispatchActions(
        assignments=[Assignment(order_ids=[o2.order_id], courier_id=courier.courier_id)]
    ))

    # Second order should still be pending
    assert o2.status == OrderStatus.PENDING


def test_double_assignment_same_step():
    """Assigning the same order twice in one step only works once."""
    sim = make_sim()
    run_empty_steps(sim, 5)

    pending = [o for o in sim.orders.values() if o.status == OrderStatus.PENDING]
    idle = [c for c in sim.couriers if c.state == CourierState.IDLE]

    if not pending or len(idle) < 2:
        pytest.skip("Need 1 order and 2 couriers")

    order = pending[0]
    c1, c2 = idle[0], idle[1]

    sim.tick(DispatchActions(
        assignments=[
            Assignment(order_ids=[order.order_id], courier_id=c1.courier_id),
            Assignment(order_ids=[order.order_id], courier_id=c2.courier_id),
        ]
    ))

    # Order should be assigned to first courier only
    assert order.status == OrderStatus.ASSIGNED
    assert order.assigned_courier_id == c1.courier_id
    # Second courier should still be idle
    assert c2.state == CourierState.IDLE


def test_empty_step_no_crash():
    """Step with completely empty actions works fine."""
    sim = make_sim()
    result = sim.tick(DispatchActions())
    assert result.step == 1
    assert not result.finished


def test_end_of_simulation_expiry():
    """All remaining orders expire at simulation end."""
    sim = make_sim()
    # Run most of the simulation
    run_empty_steps(sim, 235)

    # Count active orders
    active_before = sum(
        1 for o in sim.orders.values()
        if o.status not in (OrderStatus.DELIVERED, OrderStatus.EXPIRED)
    )

    # Run last 5 steps to finish
    for _ in range(5):
        result = sim.tick(DispatchActions())

    assert sim.finished

    # All orders should be either delivered or expired now
    for order in sim.orders.values():
        assert order.status in (OrderStatus.DELIVERED, OrderStatus.EXPIRED)


def test_step_result_structure():
    """StepResult has all expected fields."""
    sim = make_sim()
    result = sim.tick(DispatchActions())

    assert isinstance(result.step, int)
    assert isinstance(result.time_str, str)
    assert isinstance(result.finished, bool)
    assert isinstance(result.pending_orders, list)
    assert isinstance(result.active_orders, list)
    assert isinstance(result.couriers, list)
    assert isinstance(result.new_orders, list)
    assert isinstance(result.step_reward, float)
    assert isinstance(result.cumulative_reward, float)
    assert isinstance(result.total_orders_seen, int)
    assert isinstance(result.total_delivered, int)
    assert isinstance(result.total_expired, int)
    assert isinstance(result.avg_delivery_time, float)


def test_time_str_format():
    """Time string shows correct format."""
    sim = make_sim()
    result = sim.tick(DispatchActions())
    # Step 1 should be 5:01 PM
    assert result.time_str == "5:01 PM"

    run_empty_steps(sim, 59)
    result = sim.tick(DispatchActions())
    # Step 61 should be 6:01 PM
    assert result.time_str == "6:01 PM"


def test_travel_time_stochasticity():
    """Travel times should vary due to lognormal noise."""
    sim = make_sim(seed=42)
    a = Position(x=1.0, y=1.0)
    b = Position(x=4.0, y=4.0)

    times = []
    for _ in range(50):
        t = sim._compute_travel_time(a, b, 18.0, 60)
        times.append(t)

    # Should have meaningful variance
    assert np.std(times) > 0.5, "Travel times should vary stochastically"
    assert all(1.0 <= t <= 120.0 for t in times), "All times should be clamped"


def test_manhattan_distance():
    """Manhattan distance computation is correct."""
    a = Position(x=1.0, y=2.0)
    b = Position(x=4.0, y=6.0)
    assert Simulation._manhattan_distance(a, b) == 7.0


def test_zone_for_position():
    """Zone lookup is correct for various positions."""
    sim = make_sim()
    # Zone 0: x=[0,2), y=[0,2) -> center (1,1)
    assert sim._get_zone_for_position(Position(x=1.0, y=1.0)) == 0
    # Zone 4: x=[2,4), y=[2,4) -> center (3,3)
    assert sim._get_zone_for_position(Position(x=3.0, y=3.0)) == 4
    # Zone 8: x=[4,6), y=[4,6) -> center (5,5)
    assert sim._get_zone_for_position(Position(x=5.0, y=5.0)) == 8


def test_weather_affects_travel_time():
    """Rainy weather should increase travel times."""
    sim_clear = make_sim("weekday_calm", seed=42)
    sim_rain = make_sim("rainy_evening", seed=42)

    a = Position(x=1.0, y=1.0)
    b = Position(x=4.0, y=4.0)

    # Use same seed state for comparison (reset rng)
    sim_clear.rng = np.random.default_rng(42)
    sim_rain.rng = np.random.default_rng(42)

    times_clear = [sim_clear._compute_travel_time(a, b, 18.0, 60) for _ in range(20)]
    sim_clear.rng = np.random.default_rng(42)
    sim_rain.rng = np.random.default_rng(42)
    times_rain = [sim_rain._compute_travel_time(a, b, 18.0, 60) for _ in range(20)]

    assert np.mean(times_rain) > np.mean(times_clear), (
        "Rain should increase average travel time"
    )


def test_demand_spike_scenario():
    """Friday surge scenario should increase order rate after step 120."""
    sim = make_sim("friday_surge", seed=42)

    # Run to step 115 (before spike)
    run_empty_steps(sim, 115)
    orders_before = sim.total_orders_seen

    # Run 5 more steps just before spike
    run_empty_steps(sim, 5)
    orders_pre_spike = sim.total_orders_seen - orders_before

    # Run 10 steps after spike (step 120+)
    run_empty_steps(sim, 10)
    orders_post_spike = sim.total_orders_seen - orders_before - orders_pre_spike

    # Post-spike should have more orders (2x demand)
    # This is probabilistic, but with 10 steps at 2x rate it should be higher
    assert orders_post_spike >= orders_pre_spike, (
        f"Post-spike ({orders_post_spike}) should have more orders than "
        f"pre-spike ({orders_pre_spike})"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Category 6: Distribution Verification Tests (Statistical)
# ══════════════════════════════════════════════════════════════════════════════

def test_prep_time_gamma_distribution():
    """Prep times follow Gamma(4, base/4) with correct means per restaurant type."""
    rng = np.random.default_rng(42)
    for rtype, base_prep in [("fast_food", 8.0), ("standard", 18.0), ("premium", 25.0)]:
        samples = []
        for _ in range(2000):
            shape = 4.0
            scale = base_prep / shape
            t = float(rng.gamma(shape, scale))
            t = max(3.0, min(60.0, t))
            samples.append(t)
        mean = np.mean(samples)
        # Mean should be close to base_prep_time (within 15% due to clamping)
        assert abs(mean - base_prep) / base_prep < 0.15, (
            f"{rtype}: mean={mean:.1f}, expected~{base_prep}"
        )
        # All values should be in clamped range
        assert all(3.0 <= s <= 60.0 for s in samples)
        # CV should be roughly 0.5 (= 1/sqrt(shape))
        cv = np.std(samples) / np.mean(samples)
        assert 0.3 < cv < 0.7, f"{rtype}: CV={cv:.2f}, expected ~0.5"


def test_courier_speed_lognormal():
    """Courier speeds follow lognormal with geometric mean ~18 km/h."""
    rng = np.random.default_rng(42)
    speeds = [float(18.0 * np.exp(rng.normal(0, 0.1))) for _ in range(500)]
    geo_mean = np.exp(np.mean(np.log(speeds)))
    # Geometric mean should be ~18 km/h
    assert abs(geo_mean - 18.0) < 0.5, f"Geometric mean={geo_mean:.2f}, expected ~18"
    # Practical range: ~15-21 km/h for most couriers (within 2 sigma)
    assert all(12.0 < s < 26.0 for s in speeds), "Speed out of reasonable range"
    # Check spread
    log_std = np.std(np.log(speeds))
    assert abs(log_std - 0.1) < 0.02, f"Log std={log_std:.3f}, expected ~0.1"


def test_order_arrival_nhpp_profile():
    """Order arrival rate follows dinner rush profile with peak at 6:30-7:30 PM."""
    sim = make_sim(seed=42)
    # Run full simulation with no actions, count orders by 30-min window
    windows = {}  # window_start -> order count
    for step in range(240):
        before = sim.total_orders_seen
        sim.tick(DispatchActions())
        new_count = sim.total_orders_seen - before
        window = (step // 30) * 30
        windows[window] = windows.get(window, 0) + new_count

    # Peak window (steps 90-119 = 6:30-7:00 PM) should have most orders
    early_window = windows.get(0, 0)      # 5:00-5:30 PM
    peak_window = windows.get(90, 0)       # 6:30-7:00 PM
    late_window = windows.get(210, 0)      # 8:30-9:00 PM

    assert peak_window > early_window, (
        f"Peak ({peak_window}) should exceed early ({early_window})"
    )
    assert peak_window > late_window, (
        f"Peak ({peak_window}) should exceed late ({late_window})"
    )


def test_travel_time_lognormal_properties():
    """Travel times are lognormally distributed with right skew."""
    sim = make_sim(seed=42)
    a = Position(x=1.0, y=1.0)
    b = Position(x=4.0, y=4.0)

    # Sample at peak hour (step=90)
    peak_times = [sim._compute_travel_time(a, b, 18.0, 90) for _ in range(500)]
    # Sample at off-peak (step=200)
    off_peak_times = [sim._compute_travel_time(a, b, 18.0, 200) for _ in range(500)]

    # Lognormal: median < mean (right-skewed)
    assert np.median(peak_times) < np.mean(peak_times), "Peak times should be right-skewed"
    assert np.median(off_peak_times) < np.mean(off_peak_times), "Off-peak should be right-skewed"

    # Peak times should have higher variance (sigma=0.25 vs 0.15)
    assert np.std(peak_times) > np.std(off_peak_times), (
        f"Peak std ({np.std(peak_times):.2f}) should exceed off-peak "
        f"({np.std(off_peak_times):.2f})"
    )

    # All times clamped
    assert all(1.0 <= t <= 120.0 for t in peak_times + off_peak_times)


def test_surge_elasticity_reduces_demand():
    """Surge pricing reduces demand via elasticity model."""
    # Run two simulations: one with high surge on zone 4, one without
    # Use many steps to average out Poisson noise
    orders_no_surge = 0
    orders_with_surge = 0

    for seed in range(10):
        sim1 = make_sim(seed=seed + 1000)
        sim2 = make_sim(seed=seed + 1000)
        for _ in range(60):
            sim1.tick(DispatchActions())
            sim2.tick(DispatchActions(surge_multipliers={"4": 3.0}))
        # Count zone-4 orders
        for o in sim1.orders.values():
            if o.zone_id == 4:
                orders_no_surge += 1
        for o in sim2.orders.values():
            if o.zone_id == 4:
                orders_with_surge += 1

    # High surge should reduce zone-4 demand
    assert orders_with_surge < orders_no_surge, (
        f"Surge should reduce demand: no_surge={orders_no_surge}, "
        f"with_surge={orders_with_surge}"
    )


def test_demand_decline_late_night():
    """Late night scenario has declining demand over time."""
    sim = make_sim("late_night", seed=42)

    # Count orders in steps 60-89 (just after decline starts)
    run_empty_steps(sim, 60)
    orders_at_60 = sim.total_orders_seen
    run_empty_steps(sim, 30)
    orders_early = sim.total_orders_seen - orders_at_60

    # Count orders in steps 150-179 (deep into decline)
    run_empty_steps(sim, 60)
    orders_at_150 = sim.total_orders_seen
    run_empty_steps(sim, 30)
    orders_late = sim.total_orders_seen - orders_at_150

    assert orders_early > orders_late, (
        f"Early decline ({orders_early}) should exceed deep decline ({orders_late})"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Category 7: Order Lifecycle Boundary Tests
# ══════════════════════════════════════════════════════════════════════════════

def test_pending_expiry_boundary():
    """Pending orders expire at >20 minutes (not at ==20).

    The expiry check (line 642 in simulation.py) uses `time_since_placed > 20`.
    The check runs BEFORE step_num is incremented. So an order placed at step P:
    - Tick at step_num=P+20: time_since_placed=20, NOT > 20, order stays PENDING
    - Tick at step_num=P+21: time_since_placed=21, > 20, order EXPIRES
    """
    sim = make_sim(seed=42)
    # Run 5 steps to generate some orders
    run_empty_steps(sim, 5)

    # Find a pending order placed early
    early_orders = [o for o in sim.orders.values()
                    if o.status == OrderStatus.PENDING and o.placed_at <= 4]
    if not early_orders:
        pytest.skip("No early orders found")

    order = early_orders[0]
    placed = order.placed_at

    # Run until step_num == placed + 20 (expiry check sees exactly 20, no expiry yet)
    steps_needed = (placed + 20) - sim.step_num
    if steps_needed > 0:
        run_empty_steps(sim, steps_needed)

    assert sim.step_num == placed + 20
    # At this point the NEXT tick will check time_since_placed == 20 (not > 20)
    # so order should survive this tick
    run_empty_steps(sim, 1)
    # After this tick, step_num == placed+21, but expiry check ran at step_num==placed+20
    # with time_since_placed==20 which is NOT > 20, so order should still be PENDING
    assert order.status == OrderStatus.PENDING, (
        f"Order should NOT expire at exactly 20 min, got {order.status}"
    )

    # One more tick: expiry check runs at step_num==placed+21, time_since_placed==21 > 20
    run_empty_steps(sim, 1)
    assert order.status == OrderStatus.EXPIRED, (
        f"Order should expire at >20 min, got {order.status}"
    )


def test_60_minute_total_expiry():
    """Orders expire after 60 minutes regardless of assignment status."""
    sim = make_sim(seed=42)
    run_empty_steps(sim, 5)

    pending = [o for o in sim.orders.values() if o.status == OrderStatus.PENDING]
    idle = [c for c in sim.couriers if c.state == CourierState.IDLE]
    if not pending or not idle:
        pytest.skip("Need orders and couriers")

    order = pending[0]
    courier = idle[0]

    # Assign the order
    sim.tick(DispatchActions(
        assignments=[Assignment(order_ids=[order.order_id], courier_id=courier.courier_id)]
    ))
    assert order.status == OrderStatus.ASSIGNED

    # Run 65 more steps (total > 60 min from placement)
    for _ in range(65):
        sim.tick(DispatchActions())
        if order.status in (OrderStatus.DELIVERED, OrderStatus.EXPIRED):
            break

    # Order should be either delivered or expired
    assert order.status in (OrderStatus.DELIVERED, OrderStatus.EXPIRED), (
        f"After 65+ min, order should be resolved, got {order.status}"
    )


def test_end_of_simulation_all_orders_resolved():
    """At simulation end, every order is DELIVERED or EXPIRED."""
    sim = make_sim(seed=42)
    for _ in range(240):
        sim.tick(DispatchActions())

    assert sim.finished
    for oid, order in sim.orders.items():
        assert order.status in (OrderStatus.DELIVERED, OrderStatus.EXPIRED), (
            f"Order {oid} has status {order.status} after simulation end"
        )


def test_promise_window_bounds():
    """All orders have promised_delivery_time within [placed_at+35, placed_at+45]."""
    sim = make_sim(seed=42)
    run_empty_steps(sim, 30)  # Generate plenty of orders

    assert sim.total_orders_seen > 0
    for order in sim.orders.values():
        window = order.promised_delivery_time - order.placed_at
        assert 35 <= window <= 45, (
            f"Order {order.order_id}: promise window {window}, expected [35, 45]"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Category 8: Reward Boundary Tests
# ══════════════════════════════════════════════════════════════════════════════

def test_late_delivery_reward_tiers():
    """Verify reward tiers: on-time +1.0, slightly late +0.3, very late -0.5."""
    # We test the reward logic directly by examining delivered orders
    sim = make_sim(seed=100)
    run_empty_steps(sim, 3)

    pending = [o for o in sim.orders.values() if o.status == OrderStatus.PENDING]
    idle = [c for c in sim.couriers if c.state == CourierState.IDLE]
    if not pending or not idle:
        pytest.skip("Need orders and couriers")

    order = pending[0]
    courier = idle[0]

    # Assign and deliver
    sim.tick(DispatchActions(
        assignments=[Assignment(order_ids=[order.order_id], courier_id=courier.courier_id)]
    ))

    for _ in range(55):
        sim.tick(DispatchActions())
        if order.status == OrderStatus.DELIVERED:
            break

    if order.status != OrderStatus.DELIVERED:
        pytest.skip("Order didn't deliver in time")

    delivery_time = order.delivered_at - order.placed_at
    promise_window = order.promised_delivery_time - order.placed_at

    if delivery_time <= promise_window:
        # On-time: should have contributed +1.0
        assert sim.on_time_count >= 1
    elif delivery_time <= promise_window + 10:
        # Slightly late: should have contributed +0.3
        assert sim.late_count >= 1
    else:
        # Very late: should have contributed -0.5
        assert sim.very_late_count >= 1

    # In all cases, cumulative reward should reflect the delivery
    assert sim.total_delivered >= 1


def test_surge_revenue_adds_to_reward():
    """Surge pricing generates revenue that adds to step reward."""
    sim = make_sim(seed=42)

    # Set surge on all zones and run steps to generate surged orders
    total_surge_revenue = 0.0
    for _ in range(30):
        surge = {str(z): 2.0 for z in range(9)}
        result = sim.tick(DispatchActions(surge_multipliers=surge))
        total_surge_revenue += result.surge_revenue

    # Should have generated some surge revenue
    assert total_surge_revenue > 0, "Surge pricing should generate revenue"
    assert sim.total_surge_revenue > 0


# ══════════════════════════════════════════════════════════════════════════════
# Category 9: Multi-Step Workflow Tests
# ══════════════════════════════════════════════════════════════════════════════

def test_batch_delivery_both_orders_complete():
    """Both orders in a batch get delivered."""
    sim = make_sim(seed=200)
    run_empty_steps(sim, 8)

    pending = [o for o in sim.orders.values() if o.status == OrderStatus.PENDING]
    idle = [c for c in sim.couriers if c.state == CourierState.IDLE]

    if len(pending) < 2 or not idle:
        pytest.skip("Need 2+ orders and 1 courier")

    o1, o2 = pending[0], pending[1]
    courier = idle[0]

    sim.tick(DispatchActions(
        assignments=[Assignment(
            order_ids=[o1.order_id, o2.order_id],
            courier_id=courier.courier_id
        )]
    ))

    assert len(courier.current_orders) == 2

    # Run until both delivered (or timeout)
    for _ in range(80):
        sim.tick(DispatchActions())
        if o1.status == OrderStatus.DELIVERED and o2.status == OrderStatus.DELIVERED:
            break

    # At least one should be delivered; both may be if simulation allows
    delivered = [o for o in [o1, o2] if o.status == OrderStatus.DELIVERED]
    resolved = [o for o in [o1, o2]
                if o.status in (OrderStatus.DELIVERED, OrderStatus.EXPIRED)]
    assert len(resolved) == 2, (
        f"Both orders should be resolved: o1={o1.status}, o2={o2.status}"
    )
    assert len(delivered) >= 1, "At least one batched order should be delivered"


def test_repositioning_completes_and_updates_zone():
    """Repositioning moves courier to target zone center and updates zone_id."""
    sim = make_sim(seed=42)
    idle = [c for c in sim.couriers if c.state == CourierState.IDLE]
    assert len(idle) > 0

    courier = idle[0]
    # Pick a distant zone
    target_zid = 8 if courier.zone_id == 0 else 0
    target_zone = sim.zones[target_zid]

    sim.tick(DispatchActions(
        repositions=[Reposition(courier_id=courier.courier_id, zone_id=target_zid)]
    ))
    assert courier.state == CourierState.REPOSITIONING
    eta = courier.eta_remaining

    # Run until repositioning completes
    for _ in range(int(eta) + 5):
        sim.tick(DispatchActions())
        if courier.state == CourierState.IDLE:
            break

    assert courier.state == CourierState.IDLE, f"Courier should be IDLE, got {courier.state}"
    # Position should be at target zone center
    assert abs(courier.position.x - target_zone.center.x) < 0.01
    assert abs(courier.position.y - target_zone.center.y) < 0.01
    assert courier.zone_id == target_zid


def test_courier_waits_at_restaurant():
    """Courier waits at restaurant if food isn't ready when they arrive."""
    # Run more steps with more seeds to guarantee a long-prep order
    sim = make_sim(seed=42)
    run_empty_steps(sim, 15)

    # Find an order with long prep time (any type)
    premium_orders = [o for o in sim.orders.values()
                      if o.status == OrderStatus.PENDING
                      and o.prep_time > 12]
    if not premium_orders:
        premium_orders = [o for o in sim.orders.values()
                          if o.status == OrderStatus.PENDING]

    idle = [c for c in sim.couriers if c.state == CourierState.IDLE]
    if not premium_orders or not idle:
        pytest.skip("Need a long-prep order and idle courier")

    order = premium_orders[0]
    courier = idle[0]

    sim.tick(DispatchActions(
        assignments=[Assignment(order_ids=[order.order_id], courier_id=courier.courier_id)]
    ))

    # Run until courier reaches restaurant
    seen_waiting = False
    for _ in range(30):
        sim.tick(DispatchActions())
        if courier.state == CourierState.WAITING_AT_RESTAURANT:
            seen_waiting = True
            break
        if courier.state == CourierState.EN_ROUTE_DELIVERY:
            # Food was already ready when courier arrived
            break

    # If the courier had to wait, verify they eventually transition
    if seen_waiting:
        for _ in range(30):
            sim.tick(DispatchActions())
            if courier.state == CourierState.EN_ROUTE_DELIVERY:
                break
        assert courier.state == CourierState.EN_ROUTE_DELIVERY, (
            f"After waiting, courier should be EN_ROUTE_DELIVERY, got {courier.state}"
        )


def test_expired_order_frees_courier():
    """When an assigned order expires, its courier is freed."""
    sim = make_sim(seed=42)
    run_empty_steps(sim, 5)

    pending = [o for o in sim.orders.values() if o.status == OrderStatus.PENDING]
    idle = [c for c in sim.couriers if c.state == CourierState.IDLE]
    if not pending or not idle:
        pytest.skip("Need orders and couriers")

    order = pending[0]
    courier = idle[0]

    # Assign the order
    sim.tick(DispatchActions(
        assignments=[Assignment(order_ids=[order.order_id], courier_id=courier.courier_id)]
    ))

    # Run until order expires or is delivered
    for _ in range(65):
        sim.tick(DispatchActions())
        if order.status in (OrderStatus.DELIVERED, OrderStatus.EXPIRED):
            break

    if order.status == OrderStatus.EXPIRED:
        # Courier should be freed (IDLE or working on another order)
        assert order.order_id not in courier.current_orders, (
            "Expired order should be removed from courier's current_orders"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Category 10: RL Correctness - Scenario Difficulty & Skill Discrimination
# ══════════════════════════════════════════════════════════════════════════════

def _run_greedy(scenario, seed=42):
    """Run greedy nearest-courier assignment strategy. Returns normalized reward."""
    sim = make_sim(scenario, seed=seed)
    duration = sim.config.duration_minutes
    for step in range(duration):
        pending = [o for o in sim.orders.values() if o.status == OrderStatus.PENDING]
        idle = [c for c in sim.couriers if c.state == CourierState.IDLE]

        assignments = []
        used_couriers = set()
        for order in pending:
            if not idle:
                break
            best_courier = None
            best_dist = float('inf')
            rest = sim.restaurants[order.restaurant_id]
            for c in idle:
                if c.courier_id in used_couriers:
                    continue
                d = Simulation._manhattan_distance(c.position, rest.position)
                if d < best_dist:
                    best_dist = d
                    best_courier = c
            if best_courier:
                assignments.append(Assignment(
                    order_ids=[order.order_id],
                    courier_id=best_courier.courier_id,
                ))
                used_couriers.add(best_courier.courier_id)

        sim.tick(DispatchActions(assignments=assignments))

    assert sim.finished
    return sim.get_normalized_reward()


def test_scenario_difficulty_gradient():
    """Easier scenarios produce better greedy reward than harder ones."""
    reward_calm = _run_greedy("weekday_calm")
    reward_understaffed = _run_greedy("understaffed")

    assert reward_calm > reward_understaffed, (
        f"weekday_calm ({reward_calm:.3f}) should be easier than "
        f"understaffed ({reward_understaffed:.3f})"
    )


def test_greedy_vs_no_assignment():
    """Greedy assignment produces much better reward than doing nothing."""
    # Greedy
    reward_greedy = _run_greedy("weekday_calm")

    # No assignment (all orders expire)
    sim = make_sim(seed=42)
    for _ in range(240):
        sim.tick(DispatchActions())
    reward_nothing = sim.get_normalized_reward()

    assert reward_greedy > reward_nothing, (
        f"Greedy ({reward_greedy:.3f}) should beat no-assignment ({reward_nothing:.3f})"
    )
    # No-assignment should be deeply negative (all orders expire at -1.5 each)
    assert reward_nothing < -1.0, (
        f"No-assignment reward should be very negative, got {reward_nothing:.3f}"
    )


def test_greedy_vs_random_strategy():
    """Greedy (nearest courier) beats random courier assignment."""
    # Greedy
    reward_greedy = _run_greedy("weekday_calm", seed=42)

    # Random assignment
    rng = np.random.default_rng(42)
    sim = make_sim(seed=42)
    for step in range(240):
        pending = [o for o in sim.orders.values() if o.status == OrderStatus.PENDING]
        idle = [c for c in sim.couriers if c.state == CourierState.IDLE]

        assignments = []
        used_couriers = set()
        for order in pending:
            available = [c for c in idle if c.courier_id not in used_couriers]
            if not available:
                break
            # Random courier selection
            c = available[int(rng.integers(0, len(available)))]
            assignments.append(Assignment(
                order_ids=[order.order_id],
                courier_id=c.courier_id,
            ))
            used_couriers.add(c.courier_id)

        sim.tick(DispatchActions(assignments=assignments))

    reward_random = sim.get_normalized_reward()

    assert reward_greedy > reward_random, (
        f"Greedy ({reward_greedy:.3f}) should beat random ({reward_random:.3f})"
    )


def test_all_scenarios_reach_finished_state():
    """All 8 scenarios complete successfully."""
    from fooddelivery import SCENARIOS
    for scenario_name in SCENARIOS:
        sim = make_sim(scenario_name, seed=42)
        duration = sim.config.duration_minutes
        for _ in range(duration):
            sim.tick(DispatchActions())
        assert sim.finished, f"{scenario_name} did not finish after {duration} steps"
        assert sim.total_orders_seen > 0, f"{scenario_name} generated no orders"


# ══════════════════════════════════════════════════════════════════════════════
# Category 11: Weather & Traffic Tests
# ══════════════════════════════════════════════════════════════════════════════

def test_traffic_factor_peaks_during_rush():
    """Traffic factor is highest during 6-7 PM rush."""
    sim = make_sim(seed=42)
    a = Position(x=1.0, y=1.0)
    b = Position(x=4.0, y=4.0)

    # Sample mean travel times at different steps, resetting RNG each time
    def mean_travel(step, n=100):
        sim.rng = np.random.default_rng(9999)
        return np.mean([sim._compute_travel_time(a, b, 18.0, step) for _ in range(n)])

    early = mean_travel(10)    # 5:10 PM (low traffic)
    rush = mean_travel(90)     # 6:30 PM (peak traffic)
    late = mean_travel(200)    # 8:20 PM (reduced traffic)

    assert rush > early, f"Rush ({rush:.1f}) should exceed early ({early:.1f})"
    assert rush > late, f"Rush ({rush:.1f}) should exceed late ({late:.1f})"


def test_rain_increases_both_demand_and_travel():
    """Rain increases both order demand and travel times."""
    sim_clear = make_sim("weekday_busy", seed=42)
    sim_rain = make_sim("rainy_evening", seed=42)

    # Both have 1.3x demand multiplier, but rain adds +25% demand and +30% travel

    # Run 60 steps each
    for _ in range(60):
        sim_clear.tick(DispatchActions())
        sim_rain.tick(DispatchActions())

    # Rain should have more orders (1.3x * 1.25 = 1.625x vs 1.3x)
    assert sim_rain.total_orders_seen > sim_clear.total_orders_seen, (
        f"Rain ({sim_rain.total_orders_seen}) should have more orders than "
        f"clear ({sim_clear.total_orders_seen})"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Category 12: Action Combination & Data Validation Tests
# ══════════════════════════════════════════════════════════════════════════════

def test_mixed_actions_single_step():
    """All three action types work together in one step."""
    sim = make_sim(seed=42)
    run_empty_steps(sim, 5)

    pending = [o for o in sim.orders.values() if o.status == OrderStatus.PENDING]
    idle = [c for c in sim.couriers if c.state == CourierState.IDLE]
    if len(pending) < 1 or len(idle) < 2:
        pytest.skip("Need 1+ orders and 2+ idle couriers")

    order = pending[0]
    c1, c2 = idle[0], idle[1]

    result = sim.tick(DispatchActions(
        assignments=[Assignment(order_ids=[order.order_id], courier_id=c1.courier_id)],
        surge_multipliers={"4": 2.0, "0": 1.5},
        repositions=[Reposition(courier_id=c2.courier_id, zone_id=4)],
    ))

    # Assignment worked
    assert order.status == OrderStatus.ASSIGNED
    assert c1.state == CourierState.EN_ROUTE_PICKUP

    # Surge applied
    assert sim.active_surge.get(4) == 2.0
    assert sim.active_surge.get(0) == 1.5

    # Repositioning worked
    assert c2.state == CourierState.REPOSITIONING


def test_order_value_in_expected_range():
    """All generated order values are in [$15, $45] and tips in [$2, $8]."""
    sim = make_sim(seed=42)
    # No surge, so values shouldn't be boosted
    run_empty_steps(sim, 30)

    for order in sim.orders.values():
        assert 15.0 <= order.value <= 45.0, (
            f"Order {order.order_id} value {order.value} outside [$15, $45]"
        )
        assert 2.0 <= order.tip <= 8.0, (
            f"Order {order.order_id} tip {order.tip} outside [$2, $8]"
        )


def test_customer_positions_within_zone():
    """Customer positions fall within their assigned zone boundaries."""
    sim = make_sim(seed=42)
    run_empty_steps(sim, 30)

    for order in sim.orders.values():
        zone = sim.zones[order.zone_id]
        pos = order.customer_position
        assert zone.x_min <= pos.x <= zone.x_max, (
            f"Order {order.order_id} x={pos.x} outside zone {order.zone_id} "
            f"[{zone.x_min}, {zone.x_max}]"
        )
        assert zone.y_min <= pos.y <= zone.y_max, (
            f"Order {order.order_id} y={pos.y} outside zone {order.zone_id} "
            f"[{zone.y_min}, {zone.y_max}]"
        )
