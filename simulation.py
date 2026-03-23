"""
Food Delivery Dispatch Simulation Engine.

A hyper-realistic simulation of a food delivery platform with stochastic
order arrivals, travel times, and restaurant prep times. Supports courier-order
matching, order batching, surge pricing, and fleet repositioning.

Empirical Grounding and Citations
=================================
All core parameters are grounded in academic literature and industry data:

Courier speed (18 km/h, e-bike):
  - Gruber & Narayanan, "Travel time differences between cargo cycles and
    cars in commercial transport operations," Transportation Research Record, 2019.
    (~15.9 km/h for cargo bikes; lightweight food delivery e-bikes are faster)
  - Kale AI / Larry vs Harry, "Last Mile Delivery Study," 2022. (16 km/h avg for
    electric Bullitt cargo bikes in Brussels)

Restaurant prep times (fast_food=8, standard=18, premium=25 min):
  - Restaurant Times, "Restaurant Order Preparation Time -- Best Practices &
    Optimization Tips," 2024. (fast food <10 min, casual 12-20 min, fine 20-45 min)
  - CivicScience, "Average Fast Food Wait Time," 2023.

Gamma distribution for prep times (shape=4):
  - "Meal Delivery Routing Problem with Stochastic Meal Preparation Times and
    Customer Locations," Networks and Spatial Economics, 2024.
  - "Modeling stochastic service time for complex on-demand food delivery,"
    Complex & Intelligent Systems, 2022.
  Shape=4 represents 4 sequential exponential stages (order receipt, ingredient
  prep, cooking, plating/packaging).

Non-homogeneous Poisson process for order arrivals:
  - Liang et al., "A Poisson-Based Distribution Learning Framework for Short-Term
    Prediction of Food Delivery Demand Ranges," 2023.
  - Reyes et al., "The Meal Delivery Routing Problem," Optimization Online, 2018.

Dinner rush profile (peak 6:30-7:30 PM):
  - Uber Help, "Food delivery: best times and tips."
  - Gridwise, "The Best Times to DoorDash," 2024.

Surge pricing elasticity (10% demand drop per 1.0x increase):
  NOTE: This is intentionally conservative for gameplay balance. Empirical
  estimates are much higher:
  - MacKay, Svartback & Ekholm, "Dynamic Pricing, Intertemporal Spillovers, and
    Efficiency," HBS Working Paper 23-007, 2022. (delivery fee elasticity: -1.65)
  - Cohen et al., "Using Big Data to Estimate Consumer Surplus: The Case of Uber,"
    NBER Working Paper 22627, 2016.
  A realistic -1.65 elasticity would make surge pricing always counterproductive.
  The conservative value keeps surge as a viable but non-trivial optimization lever.

Lognormal stochastic travel times (sigma=0.15 off-peak, 0.25 peak):
  - Chen et al., "Exploring Travel Time Distribution and Variability Patterns Using
    Probe Vehicle Data," J. Advanced Transportation, 2018.
  - Frontiers in Built Environment, "Review on Statistical Modeling of Travel Time
    Variability for Road-Based Public Transport," 2020.

Order value ($15-$45) and delivery promise (35-45 min):
  - Business of Apps / DoorDash Statistics, 2024. (avg order ~$35-$37)
  - Intouch Insight, "DoorDash Tops Report on Delivery Performance," 2024.
    (avg delivery 33-38 min across platforms)
  - SeeLevel HX national study, 2023. (UberEATS avg 35 min 31 sec)

Order expiry thresholds (20 min unassigned, 60 min total):
  - DoorDash "Extreme Dasher Wait Time" trigger at ~15 min.
  - Ulmer et al., "The Restaurant Meal Delivery Problem," Transportation Science,
    55(1), 2021. (delivery expected "within an hour, much less if possible")

Traffic factor (1.3x during 6-7 PM peak):
  - FHWA, "Traffic Congestion and Reliability: Trends and Advanced Strategies for
    Congestion Mitigation." (Travel Time Index = 1.30 for many US urban areas)
  - Bureau of Transportation Statistics, "Travel Time Index."

Max batch size (2 orders per courier):
  - Intouch Insight study, 2023. (12% of orders batched; batched = 2 orders standard)
  - Uber/DoorDash documentation. (max 2-3 simultaneous; 2 is US norm)

Weather effects (rain: +25% demand, +30% travel time):
  - Bite Squad / Gallagher's Pizza data: +12-40% delivery demand in bad weather.
  - Tomorrow.io consumer survey: 49% more likely to order in bad weather.
  - FHWA Road Weather Management: light rain +12-20% travel time, heavy +40-50%.

Manhattan distance for urban grid routing:
  - Standard in operations research for grid-based cities.
  - Menger, "You Will Like Geometry," 1952. (taxicab geometry)
"""

import math
from enum import Enum
from typing import Any, Optional

import numpy as np
from pydantic import BaseModel


# ── Enums ──────────────────────────────────────────────────────────────────────

class ZoneType(str, Enum):
    DOWNTOWN_CORE = "downtown_core"
    COMMERCIAL = "commercial"
    RESIDENTIAL = "residential"
    SUBURBAN = "suburban"


class CourierState(str, Enum):
    IDLE = "idle"
    EN_ROUTE_PICKUP = "en_route_pickup"
    WAITING_AT_RESTAURANT = "waiting_at_restaurant"
    EN_ROUTE_DELIVERY = "en_route_delivery"
    REPOSITIONING = "repositioning"


class OrderStatus(str, Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    READY_FOR_PICKUP = "ready"
    PICKED_UP = "picked_up"
    DELIVERED = "delivered"
    EXPIRED = "expired"


# ── Data Models ────────────────────────────────────────────────────────────────

class Position(BaseModel):
    x: float
    y: float


class Zone(BaseModel):
    zone_id: int
    zone_type: ZoneType
    center: Position
    demand_multiplier: float
    x_min: float
    x_max: float
    y_min: float
    y_max: float


class Restaurant(BaseModel):
    restaurant_id: int
    name: str
    position: Position
    zone_id: int
    restaurant_type: str  # "fast_food", "standard", "premium"
    base_prep_time: float  # minutes


class Order(BaseModel):
    order_id: int
    restaurant_id: int
    customer_position: Position
    zone_id: int
    value: float
    tip: float
    prep_time: float
    placed_at: int
    promised_delivery_time: int
    status: OrderStatus = OrderStatus.PENDING
    assigned_courier_id: Optional[int] = None
    ready_at: Optional[int] = None
    picked_up_at: Optional[int] = None
    delivered_at: Optional[int] = None
    expired_at: Optional[int] = None


class Courier(BaseModel):
    courier_id: int
    position: Position
    zone_id: int
    state: CourierState = CourierState.IDLE
    speed: float  # km/h
    current_orders: list[int] = []
    target_position: Optional[Position] = None
    target_order_id: Optional[int] = None
    eta_remaining: float = 0.0
    total_deliveries: int = 0
    total_earnings: float = 0.0


# ── Configuration Models ───────────────────────────────────────────────────────

class WeatherConfig(BaseModel):
    name: str = "clear"
    demand_boost: float = 0.0
    travel_time_boost: float = 0.0


class ScenarioConfig(BaseModel):
    scenario_name: str
    seed: int
    demand_multiplier: float
    num_couriers: int
    weather: WeatherConfig = WeatherConfig()
    duration_minutes: int = 240
    demand_spike: Optional[tuple[int, float]] = None
    demand_decline_start: Optional[int] = None
    demand_decline_rate: float = 0.0


# ── Tool Input Models (extra="forbid") ─────────────────────────────────────────

class Assignment(BaseModel, extra="forbid"):
    order_ids: list[int]
    courier_id: int


class Reposition(BaseModel, extra="forbid"):
    courier_id: int
    zone_id: int


class DispatchActions(BaseModel, extra="forbid"):
    assignments: list[Assignment] = []
    surge_multipliers: dict[str, float] = {}
    repositions: list[Reposition] = []


class GetInfoParams(BaseModel, extra="forbid"):
    pass


# ── Step Result ────────────────────────────────────────────────────────────────

class StepResult(BaseModel):
    step: int
    time_str: str
    finished: bool
    pending_orders: list[dict[str, Any]]
    active_orders: list[dict[str, Any]]
    couriers: list[dict[str, Any]]
    new_orders: list[dict[str, Any]]
    deliveries_completed: list[dict[str, Any]]
    orders_expired: list[dict[str, Any]]
    step_reward: float
    cumulative_reward: float
    total_orders_seen: int
    total_delivered: int
    total_expired: int
    on_time_count: int
    late_count: int
    avg_delivery_time: float
    surge_revenue: float


# ── Zone/Restaurant Configuration ──────────────────────────────────────────────

# Zone layout (3x3 grid, index = row * 3 + col):
#   0(sub) 1(com) 2(res)
#   3(com) 4(dtc) 5(com)
#   6(res) 7(com) 8(sub)
ZONE_CONFIG = {
    0: (ZoneType.SUBURBAN, 0.7),
    1: (ZoneType.COMMERCIAL, 1.5),
    2: (ZoneType.RESIDENTIAL, 1.0),
    3: (ZoneType.COMMERCIAL, 1.5),
    4: (ZoneType.DOWNTOWN_CORE, 2.0),
    5: (ZoneType.COMMERCIAL, 1.5),
    6: (ZoneType.RESIDENTIAL, 1.0),
    7: (ZoneType.COMMERCIAL, 1.5),
    8: (ZoneType.SUBURBAN, 0.7),
}

RESTAURANT_TYPE_PREP = {
    "fast_food": 8.0,
    "standard": 18.0,
    "premium": 25.0,
}

# Distribution of restaurant types by zone type
RESTAURANT_TYPE_DISTRIBUTION = {
    ZoneType.DOWNTOWN_CORE: [("fast_food", 0.2), ("standard", 0.4), ("premium", 0.4)],
    ZoneType.COMMERCIAL: [("fast_food", 0.4), ("standard", 0.3), ("premium", 0.3)],
    ZoneType.RESIDENTIAL: [("fast_food", 0.5), ("standard", 0.3), ("premium", 0.2)],
    ZoneType.SUBURBAN: [("fast_food", 0.5), ("standard", 0.3), ("premium", 0.2)],
}

CUISINE_NAMES = {
    "fast_food": ["Burger Barn", "Quick Bites", "Wrap & Roll", "Chicken Shack",
                  "Pizza Express", "Taco Town", "Noodle Box", "Fry Station",
                  "Sub Hub", "Wings Place"],
    "standard": ["Jade Garden", "Pasta House", "Curry Corner", "Grill & Co",
                 "Seaside Kitchen", "The Bistro", "Tandoori Flame", "Pho Bowl",
                 "Olive Branch", "Blue Plate"],
    "premium": ["Sushi Palace", "Le Petit Chef", "Steakhouse Prime", "Truffle & Thyme",
                "Ocean Grill", "The Wagyu Room", "Michelin Star", "Gold Leaf",
                "Artisan Table", "Velvet Fork"],
}


# ── Simulation Engine ──────────────────────────────────────────────────────────

class Simulation:
    """Core food delivery dispatch simulation."""

    def __init__(self, config: ScenarioConfig) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.step_num = 0
        self.finished = False

        # City
        self.zones: list[Zone] = []
        self.restaurants: list[Restaurant] = []
        self.couriers: list[Courier] = []

        # Orders
        self.orders: dict[int, Order] = {}
        self.next_order_id = 0

        # Metrics
        self.total_orders_seen = 0
        self.total_delivered = 0
        self.total_expired = 0
        self.on_time_count = 0
        self.late_count = 0
        self.very_late_count = 0
        self.delivery_times: list[float] = []
        self.cumulative_reward = 0.0
        self.step_rewards: list[float] = []

        # Surge state (reset each tick)
        self.active_surge: dict[int, float] = {}
        self.total_surge_revenue = 0.0

        # Name counters for restaurants
        self._name_counters: dict[str, int] = {"fast_food": 0, "standard": 0, "premium": 0}

        # Build city
        self._generate_city()
        self._place_couriers()

    # ── City Generation ────────────────────────────────────────────────────

    def _generate_city(self) -> None:
        """Generate 9 zones and 30 restaurants."""
        # Create zones
        for zone_id in range(9):
            row = zone_id // 3
            col = zone_id % 3
            zone_type, demand_mult = ZONE_CONFIG[zone_id]
            x_min = col * 2.0
            x_max = (col + 1) * 2.0
            y_min = row * 2.0
            y_max = (row + 1) * 2.0
            self.zones.append(Zone(
                zone_id=zone_id,
                zone_type=zone_type,
                center=Position(x=x_min + 1.0, y=y_min + 1.0),
                demand_multiplier=demand_mult,
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
            ))

        # Distribute 30 restaurants proportional to demand
        total_weight = sum(z.demand_multiplier for z in self.zones)
        restaurant_counts: list[int] = []
        remainder_pool: list[tuple[float, int]] = []

        for z in self.zones:
            raw = 30.0 * z.demand_multiplier / total_weight
            base = int(raw)
            frac = raw - base
            restaurant_counts.append(base)
            remainder_pool.append((frac, z.zone_id))

        # Distribute remaining restaurants by largest fractional part
        remaining = 30 - sum(restaurant_counts)
        remainder_pool.sort(key=lambda x: -x[0])
        for i in range(remaining):
            zone_idx = remainder_pool[i][1]
            restaurant_counts[zone_idx] += 1

        # Create restaurants
        rid = 0
        for zone in self.zones:
            count = restaurant_counts[zone.zone_id]
            type_dist = RESTAURANT_TYPE_DISTRIBUTION[zone.zone_type]
            types = [t for t, _ in type_dist]
            probs = [p for _, p in type_dist]

            for _ in range(count):
                rtype = self.rng.choice(types, p=probs)
                name_idx = self._name_counters[rtype] % len(CUISINE_NAMES[rtype])
                name = CUISINE_NAMES[rtype][name_idx]
                self._name_counters[rtype] += 1

                pos = Position(
                    x=float(self.rng.uniform(zone.x_min + 0.1, zone.x_max - 0.1)),
                    y=float(self.rng.uniform(zone.y_min + 0.1, zone.y_max - 0.1)),
                )
                self.restaurants.append(Restaurant(
                    restaurant_id=rid,
                    name=name,
                    position=pos,
                    zone_id=zone.zone_id,
                    restaurant_type=rtype,
                    base_prep_time=RESTAURANT_TYPE_PREP[rtype],
                ))
                rid += 1

    def _place_couriers(self) -> None:
        """Distribute couriers across zones proportional to demand."""
        total_weight = sum(z.demand_multiplier for z in self.zones)
        courier_counts: list[int] = []
        remainder_pool: list[tuple[float, int]] = []

        for z in self.zones:
            raw = self.config.num_couriers * z.demand_multiplier / total_weight
            base = int(raw)
            frac = raw - base
            courier_counts.append(base)
            remainder_pool.append((frac, z.zone_id))

        remaining = self.config.num_couriers - sum(courier_counts)
        remainder_pool.sort(key=lambda x: -x[0])
        for i in range(remaining):
            zone_idx = remainder_pool[i][1]
            courier_counts[zone_idx] += 1

        cid = 0
        for zone in self.zones:
            count = courier_counts[zone.zone_id]
            for _ in range(count):
                speed = float(18.0 * np.exp(self.rng.normal(0, 0.1)))
                pos = Position(
                    x=float(self.rng.uniform(zone.x_min + 0.1, zone.x_max - 0.1)),
                    y=float(self.rng.uniform(zone.y_min + 0.1, zone.y_max - 0.1)),
                )
                self.couriers.append(Courier(
                    courier_id=cid,
                    position=pos,
                    zone_id=zone.zone_id,
                    state=CourierState.IDLE,
                    speed=speed,
                ))
                cid += 1

    # ── Spatial Helpers ────────────────────────────────────────────────────

    def _get_zone_for_position(self, pos: Position) -> int:
        col = min(2, max(0, int(pos.x / 2.0)))
        row = min(2, max(0, int(pos.y / 2.0)))
        return row * 3 + col

    @staticmethod
    def _manhattan_distance(a: Position, b: Position) -> float:
        return abs(a.x - b.x) + abs(a.y - b.y)

    # ── Travel Time ───────────────────────────────────────────────────────

    def _compute_travel_time(self, from_pos: Position, to_pos: Position,
                             courier_speed: float, step: int) -> float:
        """Compute stochastic travel time in minutes."""
        dist = self._manhattan_distance(from_pos, to_pos)
        if dist < 0.01:
            return 1.0  # minimum 1 minute even for very close locations

        # Traffic factor by time of day
        if step < 60:
            traffic = 1.0 + 0.2 * (step / 60.0)
        elif step < 120:
            traffic = 1.3
        elif step < 180:
            traffic = 1.3 - 0.2 * ((step - 120) / 60.0)
        else:
            traffic = 1.1

        # Weather boost
        traffic *= (1.0 + self.config.weather.travel_time_boost)

        # Stochastic noise (lognormal)
        sigma = 0.25 if 60 <= step <= 150 else 0.15
        noise = float(self.rng.lognormal(0, sigma))

        # Base time: distance / speed * 60 (hours -> minutes)
        base_time = (dist / courier_speed) * 60.0

        travel_time = base_time * traffic * noise

        # Clamp to reasonable range
        return max(1.0, min(120.0, travel_time))

    # ── Order Generation ───────────────────────────────────────────────────

    def _base_arrival_rate(self, step: int) -> float:
        """Base orders per minute across entire city."""
        if step < 60:       # 5:00-6:00 PM ramp-up
            return 0.3 + 0.7 * (step / 60.0)
        elif step < 90:     # 6:00-6:30 PM accelerating
            return 1.0 + 0.5 * ((step - 60) / 30.0)
        elif step < 150:    # 6:30-7:30 PM peak
            return 1.5
        elif step < 180:    # 7:30-8:00 PM declining
            return 1.5 - 0.5 * ((step - 150) / 30.0)
        else:               # 8:00-9:00 PM tapering
            return 1.0 - 0.5 * ((step - 180) / 60.0)

    def _generate_orders(self) -> list[Order]:
        """Generate new orders for this step using NHPP."""
        base_rate = self._base_arrival_rate(self.step_num)
        effective_rate = base_rate * self.config.demand_multiplier
        effective_rate *= (1.0 + self.config.weather.demand_boost)

        # Demand spike
        if self.config.demand_spike and self.step_num >= self.config.demand_spike[0]:
            effective_rate *= self.config.demand_spike[1]

        # Demand decline
        if (self.config.demand_decline_start is not None
                and self.step_num >= self.config.demand_decline_start):
            elapsed = self.step_num - self.config.demand_decline_start
            factor = max(0.1, 1.0 - self.config.demand_decline_rate * elapsed)
            effective_rate *= factor

        total_demand_weight = sum(z.demand_multiplier for z in self.zones)
        new_orders: list[Order] = []

        for zone in self.zones:
            zone_rate = effective_rate * zone.demand_multiplier / total_demand_weight

            # Surge elasticity: reduce demand in surged zones
            surge_mult = self.active_surge.get(zone.zone_id, 1.0)
            if surge_mult > 1.0:
                elasticity = max(0.5, 1.0 - 0.1 * (surge_mult - 1.0))
                zone_rate *= elasticity

            num_orders = int(self.rng.poisson(zone_rate))

            # Find restaurants in this zone
            zone_restaurants = [r for r in self.restaurants if r.zone_id == zone.zone_id]
            if not zone_restaurants:
                # Fall back to nearest zone restaurants
                zone_restaurants = self.restaurants[:3]

            for _ in range(num_orders):
                restaurant = zone_restaurants[int(self.rng.integers(0, len(zone_restaurants)))]

                # Customer location within zone
                cust_pos = Position(
                    x=float(self.rng.uniform(zone.x_min + 0.05, zone.x_max - 0.05)),
                    y=float(self.rng.uniform(zone.y_min + 0.05, zone.y_max - 0.05)),
                )

                # Prep time from Gamma distribution
                shape = 4.0
                scale = restaurant.base_prep_time / shape
                prep_time = float(self.rng.gamma(shape, scale))
                prep_time = max(3.0, min(60.0, prep_time))  # clamp

                # Order value and tip
                value = float(self.rng.uniform(15.0, 45.0))
                tip = float(self.rng.uniform(2.0, 8.0))

                # Surge increases value
                if surge_mult > 1.0:
                    value *= surge_mult

                # Promised delivery time: 35-45 minutes from now
                promise = self.step_num + int(self.rng.integers(35, 46))

                order = Order(
                    order_id=self.next_order_id,
                    restaurant_id=restaurant.restaurant_id,
                    customer_position=cust_pos,
                    zone_id=zone.zone_id,
                    value=round(value, 2),
                    tip=round(tip, 2),
                    prep_time=round(prep_time, 1),
                    placed_at=self.step_num,
                    promised_delivery_time=promise,
                    status=OrderStatus.PENDING,
                    ready_at=self.step_num + int(math.ceil(prep_time)),
                )
                self.orders[self.next_order_id] = order
                new_orders.append(order)
                self.next_order_id += 1
                self.total_orders_seen += 1

        return new_orders

    # ── Main Tick ──────────────────────────────────────────────────────────

    def tick(self, actions: DispatchActions) -> StepResult:
        """Process one simulation step. Returns the result after advancing."""
        if self.finished:
            return self._build_result(
                new_orders=[], deliveries=[], expired=[],
                step_reward=0.0, surge_revenue=0.0,
            )

        # Phase 1: Reset surge (agent must re-set each step)
        self.active_surge = {}

        # Phase 2: Apply surge multipliers
        surge_revenue = 0.0
        for zone_id_str, mult in actions.surge_multipliers.items():
            try:
                zid = int(zone_id_str)
            except (ValueError, TypeError):
                continue
            if 0 <= zid <= 8:
                self.active_surge[zid] = max(1.0, min(3.0, mult))

        # Phase 3: Process assignments
        for assignment in actions.assignments:
            cid = assignment.courier_id
            if cid < 0 or cid >= len(self.couriers):
                continue
            courier = self.couriers[cid]

            # Courier must be IDLE to accept new assignments
            if courier.state != CourierState.IDLE:
                continue

            valid_order_ids: list[int] = []
            for oid in assignment.order_ids:
                if oid not in self.orders:
                    continue
                order = self.orders[oid]
                if order.status != OrderStatus.PENDING:
                    continue
                if len(courier.current_orders) + len(valid_order_ids) >= 2:
                    break
                valid_order_ids.append(oid)

            if not valid_order_ids:
                continue

            # Assign orders to courier
            for oid in valid_order_ids:
                order = self.orders[oid]
                order.status = OrderStatus.ASSIGNED
                order.assigned_courier_id = courier.courier_id
                courier.current_orders.append(oid)

            # Route courier to first order's restaurant
            first_order = self.orders[courier.current_orders[0]]
            restaurant = self.restaurants[first_order.restaurant_id]
            courier.state = CourierState.EN_ROUTE_PICKUP
            courier.target_position = restaurant.position
            courier.target_order_id = first_order.order_id
            courier.eta_remaining = self._compute_travel_time(
                courier.position, restaurant.position,
                courier.speed, self.step_num,
            )

        # Phase 4: Process repositions
        for reposition in actions.repositions:
            cid = reposition.courier_id
            zid = reposition.zone_id
            if cid < 0 or cid >= len(self.couriers):
                continue
            if zid < 0 or zid > 8:
                continue
            courier = self.couriers[cid]
            if courier.state != CourierState.IDLE:
                continue
            target_zone = self.zones[zid]
            courier.state = CourierState.REPOSITIONING
            courier.target_position = target_zone.center
            courier.eta_remaining = self._compute_travel_time(
                courier.position, target_zone.center,
                courier.speed, self.step_num,
            )

        # Phase 5: Generate new orders
        new_orders = self._generate_orders()

        # Calculate surge revenue from new orders in surged zones
        for order in new_orders:
            surge_mult = self.active_surge.get(order.zone_id, 1.0)
            if surge_mult > 1.0:
                # Platform gets 10% of the price increase
                base_value = order.value / surge_mult
                surge_revenue += (order.value - base_value) * 0.1

        # Phase 6: Advance couriers
        deliveries_completed: list[Order] = []
        for courier in self.couriers:
            if courier.state == CourierState.IDLE:
                continue

            courier.eta_remaining -= 1.0

            if courier.eta_remaining <= 0:
                courier.eta_remaining = 0.0
                if courier.target_position is not None:
                    courier.position = courier.target_position.model_copy()

                if courier.state == CourierState.EN_ROUTE_PICKUP:
                    self._handle_arrival_at_restaurant(courier)

                elif courier.state == CourierState.EN_ROUTE_DELIVERY:
                    delivered = self._handle_arrival_at_customer(courier)
                    if delivered:
                        deliveries_completed.append(delivered)
                        # Check for next batched order
                        self._route_to_next_order(courier)

                elif courier.state == CourierState.REPOSITIONING:
                    courier.state = CourierState.IDLE
                    courier.zone_id = self._get_zone_for_position(courier.position)
                    courier.target_position = None

            # Handle waiting couriers: check if food is ready
            if courier.state == CourierState.WAITING_AT_RESTAURANT:
                self._check_food_ready(courier)

        # Phase 7: Update order states and expiration
        orders_expired: list[Order] = []
        for order in list(self.orders.values()):
            if order.status in (OrderStatus.DELIVERED, OrderStatus.EXPIRED):
                continue

            time_since_placed = self.step_num - order.placed_at

            # Expiry: unassigned > 20 min
            if order.status == OrderStatus.PENDING and time_since_placed > 20:
                self._expire_order(order, orders_expired)
                continue

            # Expiry: any non-delivered order > 60 min
            if time_since_placed > 60:
                self._expire_order(order, orders_expired)
                continue

        # Phase 8: Compute step reward
        step_reward = 0.0
        for order in deliveries_completed:
            delivery_time = order.delivered_at - order.placed_at
            promise_window = order.promised_delivery_time - order.placed_at
            if delivery_time <= promise_window:
                step_reward += 1.0
                self.on_time_count += 1
            elif delivery_time <= promise_window + 10:
                step_reward += 0.3
                self.late_count += 1
            else:
                step_reward -= 0.5
                self.very_late_count += 1
            self.total_delivered += 1
            self.delivery_times.append(float(delivery_time))

        for _ in orders_expired:
            step_reward -= 1.5
            self.total_expired += 1

        # Surge revenue bonus
        step_reward += surge_revenue * 0.01
        self.total_surge_revenue += surge_revenue

        self.cumulative_reward += step_reward
        self.step_rewards.append(step_reward)

        # Phase 9: Advance step, check termination
        self.step_num += 1
        if self.step_num >= self.config.duration_minutes:
            self.finished = True
            # Expire all remaining non-delivered orders
            for order in list(self.orders.values()):
                if order.status not in (OrderStatus.DELIVERED, OrderStatus.EXPIRED):
                    order.status = OrderStatus.EXPIRED
                    order.expired_at = self.step_num
                    self.total_expired += 1
                    step_reward -= 1.5
            self.cumulative_reward += (step_reward - self.step_rewards[-1])
            self.step_rewards[-1] = step_reward

        return self._build_result(
            new_orders=new_orders,
            deliveries=deliveries_completed,
            expired=orders_expired,
            step_reward=step_reward,
            surge_revenue=surge_revenue,
        )

    # ── Tick Helpers ───────────────────────────────────────────────────────

    def _handle_arrival_at_restaurant(self, courier: Courier) -> None:
        """Handle courier arriving at restaurant."""
        if courier.target_order_id is None:
            courier.state = CourierState.IDLE
            return
        order = self.orders.get(courier.target_order_id)
        if order is None or order.status == OrderStatus.EXPIRED:
            courier.current_orders = [
                oid for oid in courier.current_orders
                if oid != courier.target_order_id
            ]
            courier.target_order_id = None
            self._route_to_next_order(courier)
            return

        if order.ready_at is not None and order.ready_at <= self.step_num:
            # Food is ready, pick it up
            order.status = OrderStatus.PICKED_UP
            order.picked_up_at = self.step_num
            courier.state = CourierState.EN_ROUTE_DELIVERY
            courier.target_position = order.customer_position
            courier.eta_remaining = self._compute_travel_time(
                courier.position, order.customer_position,
                courier.speed, self.step_num,
            )
        else:
            courier.state = CourierState.WAITING_AT_RESTAURANT

    def _check_food_ready(self, courier: Courier) -> None:
        """Check if the food is ready for a waiting courier."""
        if courier.target_order_id is None:
            courier.state = CourierState.IDLE
            return
        order = self.orders.get(courier.target_order_id)
        if order is None or order.status == OrderStatus.EXPIRED:
            courier.current_orders = [
                oid for oid in courier.current_orders
                if oid != courier.target_order_id
            ]
            courier.target_order_id = None
            self._route_to_next_order(courier)
            return

        if order.ready_at is not None and order.ready_at <= self.step_num:
            order.status = OrderStatus.PICKED_UP
            order.picked_up_at = self.step_num
            courier.state = CourierState.EN_ROUTE_DELIVERY
            courier.target_position = order.customer_position
            courier.eta_remaining = self._compute_travel_time(
                courier.position, order.customer_position,
                courier.speed, self.step_num,
            )

    def _handle_arrival_at_customer(self, courier: Courier) -> Optional[Order]:
        """Handle courier arriving at customer. Returns delivered order or None."""
        if courier.target_order_id is None:
            return None
        order = self.orders.get(courier.target_order_id)
        if order is None:
            return None
        if order.status == OrderStatus.EXPIRED:
            courier.current_orders = [
                oid for oid in courier.current_orders if oid != order.order_id
            ]
            return None

        order.status = OrderStatus.DELIVERED
        order.delivered_at = self.step_num
        courier.current_orders = [
            oid for oid in courier.current_orders if oid != order.order_id
        ]
        courier.total_deliveries += 1
        courier.total_earnings += order.value + order.tip
        courier.target_order_id = None
        return order

    def _route_to_next_order(self, courier: Courier) -> None:
        """Route courier to their next batched order, or set IDLE."""
        # Remove any expired orders from current_orders
        courier.current_orders = [
            oid for oid in courier.current_orders
            if oid in self.orders and self.orders[oid].status not in
            (OrderStatus.DELIVERED, OrderStatus.EXPIRED)
        ]

        if not courier.current_orders:
            courier.state = CourierState.IDLE
            courier.target_position = None
            courier.target_order_id = None
            courier.zone_id = self._get_zone_for_position(courier.position)
            return

        next_oid = courier.current_orders[0]
        next_order = self.orders[next_oid]
        courier.target_order_id = next_oid

        if next_order.status == OrderStatus.PICKED_UP:
            # Already picked up, deliver
            courier.state = CourierState.EN_ROUTE_DELIVERY
            courier.target_position = next_order.customer_position
        elif next_order.ready_at is not None and next_order.ready_at <= self.step_num:
            # Food ready at same restaurant, pick up and deliver
            next_order.status = OrderStatus.PICKED_UP
            next_order.picked_up_at = self.step_num
            courier.state = CourierState.EN_ROUTE_DELIVERY
            courier.target_position = next_order.customer_position
        else:
            # Need to go to restaurant (different restaurant for batch)
            restaurant = self.restaurants[next_order.restaurant_id]
            courier.state = CourierState.EN_ROUTE_PICKUP
            courier.target_position = restaurant.position

        courier.eta_remaining = self._compute_travel_time(
            courier.position, courier.target_position,
            courier.speed, self.step_num,
        )

    def _expire_order(self, order: Order, expired_list: list[Order]) -> None:
        """Expire an order and free its courier if assigned."""
        order.status = OrderStatus.EXPIRED
        order.expired_at = self.step_num
        expired_list.append(order)

        # Free up courier
        if order.assigned_courier_id is not None:
            courier = self.couriers[order.assigned_courier_id]
            courier.current_orders = [
                oid for oid in courier.current_orders if oid != order.order_id
            ]
            if courier.target_order_id == order.order_id:
                courier.target_order_id = None
                self._route_to_next_order(courier)

    # ── State Serialization ────────────────────────────────────────────────

    def _step_to_time_str(self, step: int) -> str:
        """Convert step number to time string (5:00 PM + step minutes)."""
        total_minutes = 17 * 60 + step  # 5 PM = 17:00
        hour = total_minutes // 60
        minute = total_minutes % 60
        period = "PM" if hour >= 12 else "AM"
        display_hour = hour if hour <= 12 else hour - 12
        if display_hour == 0:
            display_hour = 12
        return f"{display_hour}:{minute:02d} {period}"

    def _order_to_dict(self, order: Order) -> dict[str, Any]:
        """Serialize order for step result."""
        restaurant = self.restaurants[order.restaurant_id]
        return {
            "order_id": order.order_id,
            "restaurant_id": order.restaurant_id,
            "restaurant_name": restaurant.name,
            "restaurant_position": {"x": round(restaurant.position.x, 2),
                                    "y": round(restaurant.position.y, 2)},
            "restaurant_zone": restaurant.zone_id,
            "customer_position": {"x": round(order.customer_position.x, 2),
                                  "y": round(order.customer_position.y, 2)},
            "customer_zone": order.zone_id,
            "value": order.value,
            "tip": order.tip,
            "placed_at": order.placed_at,
            "promised_delivery_time": order.promised_delivery_time,
            "status": order.status.value,
            "assigned_courier_id": order.assigned_courier_id,
            "minutes_waiting": self.step_num - order.placed_at,
        }

    def _courier_to_dict(self, courier: Courier) -> dict[str, Any]:
        """Serialize courier for step result."""
        return {
            "courier_id": courier.courier_id,
            "position": {"x": round(courier.position.x, 2),
                         "y": round(courier.position.y, 2)},
            "zone_id": courier.zone_id,
            "state": courier.state.value,
            "speed": round(courier.speed, 1),
            "current_orders": list(courier.current_orders),
            "eta_remaining": round(courier.eta_remaining, 1),
            "total_deliveries": courier.total_deliveries,
        }

    def _build_result(self, new_orders: list[Order],
                      deliveries: list[Order], expired: list[Order],
                      step_reward: float, surge_revenue: float) -> StepResult:
        """Build the step result with current state."""
        pending = [self._order_to_dict(o) for o in self.orders.values()
                   if o.status == OrderStatus.PENDING]
        active = [self._order_to_dict(o) for o in self.orders.values()
                  if o.status in (OrderStatus.ASSIGNED, OrderStatus.READY_FOR_PICKUP,
                                  OrderStatus.PICKED_UP)]
        avg_dt = (sum(self.delivery_times) / len(self.delivery_times)
                  if self.delivery_times else 0.0)

        return StepResult(
            step=self.step_num,
            time_str=self._step_to_time_str(self.step_num),
            finished=self.finished,
            pending_orders=pending,
            active_orders=active,
            couriers=[self._courier_to_dict(c) for c in self.couriers],
            new_orders=[self._order_to_dict(o) for o in new_orders],
            deliveries_completed=[self._order_to_dict(o) for o in deliveries],
            orders_expired=[self._order_to_dict(o) for o in expired],
            step_reward=round(step_reward, 4),
            cumulative_reward=round(self.cumulative_reward, 4),
            total_orders_seen=self.total_orders_seen,
            total_delivered=self.total_delivered,
            total_expired=self.total_expired,
            on_time_count=self.on_time_count,
            late_count=self.late_count,
            avg_delivery_time=round(avg_dt, 1),
            surge_revenue=round(surge_revenue, 4),
        )

    def get_city_info(self) -> dict[str, Any]:
        """Return static city information for the get_info tool."""
        return {
            "city_size_km": "6x6",
            "num_zones": 9,
            "zones": [
                {
                    "zone_id": z.zone_id,
                    "type": z.zone_type.value,
                    "center": {"x": z.center.x, "y": z.center.y},
                    "demand_multiplier": z.demand_multiplier,
                    "bounds": {"x_min": z.x_min, "x_max": z.x_max,
                               "y_min": z.y_min, "y_max": z.y_max},
                }
                for z in self.zones
            ],
            "num_restaurants": len(self.restaurants),
            "restaurants": [
                {
                    "id": r.restaurant_id,
                    "name": r.name,
                    "type": r.restaurant_type,
                    "zone_id": r.zone_id,
                    "position": {"x": round(r.position.x, 2),
                                 "y": round(r.position.y, 2)},
                    "avg_prep_time_min": r.base_prep_time,
                }
                for r in self.restaurants
            ],
            "num_couriers": len(self.couriers),
            "couriers": [
                {
                    "id": c.courier_id,
                    "zone_id": c.zone_id,
                    "speed_kmh": round(c.speed, 1),
                    "position": {"x": round(c.position.x, 2),
                                 "y": round(c.position.y, 2)},
                }
                for c in self.couriers
            ],
            "max_batch_size": 2,
            "courier_vehicle": "e-bike (~18 km/h avg)",
            "scenario": self.config.scenario_name,
            "duration_minutes": self.config.duration_minutes,
            "weather": self.config.weather.name,
        }

    def get_normalized_reward(self) -> float:
        """Final normalized reward = cumulative / total_orders_seen."""
        if self.total_orders_seen == 0:
            return 0.0
        return self.cumulative_reward / self.total_orders_seen
