"""
TOOL 2: VIABILITY CHECKER
Check if a proposed transit solution is economically viable.

Takes ridership + mode type and returns cost analysis + viability verdict.
All costs extracted from researched Indian transport benchmarks (2025 prices).
"""

from typing import Dict, Tuple


# ============================================================
# HARDCODED COSTS FROM RESEARCH (2025 Price Levels, INR Crores)
# ============================================================

TRANSPORT_COSTS = {
    "metro": {
        "capital_cost_per_km": {
            "elevated": 250,      # ₹150-300, using mid-high estimate
            "underground": 375,   # ₹250-500, using mid estimate
            "default": 250        # Assume elevated unless specified
        },
        "annual_opex_per_km": 10,       # ₹8-12 cr/km/year
        "train_cost": 50,               # ₹48-50 cr per 6-coach train
        "train_capacity": 2500,         # ~2,500 people per train
        "train_lifespan": 30,           # years
        "fare_per_trip": 50,            # ₹ (average metro fare)
        "peak_capacity_pphpd": 45000,   # persons per hour per direction
    },
    
    "brt": {
        "capital_cost_per_km": 35,      # ₹20-50, using mid estimate
        "annual_opex_per_km": 0.175,    # ₹0.15-0.2 cr/km/year
        "bus_cost": 0.75,               # ₹75 lakh per bus
        "bus_capacity": 100,            # people per bus
        "bus_lifespan": 10,             # years
        "fare_per_trip": 15,            # ₹ (average BRT fare)
        "peak_capacity_pphpd": 12000,   # persons per hour per direction
    },
    
    "cycling": {
        "capital_cost_per_km": 1.5,     # ₹0.3-3, using mid estimate (₹150 lakh)
        "annual_opex_per_km": 0.01,     # ₹5-10 lakh/km/year
        "fare_per_trip": 0,             # Free
        "peak_capacity_pphpd": 5000,    # persons per hour per direction (local trips)
    }
}

# Assumption: 12% of daily riders use during peak hour
PEAK_HOUR_SHARE = 0.12

# Assumption: Average corridor length assumption (if not provided)
DEFAULT_LIFECYCLE_YEARS = 30

# Inflation/risk buffer
CONTINGENCY_FACTOR = 1.05  # 5% contingency


def check_viability(
    mode: str,
    route_length_km: float,
    daily_ridership: float,
    peak_hour_ridership: float = None,
    alignment_type: str = "default"  # For metro: "elevated" or "underground"
) -> Dict:
    """
    Check economic viability of a proposed transit mode.
    
    Args:
        mode: "metro", "brt", or "cycling"
        route_length_km: Length of proposed corridor
        daily_ridership: Estimated daily ridership (from Tool 1)
        peak_hour_ridership: Peak hour riders (optional, calculated if not provided)
        alignment_type: For metro, specify "elevated" or "underground"
    
    Returns:
        Dict with:
        - capital_cost
        - annual_opex
        - annual_revenue
        - annual_net_cashflow
        - break_even_years
        - lifecycle_npv (simplified)
        - viability_verdict (VIABLE / MARGINAL / NOT VIABLE)
        - reasoning
    """
    
    mode = mode.lower()
    
    if mode not in TRANSPORT_COSTS:
        return {"ERROR": f"Invalid mode. Choose from: {list(TRANSPORT_COSTS.keys())}"}
    
    # Calculate peak hour if not provided
    if peak_hour_ridership is None:
        peak_hour_ridership = daily_ridership * PEAK_HOUR_SHARE
    
    # Get cost parameters
    costs = TRANSPORT_COSTS[mode]
    
    # ====== CAPITAL COST ======
    if mode == "metro":
        base_capex_per_km = costs["capital_cost_per_km"][alignment_type]
    else:
        base_capex_per_km = costs["capital_cost_per_km"]
    
    total_capex = route_length_km * base_capex_per_km * CONTINGENCY_FACTOR
    
    # Add vehicle/rolling stock cost
    if mode == "metro":
        # Peak hour ÷ capacity = trains needed
        trains_needed = int(peak_hour_ridership / costs["train_capacity"]) + 1
        vehicle_cost = trains_needed * costs["train_cost"]
        total_capex += vehicle_cost
        
    elif mode == "brt":
        # Assuming 5 buses per km for frequency (rough estimate)
        buses_needed = max(int(peak_hour_ridership / costs["bus_capacity"]), 5)
        vehicle_cost = buses_needed * costs["bus_cost"]
        total_capex += vehicle_cost
    
    # ====== ANNUAL OPERATING COSTS ======
    annual_opex = route_length_km * costs["annual_opex_per_km"]
    
    # ====== ANNUAL REVENUE ======
    # Calculate in rupees, then convert to crores
    if "fare_per_trip" in costs:
        annual_revenue_rupees = daily_ridership * costs["fare_per_trip"] * 365
        annual_revenue = annual_revenue_rupees / 10_000_000  # Convert to crores
    else:
        annual_revenue = 0
    
    # ====== ANNUAL NET CASHFLOW ======
    annual_net = annual_revenue - annual_opex
    
    # ====== BREAK-EVEN ANALYSIS ======
    if annual_net > 0:
        break_even_years = total_capex / annual_net
    else:
        break_even_years = float('inf')  # Never breaks even operationally
    
    # ====== VIABILITY VERDICT ======
    # Criteria: Can it break even within reasonable timeframe?
    # Metro/BRT: 7-15 years is acceptable (govt often subsidizes)
    # Cycling: Always viable (low cost)
    
    if mode == "cycling":
        verdict = "VIABLE"
        reasoning = "Cycling infrastructure: Low cost, quick payback, minimal O&M."
    
    elif mode == "metro":
        if break_even_years < 10:
            verdict = "VIABLE"
            reasoning = f"Strong ridership ({daily_ridership:,}/day). Break-even in {break_even_years:.1f} years."
        elif break_even_years < 20:
            verdict = "MARGINAL"
            reasoning = f"Acceptable ridership. Break-even in {break_even_years:.1f} years. Govt subsidy likely needed."
        else:
            verdict = "NOT VIABLE"
            reasoning = f"Ridership too low ({daily_ridership:,}/day) for metro cost. Break-even {break_even_years:.1f}+ years. Consider BRT instead."
    
    elif mode == "brt":
        if break_even_years < 8:
            verdict = "VIABLE"
            reasoning = f"Good ridership. Break-even in {break_even_years:.1f} years."
        elif break_even_years < 15:
            verdict = "MARGINAL"
            reasoning = f"Moderate ridership. Break-even in {break_even_years:.1f} years. Operational subsidy may be needed."
        else:
            verdict = "NOT VIABLE"
            reasoning = f"Ridership too low ({daily_ridership:,}/day). Consider smaller bus service or cycling."
    
    # ====== RETURN RESULTS ======
    return {
        "mode": mode,
        "route_length_km": route_length_km,
        "alignment_type": alignment_type,
        
        "financial_summary": {
            "capital_cost_crore": round(total_capex, 2),
            "annual_opex_crore": round(annual_opex, 2),
            "annual_revenue_crore": round(annual_revenue, 2),
            "annual_net_cashflow_crore": round(annual_net, 2),
        },
        
        "ridership_metrics": {
            "daily_ridership": float(daily_ridership),
            "peak_hour_ridership": float(peak_hour_ridership),
        },
        
        "viability_metrics": {
            "break_even_years": round(break_even_years, 1) if break_even_years != float('inf') else "Never (operational subsidy needed)",
            "annual_roi": round((annual_net / total_capex * 100), 2) if total_capex > 0 else 0,
        },
        
        "verdict": verdict,
        "reasoning": reasoning,
    }


# ============================================================
# EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":
    
    print("\n" + "="*70)
    print("TOOL 2: VIABILITY CHECKER - TEST SCENARIOS")
    print("="*70)
    
    # Scenario 1: Metro corridor with strong ridership
    print("\n[SCENARIO 1: Proposed Metro on High-Demand Corridor]")
    print("-" * 70)
    result1 = check_viability(
        mode="metro",
        route_length_km=15,
        daily_ridership=166667,
        alignment_type="elevated"
    )
    print(f"Mode: {result1['mode'].upper()}")
    print(f"Route Length: {result1['route_length_km']} km")
    print(f"Daily Ridership: {result1['ridership_metrics']['daily_ridership']:,}")
    print(f"Peak Hour: {result1['ridership_metrics']['peak_hour_ridership']:,}")
    print(f"\nFinancial Summary:")
    print(f"  Capital Cost: ₹{result1['financial_summary']['capital_cost_crore']} crore")
    print(f"  Annual OpEx: ₹{result1['financial_summary']['annual_opex_crore']} crore")
    print(f"  Annual Revenue: ₹{result1['financial_summary']['annual_revenue_crore']} crore")
    print(f"  Annual Net: ₹{result1['financial_summary']['annual_net_cashflow_crore']} crore")
    print(f"\nViability:")
    print(f"  Break-even: {result1['viability_metrics']['break_even_years']} years")
    print(f"  Annual ROI: {result1['viability_metrics']['annual_roi']}%")
    print(f"  Verdict: {result1['verdict']}")
    print(f"  Reasoning: {result1['reasoning']}")
    
    # Scenario 2: BRT on same corridor
    print("\n" + "="*70)
    print("[SCENARIO 2: Proposed BRT on Same Corridor]")
    print("-" * 70)
    result2 = check_viability(
        mode="brt",
        route_length_km=15,
        daily_ridership=166667
    )
    print(f"Mode: {result2['mode'].upper()}")
    print(f"Route Length: {result2['route_length_km']} km")
    print(f"Daily Ridership: {result2['ridership_metrics']['daily_ridership']:,}")
    print(f"Peak Hour: {result2['ridership_metrics']['peak_hour_ridership']:,}")
    print(f"\nFinancial Summary:")
    print(f"  Capital Cost: ₹{result2['financial_summary']['capital_cost_crore']} crore")
    print(f"  Annual OpEx: ₹{result2['financial_summary']['annual_opex_crore']} crore")
    print(f"  Annual Revenue: ₹{result2['financial_summary']['annual_revenue_crore']} crore")
    print(f"  Annual Net: ₹{result2['financial_summary']['annual_net_cashflow_crore']} crore")
    print(f"\nViability:")
    print(f"  Break-even: {result2['viability_metrics']['break_even_years']} years")
    print(f"  Annual ROI: {result2['viability_metrics']['annual_roi']}%")
    print(f"  Verdict: {result2['verdict']}")
    print(f"  Reasoning: {result2['reasoning']}")
    
    # Scenario 3: Metro with LOW ridership (should be NOT VIABLE)
    print("\n" + "="*70)
    print("[SCENARIO 3: Proposed Metro with LOW Ridership (Reality Check)]")
    print("-" * 70)
    result3 = check_viability(
        mode="metro",
        route_length_km=15,
        daily_ridership=30000.0  # Low demand
    )
    print(f"Mode: {result3['mode'].upper()}")
    print(f"Daily Ridership: {result3['ridership_metrics']['daily_ridership']:,}")
    print(f"\nFinancial Summary:")
    print(f"  Capital Cost: ₹{result3['financial_summary']['capital_cost_crore']} crore")
    print(f"  Annual Revenue: ₹{result3['financial_summary']['annual_revenue_crore']} crore")
    print(f"\nViability:")
    print(f"  Break-even: {result3['viability_metrics']['break_even_years']} years")
    print(f"  Verdict: {result3['verdict']}")
    print(f"  Reasoning: {result3['reasoning']}")
    
    # Scenario 4: Cycling (always viable)
    print("\n" + "="*70)
    print("[SCENARIO 4: Proposed Cycling Infrastructure (First/Last Mile)]")
    print("-" * 70)
    result4 = check_viability(
        mode="cycling",
        route_length_km=5,
        daily_ridership=10000.0
    )
    print(f"Mode: {result4['mode'].upper()}")
    print(f"Route Length: {result4['route_length_km']} km")
    print(f"Daily Ridership: {result4['ridership_metrics']['daily_ridership']:,}")
    print(f"\nFinancial Summary:")
    print(f"  Capital Cost: ₹{result4['financial_summary']['capital_cost_crore']} crore")
    print(f"  Annual OpEx: ₹{result4['financial_summary']['annual_opex_crore']} crore")
    print(f"  Annual Revenue: ₹{result4['financial_summary']['annual_revenue_crore']} crore")
    print(f"\nViability:")
    print(f"  Verdict: {result4['verdict']}")
    print(f"  Reasoning: {result4['reasoning']}")
    
    print("\n" + "="*70)
