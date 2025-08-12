#!/usr/bin/env python3
"""
EventBridge Cost Estimator and Testing Tool
Calculates estimated costs based on agent communication patterns
"""

import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import argparse


class EventBridgeCostEstimator:
    """Calculate and monitor EventBridge costs for agent communication"""
    
    # AWS EventBridge Pricing (as of 2024)
    PRICE_PER_MILLION_PUBLISHED = 1.00  # USD
    PRICE_PER_MILLION_MATCHED = 0.64     # USD
    
    # Budget limits
    MONTHLY_BUDGET = 10.00  # USD
    ALERT_THRESHOLD = 0.9   # Alert at 90% of budget
    
    def __init__(self):
        self.agent_patterns = {
            "monitor": {
                "events_per_hour": 2100,
                "events_per_day": 50000,
                "rule_matches_per_event": 2.5,  # Average rules triggered
                "description": "System monitoring and issue detection"
            },
            "product": {
                "events_per_hour": 200,
                "events_per_day": 5000,
                "rule_matches_per_event": 1.5,
                "description": "Feature requests and product feedback"
            },
            "developer": {
                "events_per_hour": 850,
                "events_per_day": 20000,
                "rule_matches_per_event": 3.0,  # Multiple routing rules
                "description": "Code changes and deployments"
            },
            "tester": {
                "events_per_hour": 1250,
                "events_per_day": 30000,
                "rule_matches_per_event": 2.0,
                "description": "Test execution and results"
            }
        }
        
    def calculate_agent_cost(self, agent_type: str, days: int = 30) -> Dict[str, float]:
        """Calculate cost for a single agent type"""
        if agent_type not in self.agent_patterns:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        pattern = self.agent_patterns[agent_type]
        
        # Calculate total events
        total_events = pattern["events_per_day"] * days
        
        # Calculate publishing cost
        publish_cost = (total_events / 1_000_000) * self.PRICE_PER_MILLION_PUBLISHED
        
        # Calculate rule matching cost
        total_matches = total_events * pattern["rule_matches_per_event"]
        match_cost = (total_matches / 1_000_000) * self.PRICE_PER_MILLION_MATCHED
        
        return {
            "agent_type": agent_type,
            "total_events": total_events,
            "total_matches": total_matches,
            "publish_cost": publish_cost,
            "match_cost": match_cost,
            "total_cost": publish_cost + match_cost,
            "daily_events": pattern["events_per_day"],
            "hourly_events": pattern["events_per_hour"]
        }
    
    def calculate_total_cost(self, days: int = 30) -> Dict[str, any]:
        """Calculate total cost for all agents"""
        results = {}
        total_events = 0
        total_matches = 0
        total_publish_cost = 0
        total_match_cost = 0
        
        # Calculate per-agent costs
        for agent_type in self.agent_patterns:
            agent_cost = self.calculate_agent_cost(agent_type, days)
            results[agent_type] = agent_cost
            
            total_events += agent_cost["total_events"]
            total_matches += agent_cost["total_matches"]
            total_publish_cost += agent_cost["publish_cost"]
            total_match_cost += agent_cost["match_cost"]
        
        total_cost = total_publish_cost + total_match_cost
        
        return {
            "period_days": days,
            "agents": results,
            "summary": {
                "total_events": total_events,
                "total_matches": total_matches,
                "total_publish_cost": total_publish_cost,
                "total_match_cost": total_match_cost,
                "total_cost": total_cost,
                "budget_remaining": self.MONTHLY_BUDGET - total_cost,
                "budget_utilization": (total_cost / self.MONTHLY_BUDGET) * 100,
                "is_within_budget": total_cost <= self.MONTHLY_BUDGET,
                "needs_alert": total_cost >= (self.MONTHLY_BUDGET * self.ALERT_THRESHOLD)
            }
        }
    
    def optimize_for_budget(self, target_budget: float = None) -> Dict[str, any]:
        """Suggest optimizations to stay within budget"""
        if target_budget is None:
            target_budget = self.MONTHLY_BUDGET
        
        current = self.calculate_total_cost()
        current_cost = current["summary"]["total_cost"]
        
        if current_cost <= target_budget:
            return {
                "optimization_needed": False,
                "current_cost": current_cost,
                "target_budget": target_budget,
                "message": "Current configuration is within budget"
            }
        
        # Calculate reduction needed
        reduction_factor = target_budget / current_cost
        
        optimizations = []
        
        # Suggest event reduction
        for agent_type, data in current["agents"].items():
            new_daily = int(data["daily_events"] * reduction_factor)
            new_hourly = int(data["hourly_events"] * reduction_factor)
            
            optimizations.append({
                "agent_type": agent_type,
                "current_daily": data["daily_events"],
                "suggested_daily": new_daily,
                "current_hourly": data["hourly_events"],
                "suggested_hourly": new_hourly,
                "reduction_percent": (1 - reduction_factor) * 100
            })
        
        return {
            "optimization_needed": True,
            "current_cost": current_cost,
            "target_budget": target_budget,
            "reduction_needed": (1 - reduction_factor) * 100,
            "optimizations": optimizations,
            "alternative_strategies": [
                "Implement event batching (combine multiple events)",
                "Use event sampling for non-critical monitoring",
                "Reduce rule complexity to decrease matches",
                "Archive old events instead of processing",
                "Use SQS for high-volume, low-priority events"
            ]
        }
    
    def simulate_burst_scenario(self, burst_multiplier: float = 3.0, 
                               burst_duration_hours: int = 4) -> Dict[str, any]:
        """Simulate cost impact of traffic bursts"""
        
        # Normal daily cost
        normal_day = self.calculate_total_cost(days=1)
        
        # Calculate burst events
        burst_events = {}
        for agent_type in self.agent_patterns:
            pattern = self.agent_patterns[agent_type]
            normal_hourly = pattern["events_per_hour"]
            burst_hourly = normal_hourly * burst_multiplier
            
            # Events during burst
            burst_period_events = burst_hourly * burst_duration_hours
            
            # Events during normal hours (24 - burst_duration)
            normal_period_events = normal_hourly * (24 - burst_duration_hours)
            
            total_day_events = burst_period_events + normal_period_events
            
            burst_events[agent_type] = {
                "normal_daily": pattern["events_per_day"],
                "burst_daily": total_day_events,
                "burst_hourly_rate": burst_hourly,
                "increase_percent": ((total_day_events / pattern["events_per_day"]) - 1) * 100
            }
        
        # Calculate burst day cost
        burst_cost = 0
        for agent_type, burst_data in burst_events.items():
            pattern = self.agent_patterns[agent_type]
            events = burst_data["burst_daily"]
            matches = events * pattern["rule_matches_per_event"]
            
            publish = (events / 1_000_000) * self.PRICE_PER_MILLION_PUBLISHED
            match = (matches / 1_000_000) * self.PRICE_PER_MILLION_MATCHED
            burst_cost += publish + match
        
        return {
            "scenario": "burst_traffic",
            "burst_multiplier": burst_multiplier,
            "burst_duration_hours": burst_duration_hours,
            "normal_daily_cost": normal_day["summary"]["total_cost"],
            "burst_daily_cost": burst_cost,
            "cost_increase": burst_cost - normal_day["summary"]["total_cost"],
            "cost_increase_percent": ((burst_cost / normal_day["summary"]["total_cost"]) - 1) * 100,
            "burst_events": burst_events,
            "monthly_impact": {
                "with_1_burst_day": (normal_day["summary"]["total_cost"] * 29) + burst_cost,
                "with_5_burst_days": (normal_day["summary"]["total_cost"] * 25) + (burst_cost * 5),
                "with_10_burst_days": (normal_day["summary"]["total_cost"] * 20) + (burst_cost * 10)
            }
        }
    
    def generate_report(self) -> str:
        """Generate a comprehensive cost report"""
        report = []
        report.append("=" * 60)
        report.append("EventBridge Agent Communication Cost Report")
        report.append("=" * 60)
        report.append(f"Generated: {datetime.now().isoformat()}")
        report.append(f"Monthly Budget: ${self.MONTHLY_BUDGET:.2f}")
        report.append("")
        
        # Current costs
        current = self.calculate_total_cost()
        report.append("MONTHLY COST PROJECTION (30 days)")
        report.append("-" * 40)
        
        for agent_type, data in current["agents"].items():
            report.append(f"\n{agent_type.upper()} Agent:")
            report.append(f"  Daily Events: {data['daily_events']:,}")
            report.append(f"  Monthly Events: {data['total_events']:,}")
            report.append(f"  Publishing Cost: ${data['publish_cost']:.2f}")
            report.append(f"  Matching Cost: ${data['match_cost']:.2f}")
            report.append(f"  Total Cost: ${data['total_cost']:.2f}")
        
        summary = current["summary"]
        report.append("\nTOTAL SUMMARY:")
        report.append("-" * 40)
        report.append(f"Total Monthly Events: {summary['total_events']:,}")
        report.append(f"Total Monthly Matches: {summary['total_matches']:,}")
        report.append(f"Total Publishing Cost: ${summary['total_publish_cost']:.2f}")
        report.append(f"Total Matching Cost: ${summary['total_match_cost']:.2f}")
        report.append(f"TOTAL MONTHLY COST: ${summary['total_cost']:.2f}")
        report.append(f"Budget Utilization: {summary['budget_utilization']:.1f}%")
        report.append(f"Budget Remaining: ${summary['budget_remaining']:.2f}")
        
        if summary["needs_alert"]:
            report.append("\n⚠️  WARNING: Approaching budget limit!")
        
        if not summary["is_within_budget"]:
            report.append("\n❌ OVER BUDGET - Optimization required")
            
            # Add optimization suggestions
            opt = self.optimize_for_budget()
            report.append("\nRECOMMENDED OPTIMIZATIONS:")
            report.append("-" * 40)
            report.append(f"Reduce events by {opt['reduction_needed']:.1f}%")
            
            for optimization in opt["optimizations"]:
                report.append(f"\n{optimization['agent_type'].upper()}:")
                report.append(f"  Current: {optimization['current_daily']:,} events/day")
                report.append(f"  Suggested: {optimization['suggested_daily']:,} events/day")
        else:
            report.append("\n✅ Within budget")
        
        # Burst scenario
        burst = self.simulate_burst_scenario()
        report.append("\nBURST TRAFFIC ANALYSIS:")
        report.append("-" * 40)
        report.append(f"Scenario: {burst['burst_multiplier']}x traffic for {burst['burst_duration_hours']} hours")
        report.append(f"Normal Daily Cost: ${burst['normal_daily_cost']:.2f}")
        report.append(f"Burst Daily Cost: ${burst['burst_daily_cost']:.2f}")
        report.append(f"Cost Increase: ${burst['cost_increase']:.2f} ({burst['cost_increase_percent']:.1f}%)")
        
        report.append("\nMonthly impact with burst days:")
        for scenario, cost in burst["monthly_impact"].items():
            days = scenario.split("_")[1]
            report.append(f"  {days} burst day(s): ${cost:.2f}")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description="EventBridge Cost Estimator")
    parser.add_argument("--days", type=int, default=30, 
                       help="Number of days to calculate (default: 30)")
    parser.add_argument("--budget", type=float, default=10.0,
                       help="Monthly budget in USD (default: 10.0)")
    parser.add_argument("--output", choices=["text", "json"], default="text",
                       help="Output format (default: text)")
    parser.add_argument("--optimize", action="store_true",
                       help="Show optimization suggestions")
    parser.add_argument("--burst", action="store_true",
                       help="Include burst scenario analysis")
    
    args = parser.parse_args()
    
    estimator = EventBridgeCostEstimator()
    estimator.MONTHLY_BUDGET = args.budget
    
    if args.output == "json":
        result = estimator.calculate_total_cost(args.days)
        if args.optimize:
            result["optimizations"] = estimator.optimize_for_budget()
        if args.burst:
            result["burst_analysis"] = estimator.simulate_burst_scenario()
        print(json.dumps(result, indent=2))
    else:
        print(estimator.generate_report())


if __name__ == "__main__":
    main()
