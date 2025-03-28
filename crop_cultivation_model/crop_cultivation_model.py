import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from scipy.optimize import differential_evolution
import random

class CropSimulation:
    def __init__(self, crop_type="corn"):
        """Initialize the crop simulation model with default parameters."""
        self.crop_type = crop_type
        
        
        if crop_type == "corn":
            self.base_temp = 10.0  
            self.optimal_temp = 25.0  
            self.max_temp = 35.0  
            self.gdd_to_maturity = 2700  
            self.water_req_daily = 6.0  
            self.drought_sensitivity = 0.8  
            self.growth_stages = {
                "emergence": 0.05,  
                "vegetative": 0.30,
                "flowering": 0.55,
                "grain_filling": 0.80,
                "maturity": 1.0
            }
        elif crop_type == "wheat":
            self.base_temp = 4.0
            self.optimal_temp = 22.0
            self.max_temp = 30.0
            self.gdd_to_maturity = 2000
            self.water_req_daily = 4.0
            self.drought_sensitivity = 0.7
            self.growth_stages = {
                "emergence": 0.05,
                "tillering": 0.25,
                "stem_extension": 0.45,
                "heading": 0.65,
                "grain_filling": 0.85,
                "maturity": 1.0
            }
        else:  
            self.base_temp = 8.0
            self.optimal_temp = 23.0
            self.max_temp = 32.0
            self.gdd_to_maturity = 2400
            self.water_req_daily = 5.0
            self.drought_sensitivity = 0.75
            self.growth_stages = {
                "emergence": 0.05,
                "vegetative": 0.35,
                "flowering": 0.60,
                "maturity": 1.0
            }
            
        
        self.current_gdd = 0
        self.accumulated_stress = 0
        self.current_growth_stage = "not_planted"
        self.yield_potential = 100.0  
        
    def calculate_gdd(self, t_min, t_max):
        """Calculate growing degree days for a single day."""
        t_min = max(t_min, self.base_temp)
        t_max = min(t_max, self.max_temp)
        return max(0, (t_min + t_max) / 2 - self.base_temp)
    
    def calculate_water_stress(self, available_water):
        """Calculate water stress factor (0-1) where 1 is no stress."""
        water_ratio = min(1.0, available_water / self.water_req_daily)
        
        stress_factor = water_ratio ** self.drought_sensitivity
        return stress_factor
    
    def calculate_temp_stress(self, t_mean):
        """Calculate temperature stress factor (0-1) where 1 is no stress."""
        if t_mean < self.base_temp:
            return max(0, t_mean / self.base_temp)
        elif t_mean <= self.optimal_temp:
            return 1.0
        else:
            
            temp_ratio = max(0, 1 - (t_mean - self.optimal_temp) / (self.max_temp - self.optimal_temp))
            return temp_ratio ** 2  
    
    def update_growth_stage(self):
        """Update the crop growth stage based on accumulated GDD."""
        progress = self.current_gdd / self.gdd_to_maturity
        
        if progress < self.growth_stages["emergence"]:
            self.current_growth_stage = "not_emerged"
        else:
            
            for stage, threshold in sorted(self.growth_stages.items()):
                if progress <= threshold:
                    self.current_growth_stage = stage
                    break
    
    def simulate_day(self, t_min, t_max, rainfall, irrigation=0):
        """Simulate a single day of crop growth."""
        
        t_mean = (t_min + t_max) / 2
        daily_gdd = self.calculate_gdd(t_min, t_max)
        
        
        available_water = rainfall + irrigation
        
        
        water_stress = self.calculate_water_stress(available_water)
        temp_stress = self.calculate_temp_stress(t_mean)
        daily_stress = 1 - (water_stress * temp_stress)
        
        
        if self.current_growth_stage == "flowering":
            stress_impact = daily_stress * 1.5  
        elif self.current_growth_stage == "grain_filling":
            stress_impact = daily_stress * 1.3  
        else:
            stress_impact = daily_stress
        
        
        self.accumulated_stress += stress_impact * (1 - self.accumulated_stress / 100)
        
        
        effective_gdd = daily_gdd * (1 - daily_stress / 2)  
        self.current_gdd += effective_gdd
        
        
        self.update_growth_stage()
        
        
        
        if self.current_growth_stage in ["flowering", "grain_filling"]:
            self.yield_potential -= stress_impact * 0.7  
        else:
            self.yield_potential -= stress_impact * 0.3  
            
        
        self.yield_potential = max(0, self.yield_potential)
        
        return {
            "gdd_added": effective_gdd,
            "total_gdd": self.current_gdd,
            "growth_stage": self.current_growth_stage,
            "water_stress": 1 - water_stress,  
            "temp_stress": 1 - temp_stress,
            "daily_stress": daily_stress,
            "yield_potential": self.yield_potential,
            "progress": min(1.0, self.current_gdd / self.gdd_to_maturity)
        }
    
    def simulate_season(self, weather_data, planting_date, irrigation_schedule=None):
        """
        Simulate an entire growing season.
        
        Parameters:
        - weather_data: DataFrame with columns [date, t_min, t_max, rainfall]
        - planting_date: Start date for the simulation
        - irrigation_schedule: Dictionary with {date: amount} for irrigation
        
        Returns:
        - DataFrame with daily simulation results
        """
        
        self.current_gdd = 0
        self.accumulated_stress = 0
        self.current_growth_stage = "not_emerged"
        self.yield_potential = 100.0
        
        
        weather_data = weather_data.sort_values('date')
        
        
        if irrigation_schedule is None:
            irrigation_schedule = {}
        
        
        planting_date_pd = pd.to_datetime(planting_date)
        
        
        if planting_date_pd is None:
            print("Warning: Invalid planting date")
            return pd.DataFrame()
            
        seasonal_weather = weather_data[weather_data['date'] >= planting_date_pd].copy()
        
        if len(seasonal_weather) == 0:
            return pd.DataFrame()  
        
        
        results = []
        
        
        for _, day in seasonal_weather.iterrows():
            date = day['date']
            t_min = day['t_min']
            t_max = day['t_max']
            rainfall = day['rainfall']
            
            
            irrigation = irrigation_schedule.get(date.strftime('%Y-%m-%d'), 0)
            
            
            day_result = self.simulate_day(t_min, t_max, rainfall, irrigation)
            
            
            day_result['date'] = date
            day_result['irrigation'] = irrigation
            day_result['rainfall'] = rainfall
            day_result['t_min'] = t_min
            day_result['t_max'] = t_max
            
            results.append(day_result)
            
            
            if day_result['growth_stage'] == 'maturity' and day_result['progress'] >= 1.0:
                break
        
        return pd.DataFrame(results)


def generate_weather_scenarios(start_date, days, scenario="normal"):
    """
    Generate synthetic weather data for various scenarios.
    
    Parameters:
    - start_date: Beginning date for the dataset (string "YYYY-MM-DD")
    - days: Number of days to generate
    - scenario: "normal", "drought", "excessive_rain", "heat_wave", "cold_snap"
    
    Returns:
    - DataFrame with columns [date, t_min, t_max, rainfall]
    """
    dates = pd.date_range(start=start_date, periods=days)
    
    
    base_temp_min = 15
    base_temp_max = 28
    base_temp_amplitude = 10  
    base_rain_prob = 0.3
    base_rain_amount = 4
    
    
    if scenario == "drought":
        base_temp_min += 2
        base_temp_max += 5
        base_rain_prob = 0.1
        base_rain_amount = 1.5
    elif scenario == "excessive_rain":
        base_temp_min -= 1
        base_temp_max -= 2
        base_rain_prob = 0.7
        base_rain_amount = 8
    elif scenario == "heat_wave":
        base_temp_min += 5
        base_temp_max += 8
        base_rain_prob = 0.15
        base_rain_amount = 2
    elif scenario == "cold_snap":
        base_temp_min -= 6
        base_temp_max -= 5
        base_rain_prob = 0.4
        base_rain_amount = 3
    
    
    t_min = []
    t_max = []
    rainfall = []
    
    for i, date in enumerate(dates):
        
        day_of_year = date.dayofyear
        season_factor = np.sin(day_of_year / 365 * 2 * np.pi)
        
        
        daily_t_min = base_temp_min + base_temp_amplitude * season_factor + np.random.normal(0, 2)
        daily_t_max = base_temp_max + base_temp_amplitude * season_factor + np.random.normal(0, 3)
        
        
        daily_t_max = max(daily_t_min + 2, daily_t_max)
        
        
        rain_prob = base_rain_prob
        
        
        if i > 0 and rainfall[i-1] > 0:
            rain_prob = min(0.8, rain_prob * 1.5)
        
        if np.random.random() < rain_prob:
            
            daily_rainfall = np.random.lognormal(mean=np.log(base_rain_amount), sigma=0.6)
        else:
            daily_rainfall = 0
            
        t_min.append(daily_t_min)
        t_max.append(daily_t_max)
        rainfall.append(daily_rainfall)
    
    
    weather_df = pd.DataFrame({
        'date': dates,
        't_min': t_min,
        't_max': t_max,
        'rainfall': rainfall
    })
    
    return weather_df


def optimize_planting_date(crop_model, weather_data, possible_dates):
    """
    Find the optimal planting date to maximize yield.
    
    Parameters:
    - crop_model: Instance of CropSimulation
    - weather_data: DataFrame with weather data
    - possible_dates: List of possible planting dates
    
    Returns:
    - best_date: The optimal planting date
    - best_yield: The projected yield with this planting date
    """
    best_yield = 0
    best_date = possible_dates[0]  
    
    for date in possible_dates:
        results = crop_model.simulate_season(weather_data, date)
        
        if len(results) > 0:
            final_yield = results.iloc[-1]['yield_potential']
            if final_yield > best_yield:
                best_yield = final_yield
                best_date = date
    
    return best_date, best_yield

def optimize_irrigation(crop_model, weather_data, planting_date, max_irrigation_events=10, max_amount=15):
    """
    Optimize irrigation schedule using a genetic algorithm approach.
    
    Parameters:
    - crop_model: CropSimulation instance
    - weather_data: Weather DataFrame
    - planting_date: When crop is planted
    - max_irrigation_events: Maximum number of irrigation events
    - max_amount: Maximum amount per irrigation (mm)
    
    Returns:
    - optimal_schedule: Dictionary of {date: amount}
    - optimized_yield: Yield with optimal irrigation
    """
    
    planting_date_pd = pd.to_datetime(planting_date)
    
    
    if planting_date_pd is None:
        return {}, 0
        
    season_dates = weather_data[weather_data['date'] >= planting_date_pd]['date'].tolist()
    
    if len(season_dates) == 0:
        return {}, 0
    
    
    if len(season_dates) > 100:
        season_dates = season_dates[:100]
    
    
    def evaluate_schedule(irrigation_amounts):
        
        irrigation_schedule = {}
        for i in range(min(len(irrigation_amounts), len(season_dates))):
            amount = irrigation_amounts[i]
            if amount > 0:  
                date_str = season_dates[i].strftime('%Y-%m-%d')
                irrigation_schedule[date_str] = amount
        
        
        results = crop_model.simulate_season(weather_data, planting_date, irrigation_schedule)
        
        if len(results) > 0:
            final_yield = results.iloc[-1]['yield_potential']
            
            
            total_water = sum(irrigation_schedule.values())
            
            
            water_efficiency = 1.0
            if total_water > 0:
                water_efficiency = min(1.0, 100 / total_water)  
            
            
            return -(final_yield * water_efficiency)  
        else:
            return -0  
    
    
    bounds = [(0, max_amount) for _ in range(len(season_dates))]
    
    
    result = differential_evolution(
        evaluate_schedule, 
        bounds, 
        maxiter=10,  
        popsize=10,
        mutation=(0.5, 1.0),
        recombination=0.7
    )
    
    
    optimal_schedule = {}
    for i, amount in enumerate(result.x):
        if amount > 1.0:  
            date_str = season_dates[i].strftime('%Y-%m-%d')
            optimal_schedule[date_str] = round(amount, 1)
    
    
    results = crop_model.simulate_season(weather_data, planting_date, optimal_schedule)
    optimized_yield = results.iloc[-1]['yield_potential'] if len(results) > 0 else 0
    
    return optimal_schedule, optimized_yield


def plot_weather_data(weather_df, title="Weather Data"):
    """Plot temperature and rainfall data."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    
    ax1.plot(weather_df['date'], weather_df['t_max'], 'r-', label='Max Temperature')
    ax1.plot(weather_df['date'], weather_df['t_min'], 'b-', label='Min Temperature')
    ax1.set_ylabel('Temperature (°C)')
    ax1.set_title(f"{title} - Temperature")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    
    ax2.bar(weather_df['date'], weather_df['rainfall'], color='skyblue', width=1)
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Rainfall (mm)')
    ax2.set_title('Rainfall')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_crop_growth(results, title="Crop Growth Simulation"):
    """Plot key metrics from crop growth simulation."""
    if len(results) == 0:
        
        fig = plt.figure(figsize=(12, 10))
        plt.text(0.5, 0.5, "No data to display", ha='center', va='center')
        return fig
        
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    
    ax1.plot(results['date'], results['progress'] * 100, 'g-', label='Growth Progress (%)')
    ax1.plot(results['date'], results['yield_potential'], 'b-', label='Yield Potential (%)')
    ax1.set_ylabel('Percentage')
    ax1.set_title(f"{title} - Growth Progress and Yield Potential")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    
    ax2.plot(results['date'], results['water_stress'] * 100, 'r-', label='Water Stress (%)')
    ax2.plot(results['date'], results['temp_stress'] * 100, 'orange', label='Temperature Stress (%)')
    ax2.set_ylabel('Stress Level (%)')
    ax2.set_title('Stress Factors')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    
    ax3.bar(results['date'], results['rainfall'], color='skyblue', label='Rainfall', width=1)
    ax3.bar(results['date'], results['irrigation'], color='blue', bottom=results['rainfall'], label='Irrigation', width=1)
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Water (mm)')
    ax3.set_title('Water Inputs')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_yield_heatmap(results_dict, title="Yield Comparison"):
    """
    Create a heatmap comparing yield across different scenarios and strategies.
    
    Parameters:
    - results_dict: Nested dictionary {scenario: {strategy: yield}}
    """
    
    scenarios = list(results_dict.keys())
    strategies = list(results_dict[scenarios[0]].keys())
    
    data = []
    for scenario in scenarios:
        row = []
        for strategy in strategies:
            row.append(results_dict[scenario][strategy])
        data.append(row)
    
    df = pd.DataFrame(data, index=scenarios, columns=strategies)
    
    
    cmap = LinearSegmentedColormap.from_list('blue_gradient', ['#D1E5F0', '#4393C3', '#2166AC'])
    
    
    plt.figure(figsize=(10, 6))
    ax = sns.heatmap(df, annot=True, fmt=".1f", cmap=cmap, linewidths=.5)
    plt.title(title)
    plt.tight_layout()
    
    return plt.gcf()


def run_complete_simulation():
    """Run a complete simulation with multiple scenarios and optimization strategies."""
    
    yield_comparison = {}
    simulation_results = {}
    
    
    weather_scenarios = {
        "Normal": generate_weather_scenarios("2023-04-01", 180, "normal"),
        "Drought": generate_weather_scenarios("2023-04-01", 180, "drought"),
        "Excessive Rain": generate_weather_scenarios("2023-04-01", 180, "excessive_rain"),
        "Heat Wave": generate_weather_scenarios("2023-04-01", 180, "heat_wave")
    }
    
    
    corn_model = CropSimulation(crop_type="corn")
    
    
    planting_dates = [f"2023-04-{i:02d}" for i in range(1, 30)]
    
    
    for scenario_name, weather_data in weather_scenarios.items():
        yield_comparison[scenario_name] = {}
        simulation_results[scenario_name] = {}
        
        
        baseline_date = "2023-04-15"
        baseline_results = corn_model.simulate_season(weather_data, baseline_date)
        baseline_yield = baseline_results.iloc[-1]['yield_potential'] if len(baseline_results) > 0 else 0
        
        yield_comparison[scenario_name]["Baseline"] = baseline_yield
        simulation_results[scenario_name]["Baseline"] = baseline_results
        
        best_date, best_yield = optimize_planting_date(corn_model, weather_data, planting_dates)
        opt_date_results = corn_model.simulate_season(weather_data, best_date)
        
        print(f"Scenario: {scenario_name}, Best date: {best_date}, Yield: {best_yield:.1f}%")
        
        yield_comparison[scenario_name]["Optimized Planting"] = best_yield
        simulation_results[scenario_name]["Optimized Planting"] = opt_date_results
        
        optimal_schedule, opt_irr_yield = optimize_irrigation(corn_model, weather_data, baseline_date)
        opt_irr_results = corn_model.simulate_season(weather_data, baseline_date, optimal_schedule)
        
        yield_comparison[scenario_name]["Optimized Irrigation"] = opt_irr_yield
        simulation_results[scenario_name]["Optimized Irrigation"] = opt_irr_results
        
        combined_schedule, combined_yield = optimize_irrigation(corn_model, weather_data, best_date)
        combined_results = corn_model.simulate_season(weather_data, best_date, combined_schedule)
        
        yield_comparison[scenario_name]["Combined Optimization"] = combined_yield
        simulation_results[scenario_name]["Combined Optimization"] = combined_results
    
    return weather_scenarios, yield_comparison, simulation_results

if __name__ == "__main__":
    weather_scenarios, yield_comparison, simulation_results = run_complete_simulation()
    
    for scenario in weather_scenarios:
        weather_fig = plot_weather_data(weather_scenarios[scenario], title=f"{scenario} Weather")
        weather_fig.savefig(f"{scenario}_weather.png")

        for strategy in simulation_results[scenario]:
            results = simulation_results[scenario][strategy]
            if len(results) > 0:
                growth_fig = plot_crop_growth(results, title=f"{scenario} - {strategy}")
                growth_fig.savefig(f"{scenario}_{strategy}_growth.png")

    heatmap_fig = create_yield_heatmap(yield_comparison, title="Crop Yield Comparison (%)")
    heatmap_fig.savefig("yield_comparison.png")
    
    print("Simulation completed successfully!")