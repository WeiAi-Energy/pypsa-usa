import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.widgets import CheckButtons

warnings.filterwarnings("ignore")


class NZAScenarioAnalyzer:
    """
    Analyzer for PyPSA-USA NZA scenario energy demand data with emission cap analysis
    """

    def __init__(
        self,
        demand_path="../repo_data/MY_simpsec_demand",
        emissions_path="../repo_data/MY_nza_emission/nza_emissions.csv",
    ):
        self.demand_path = Path(demand_path)
        self.emissions_path = Path(emissions_path)

        # NZA scenarios - based on PyPSA-USA codebase analysis
        self.scenarios = ["E+", "E-", "E+RE+", "E+RE-", "E-B+"]

        # Energy carriers
        self.carriers = ["electricity", "natural_gas", "hydrogen"]

        # Years (typically 2025-2050 in 5-year increments)
        self.years = list(range(2050, 2051, 5))

        # Color mapping for carriers
        self.carrier_colors = {
            "electricity": "#1f77b4",  # Blue
            "natural_gas": "#ff7f0e",  # Orange
            "hydrogen": "#2ca02c",  # Green
        }

        # Color for emission cap line
        self.emission_color = "#d62728"  # Red

        self.demand_data = {}
        self.emissions_data = {}

    def load_demand_data(self):
        """
        Load demand data from CSV files for all scenarios, years, and carriers
        """
        print("Loading PyPSA-USA demand data...")

        for scenario in self.scenarios:
            self.demand_data[scenario] = {}

            for year in self.years:
                self.demand_data[scenario][year] = {}

                for carrier in self.carriers:
                    file_path = self.demand_path / scenario / f"{carrier}_{year}.csv"

                    try:
                        # Load CSV with first column as index (timestamp)
                        df = pd.read_csv(file_path, index_col=0)

                        # Calculate total national consumption (sum all regions)
                        # Convert from MW*hours to TWh: sum over time series then divide by 1e6
                        if not df.empty:
                            # Sum across all regions/buses (columns)
                            regional_totals = df.sum(axis=1)
                            # Sum across time to get annual total, convert MW·h to TWh
                            annual_total_twh = regional_totals.sum() / 1e6

                            self.demand_data[scenario][year][carrier] = annual_total_twh
                            print(f"✓ Loaded {scenario} {carrier} {year}: {annual_total_twh:.1f} TWh")
                        else:
                            self.demand_data[scenario][year][carrier] = 0.0
                            print(f"⚠ Empty file: {file_path}")

                    except Exception as e:
                        print(f"❌ Error loading {file_path}: {e}")
                        self.demand_data[scenario][year][carrier] = 0.0

    def load_emissions_data(self):
        """
        Load emissions data from CSV file
        """
        print("Loading NZA emissions data...")

        try:
            df = pd.read_csv(self.emissions_path)

            # Initialize emissions data structure
            for scenario in self.scenarios:
                self.emissions_data[scenario] = {}
                for year in self.years:
                    self.emissions_data[scenario][year] = 0.0

            # Load data from CSV
            for _, row in df.iterrows():
                scenario = row["scenario"]
                year = int(row["year"])

                # Map scenario names if needed
                if scenario == "REF":
                    continue  # Skip REF scenario as it's not in our demand scenarios

                if scenario in self.scenarios and year in self.years:
                    self.emissions_data[scenario][year] = row["emission_cap"]
                    print(f"✓ Loaded emissions {scenario} {year}: Total={row['emission_cap']:.1f}")

            print("Emissions data loading completed!")

        except Exception as e:
            print(f"❌ Error loading emissions data: {e}")

    def load_data(self):
        """
        Load both demand and emissions data
        """
        self.load_demand_data()
        self.load_emissions_data()

    def create_demand_summary(self):
        """
        Create a summary DataFrame of all demand data
        """
        summary_data = []

        for scenario in self.scenarios:
            for year in self.years:
                for carrier in self.carriers:
                    demand = self.demand_data[scenario][year][carrier]
                    summary_data.append(
                        {
                            "Scenario": scenario,
                            "Year": year,
                            "Carrier": carrier,
                            "Demand_TWh": demand,
                        }
                    )

        return pd.DataFrame(summary_data)

    def plot_demand_with_emissions(self, selected_years=None):
        """
        Create bar chart array showing demand by scenario and year with emission cap info on right y-axis
        """
        if selected_years is None:
            selected_years = self.years

        # Filter years
        plot_years = [year for year in selected_years if year in self.years]

        # Create subplot grid: 1 row, 5 columns (one for each scenario)
        fig, axes = plt.subplots(1, 5, figsize=(24, 8))
        fig.suptitle("Energy Demand by NZA Scenario with Emission Cap", fontsize=16, fontweight="bold")

        # Width of bars
        bar_width = 0.25

        # STEP 1: Calculate global maximum for demand across all scenarios and carriers
        global_demand_max = 0
        all_demand_data = {}  # Store all data for plotting later

        for scenario in self.scenarios:
            all_demand_data[scenario] = {}
            for year in plot_years:
                electricity = self.demand_data[scenario][year]["electricity"]
                gas = self.demand_data[scenario][year]["natural_gas"]
                hydrogen = self.demand_data[scenario][year]["hydrogen"]

                all_demand_data[scenario][year] = {
                    "electricity": electricity,
                    "gas": gas,
                    "hydrogen": hydrogen,
                }

                # Update global maximum for demand
                global_demand_max = max(global_demand_max, electricity, gas, hydrogen)

        # Add some padding to the demand maximum (10% margin)
        demand_y_max = global_demand_max * 1.1

        # STEP 2: Calculate global range for emissions
        global_emission_min = float("inf")
        global_emission_max = float("-inf")

        for scenario in self.scenarios:
            for year in plot_years:
                value = self.emissions_data[scenario][year]
                global_emission_min = min(global_emission_min, value)
                global_emission_max = max(global_emission_max, value)

        # Add padding to emission range (10% margin)
        emission_range = global_emission_max - global_emission_min
        emission_y_min = global_emission_min - emission_range * 0.1
        emission_y_max = global_emission_max + emission_range * 0.1

        # STEP 3: Plot each scenario with uniform y-axis
        for i, scenario in enumerate(self.scenarios):
            ax = axes[i]

            # Prepare demand data for this scenario
            years_data = []
            electricity_data = []
            gas_data = []
            hydrogen_data = []

            for year in plot_years:
                years_data.append(year)
                electricity_data.append(all_demand_data[scenario][year]["electricity"])
                gas_data.append(all_demand_data[scenario][year]["gas"])
                hydrogen_data.append(all_demand_data[scenario][year]["hydrogen"])

            # X positions for bars
            x_pos = np.arange(len(years_data))

            # Create grouped bars for demand (left y-axis)
            bars1 = ax.bar(
                x_pos - bar_width,
                electricity_data,
                bar_width,
                label="Electricity",
                color=self.carrier_colors["electricity"],
                alpha=0.8,
            )
            bars2 = ax.bar(
                x_pos,
                gas_data,
                bar_width,
                label="Natural Gas",
                color=self.carrier_colors["natural_gas"],
                alpha=0.8,
            )
            bars3 = ax.bar(
                x_pos + bar_width,
                hydrogen_data,
                bar_width,
                label="Hydrogen",
                color=self.carrier_colors["hydrogen"],
                alpha=0.8,
            )

            # Create second y-axis for emissions (right side)
            ax2 = ax.twinx()

            # Prepare emissions data for this scenario
            emission_cap_data = [self.emissions_data[scenario][year] for year in plot_years]

            # Plot emission cap line with markers on right y-axis
            ax2.plot(
                x_pos,
                emission_cap_data,
                "o-",
                linewidth=2,
                markersize=6,
                color=self.emission_color,
                label="Emission Cap",
                alpha=0.9,
            )

            # Customize left subplot (demand)
            ax.set_title(scenario, fontsize=12, fontweight="bold")
            ax.set_xlabel("Year", fontsize=10)
            if i == 0:  # Only label left y-axis for first subplot
                ax.set_ylabel("Energy Demand (TWh)", fontsize=10, color="black")

            ax.set_xticks(x_pos)
            ax.set_xticklabels(years_data, rotation=45)
            ax.grid(True, alpha=0.3, axis="y")

            # Set uniform y-axis limits for demand (left axis)
            ax.set_ylim(0, demand_y_max)

            # Customize right y-axis (emission cap)
            if i == len(self.scenarios) - 1:  # Only label right y-axis for last subplot
                ax2.set_ylabel("Emission Cap (Mt CO2-eq)", fontsize=10, color="red")

            # Set uniform y-axis limits for emissions (right axis)
            ax2.set_ylim(emission_y_min, emission_y_max)
            ax2.tick_params(axis="y", labelcolor="red")

            # Add value labels on bars for demand (optional - can be removed if crowded)
            def add_value_labels(bars, threshold=demand_y_max * 0.03):  # Only show labels for values > 3% of max
                for bar in bars:
                    height = bar.get_height()
                    if height > threshold:  # Only label significant bars
                        ax.annotate(
                            f"{height:.0f}",
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha="center",
                            va="bottom",
                            fontsize=8,
                        )

            add_value_labels(bars1)
            add_value_labels(bars2)
            add_value_labels(bars3)

        # Create combined legend
        # Demand legend (bars)
        demand_handles = [
            plt.Rectangle((0, 0), 1, 1, color=self.carrier_colors["electricity"], alpha=0.8),
            plt.Rectangle((0, 0), 1, 1, color=self.carrier_colors["natural_gas"], alpha=0.8),
            plt.Rectangle((0, 0), 1, 1, color=self.carrier_colors["hydrogen"], alpha=0.8),
        ]
        demand_labels = ["Electricity", "Natural Gas", "Hydrogen"]

        # Emission cap legend (line)
        emission_handle = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color=self.emission_color,
                linewidth=2,
                markersize=6,
            ),
        ]
        emission_label = ["Emission Cap"]

        # Combined legend
        all_handles = demand_handles + emission_handle
        all_labels = demand_labels + emission_label

        # Add legend to the last subplot
        axes[-1].legend(
            all_handles,
            all_labels,
            loc="upper left",
            bbox_to_anchor=(1.15, 1),
            fontsize=9,
        )

        plt.tight_layout()
        return fig

    def plot_scenario_trends(self, selected_years=None):
        """
        Create line plots showing demand trends over time for each carrier
        """
        if selected_years is None:
            selected_years = self.years

        plot_years = [year for year in selected_years if year in self.years]

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle("Energy Demand Trends by Carrier and Scenario", fontsize=16, fontweight="bold")

        for i, carrier in enumerate(self.carriers):
            ax = axes[i]

            for scenario in self.scenarios:
                demand_values = [self.demand_data[scenario][year][carrier] for year in plot_years]
                ax.plot(
                    plot_years,
                    demand_values,
                    marker="o",
                    linewidth=2,
                    markersize=6,
                    label=scenario,
                )

            ax.set_title(f"{carrier.replace('_', ' ').title()} Demand", fontsize=12, fontweight="bold")
            ax.set_xlabel("Year", fontsize=10)
            if i == 0:
                ax.set_ylabel("Energy Demand (TWh)", fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)

        plt.tight_layout()
        return fig

    def create_interactive_plot(self):
        """
        Create an interactive plot with year selection checkboxes
        """
        # Create the main plot
        fig = self.plot_demand_with_emissions()

        # Add checkboxes for year selection
        plt.subplots_adjust(right=0.80)

        # Create checkbox axes
        checkbox_ax = plt.axes([0.82, 0.4, 0.1, 0.4])

        # Year labels and initial states
        year_labels = [str(year) for year in self.years]
        year_states = [True] * len(self.years)  # All years initially selected

        # Create checkboxes
        checkbox = CheckButtons(checkbox_ax, year_labels, year_states)

        def update_plot(label):
            # Get selected years
            selected_years = []
            for i, year_label in enumerate(year_labels):
                if checkbox.get_status()[i]:
                    selected_years.append(int(year_label))

            # Clear and redraw
            for ax in fig.axes[:-1]:  # Exclude checkbox axis
                ax.clear()
                # Clear twin axes as well
                if hasattr(ax, "twin_axes"):
                    for twin_ax in ax.twin_axes:
                        twin_ax.clear()

            # Recreate plot with selected years
            if selected_years:
                new_fig = self.plot_demand_with_emissions(selected_years)
                plt.close(new_fig)

            fig.canvas.draw()

        # Connect callback
        checkbox.on_clicked(update_plot)

        return fig

    def export_summary_table(self, filename="nza_scenario_summary.csv"):
        """
        Export summary data to CSV including both demand and emissions
        """
        summary_data = []

        for scenario in self.scenarios:
            for year in self.years:
                # Demand data
                for carrier in self.carriers:
                    demand = self.demand_data[scenario][year][carrier]
                    summary_data.append(
                        {
                            "Scenario": scenario,
                            "Year": year,
                            "Type": "Demand",
                            "Category": carrier,
                            "Value": demand,
                            "Unit": "TWh",
                        }
                    )

                # Emissions data (emission cap only)
                emissions = self.emissions_data[scenario][year]
                summary_data.append(
                    {
                        "Scenario": scenario,
                        "Year": year,
                        "Type": "Emissions",
                        "Category": "emission_cap",
                        "Value": emissions,
                        "Unit": "Mt CO2-eq",
                    }
                )

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(filename, index=False)
        print(f"📊 Summary data exported to {filename}")
        return summary_df


def main():
    """
    Main function to run the NZA scenario analysis
    """
    # Initialize analyzer
    analyzer = NZAScenarioAnalyzer()

    # Load data
    analyzer.load_data()

    # Create visualizations
    print("\nCreating visualizations...")

    # 1. Main comparison plot with emissions
    fig1 = analyzer.plot_demand_with_emissions()
    plt.show()

    # 2. Trend lines for demand only
    fig2 = analyzer.plot_scenario_trends()
    plt.show()

    # 3. Export combined data
    summary_df = analyzer.export_summary_table()
    print("\nFirst few rows of summary data:")
    print(summary_df.head(10))

    print("\n✅ NZA Scenario Analysis completed!")


if __name__ == "__main__":
    main()
