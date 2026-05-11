import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

output_dir = Path(__file__).resolve().parent

# Plot 1: Native Californian Population Decline (1769-1900)
years = [1769, 1800, 1821, 1848, 1870, 1900]
population = [300000, 250000, 200000, 150000, 300000, 15000] # Adjusted roughly based on Cook's estimates
population = [300000, 260000, 200000, 150000, 30000, 15377]

plt.figure(figsize=(10, 6))
plt.plot(years, population, marker='o', color='darkred', linewidth=2.5)
plt.fill_between(years, population, color='salmon', alpha=0.3)
plt.title('Native American Population Decline in California (1769 - 1900)', fontsize=14, fontweight='bold')
plt.xlabel('Year', fontsize=12)
plt.ylabel('Estimated Population', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.annotate('Spanish Colonization (1769)', xy=(1769, 300000), xytext=(1780, 280000),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6))
plt.annotate('Gold Rush & Statehood (1848)', xy=(1848, 150000), xytext=(1855, 180000),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6))
plt.tight_layout()
plt.savefig(output_dir / 'population_decline.png', dpi=300)
plt.close()

# Plot 2: Mission Demographics (Births vs Deaths proxy)
# Data represents crude rates per 1000 people
categories = ['Birth Rate (Missions)', 'Death Rate (Missions)']
rates = [35, 90] # Approximations representing the fertility crisis

plt.figure(figsize=(8, 6))
bars = plt.bar(categories, rates, color=['#4C72B0', '#C44E52'], width=0.5)
plt.title('Demographic Crisis in the Missions (Approx. Rates per 1,000)', fontsize=14, fontweight='bold')
plt.ylabel('Rate per 1,000 individuals', fontsize=12)
plt.ylim(0, 100)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 2, f'{yval}', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.text(-0.25, 40, 'Insufficient to sustain\npopulation (1.5 live births/woman)', style='italic', fontsize=10, bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 5})
plt.text(0.75, 95, 'Driven by disease\n(syphilis, measles) & poor conditions', style='italic', fontsize=10, bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 5})

plt.tight_layout()
plt.savefig(output_dir / 'mission_demographics.png', dpi=300)
plt.close()


# Plot 3: Land Ownership Transition (1820 - 1860)
# This is a conceptual stacked area chart showing the rapid transfer of land wealth
years_land = [1820, 1833, 1845, 1855, 1865]
# Percentage of "usable/coastal" estate land controlled by group
native = [10, 5, 2, 1, 0] # Mostly marginalized even before secularization, completely dispossessed after
missions = [85, 0, 0, 0, 0] # Secularized in 1833
californios = [5, 95, 90, 30, 5] # Gained everything in 1833, lost it to US courts/settlers by 1860s
us_settlers = [0, 0, 8, 69, 95] 

plt.figure(figsize=(10, 6))
plt.stackplot(years_land, native, missions, californios, us_settlers, 
              labels=['Native Americans', 'Franciscan Missions', 'Californio Families (800)', 'U.S. Settlers/Speculators'],
              colors=['#55A868', '#CCB974', '#DD8452', '#4C72B0'], alpha=0.8)

plt.title('Transition of Prime Estate Land Ownership in California (1820 - 1865)', fontsize=14, fontweight='bold')
plt.xlabel('Year', fontsize=12)
plt.ylabel('Approximate Percentage of Arable Land Wealth (%)', fontsize=12)
plt.legend(loc='upper left')
plt.margins(x=0, y=0)
plt.axvline(x=1833, color='black', linestyle='--', alpha=0.6)
plt.text(1834, 40, '1833: Secularization Act\n(Missions Dissolved)', rotation=90, verticalalignment='center', fontsize=10)

plt.axvline(x=1851, color='black', linestyle='--', alpha=0.6)
plt.text(1852, 40, '1851: CA Land Act\n(Californios bankrupt in US Courts)', rotation=90, verticalalignment='center', fontsize=10)

plt.tight_layout()
plt.savefig(output_dir / 'land_ownership.png', dpi=300)
plt.close()

print("Plots generated successfully.")
