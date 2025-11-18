# oecd_education_visualizations.py
# OECD Education Data - Interactive Visualization Dashboard
# Uses Plotly for interactive charts with filters and legends

import dask.dataframe as dd
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np

# HELPER FUNCTION: Categorize education levels
def categorize_education(attainment_level):
    if pd.isna(attainment_level):
        return 'Other'
    
    # This line converts any input (int or str) to a string
    level_str = str(attainment_level) 
    
    # This block CHECKS FOR SUBSTRINGS, it doesn't try to convert to int
    if any(x in level_str for x in ['ISCED11A_0', 'ISCED11A_1', 'ISCED11A_2']):
        return 'Primary'
    elif any(x in level_str for x in ['ISCED11A_3', 'ISCED11A_4']):
        return 'Secondary'
    elif any(x in level_str for x in ['ISCED11A_5', 'ISCED11A_6', 'ISCED11A_7', 'ISCED11A_8']):
        return 'Tertiary'
    return 'Other'

print("=" * 80)
print("OECD EDUCATION DATA - INTERACTIVE VISUALIZATIONS")
print("=" * 80)

# ----------------------------------------------------------------------------
# MAPPINGS AND CONSTANTS
# ----------------------------------------------------------------------------

# Education field mapping
FIELD_MAP = {
    'FIELD001': 'Education',
    'FIELD002': 'Arts/Humanities',
    'FIELD003': 'Social Sciences',
    'FIELD004': 'Business/Law',
    'FIELD005': 'Natural Sciences',
    'FIELD006': 'ICT',
    'FIELD007': 'Engineering',
    'FIELD008': 'Agriculture',
    'FIELD009': 'Health/Welfare',
    'FIELD010': 'Services',
    '_T': 'Total'
}

# Age group mapping
AGE_MAP = {
    'Y25T64': 'Ages 25-64',
    'Y15T64': 'Ages 15-64',
    'Y15T24': 'Ages 15-24',
    'Y25T34': 'Ages 25-34',
    'Y35T44': 'Ages 35-44',
    'Y45T54': 'Ages 45-54',
    'Y55T64': 'Ages 55-64',
    '_T': 'All Ages'
}

# Gender mapping
SEX_MAP = {
    'F': 'Female',
    'M': 'Male',
    '_T': 'Total'
}
# Birth place mapping
BIRTH_PLACE_MAP = {
    'NATIVE': 'Native-Born',
    'FOREIGN': 'Foreign-Born',
    '_T': 'Total',
    '_Z': 'Not Applicable' 
}

# Country / Reference Area mapping
COUNTRY_MAP = {
    # OECD Members
    'AUS': 'Australia', 'AUT': 'Austria', 'BEL': 'Belgium', 'CAN': 'Canada',
    'CHL': 'Chile', 'COL': 'Colombia', 'CRI': 'Costa Rica', 'CZE': 'Czechia',
    'DNK': 'Denmark', 'EST': 'Estonia', 'FIN': 'Finland', 'FRA': 'France',
    'DEU': 'Germany', 'GRC': 'Greece', 'HUN': 'Hungary', 'ISL': 'Iceland',
    'IRL': 'Ireland', 'ISR': 'Israel', 'ITA': 'Italy', 'JPN': 'Japan',
    'KOR': 'Korea', 'LVA': 'Latvia', 'LTU': 'Lithuania', 'LUX': 'Luxembourg',
    'MEX': 'Mexico', 'NLD': 'Netherlands', 'NZL': 'New Zealand', 'NOR': 'Norway',
    'POL': 'Poland', 'PRT': 'Portugal', 'SVK': 'Slovak Republic',
    'SVN': 'Slovenia', 'ESP': 'Spain', 'SWE': 'Sweden', 'CHE': 'Switzerland',
    'TUR': 'Türkiye', 'GBR': 'United Kingdom', 'USA': 'United States',
    
    # Non-OECD Partner Countries / Other
    'IDN': 'Indonesia', 'IND': 'India', 'CHN': 'China', 'BRA': 'Brazil',
    'ARG': 'Argentina', 'ZAF': 'South Africa', 'ROU': 'Romania', 'PER': 'Peru',
    'SAU': 'Saudi Arabia', 'RUS': 'Russia', 'BGR': 'Bulgaria', 'HRV': 'Croatia',
    
    # Pre-calculated Aggregates
    'OECD': 'OECD Average', 'EU25': 'EU25 Average', 'G20': 'G20 Average'
}

# --- ADD THESE NEW GROUPINGS ---

# Note: Classification is based on general IMF/World Bank lists
ADVANCED_ECONOMIES = [
    'Australia', 'Austria', 'Belgium', 'Canada', 'Czechia', 'Denmark',
    'Estonia', 'Finland', 'France', 'Germany', 'Greece', 'Hungary',
    'Iceland', 'Ireland', 'Israel', 'Italy', 'Japan', 'Korea', 'Latvia',
    'Lithuania', 'Luxembourg', 'Netherlands', 'New Zealand', 'Norway',
    'Poland', 'Portugal', 'Slovak Republic', 'Slovenia', 'Spain',
    'Sweden', 'Switzerland', 'United Kingdom', 'United States',
    'Bulgaria', 'Croatia', 'Romania'
]

EMERGING_DEVELOPING_ECONOMIES = [
    # OECD Emerging
    'Chile', 'Colombia', 'Costa Rica', 'Mexico', 'Türkiye',
    # Partner Countries
    'Indonesia', 'India', 'China', 'Brazil', 'Argentina',
    'South Africa', 'Peru', 'Saudi Arabia', 'Russia'
]

# List of aggregate groups already in the data
AGGREGATE_GROUPS = ['OECD Average', 'EU25 Average', 'G20 Average']

# HELPER FUNCTION: Assign country groups
def assign_group(country):
    if country in ADVANCED_ECONOMIES:
        return 'Advanced Economies'
    if country in EMERGING_DEVELOPING_ECONOMIES:
        return 'Emerging/Developing Economies'
    if country in AGGREGATE_GROUPS:
        return country # Return the group name itself
    return 'Other'


# ----------------------------------------------------------------------------
# DATA LOADING AND PREPROCESSING
# ----------------------------------------------------------------------------
print("\nLoading and preprocessing dataset (this may take a moment)...")

# Load the data
ddf = dd.read_parquet('oecd_data.parquet')

# Compute the relevant subset of data once
df_computed = ddf[
    (ddf['MEASURE'] == 'POP') | (ddf['LABOUR_FORCE_STATUS'].isin(['EMP', 'POP']))
].compute()

print(f"Data computed. {len(df_computed)} rows loaded into memory.")

# Apply all mappings
print("Applying human-readable labels...")
df_computed['Education Level'] = df_computed['ATTAINMENT_LEV'].apply(categorize_education)

# This first line is OK because it fills with a simple string 'Other/Total'
df_computed['Field of Education'] = df_computed['EDUCATION_FIELD'].map(FIELD_MAP).fillna('Other/Total')

# --- APPLY .astype(object) TO THE LINES BELOW ---

# Add .astype(object) before .fillna()
df_computed['Age Group'] = df_computed['AGE'].map(AGE_MAP).astype(object).fillna(df_computed['AGE'])
df_computed['Gender'] = df_computed['SEX'].map(SEX_MAP).astype(object).fillna(df_computed['SEX'])
df_computed['Birth Place'] = df_computed['BIRTH_PLACE'].map(BIRTH_PLACE_MAP).astype(object).fillna(df_computed['BIRTH_PLACE'])
df_computed['Country'] = df_computed['REF_AREA'].map(COUNTRY_MAP).astype(object).fillna(df_computed['REF_AREA'])
df_computed['Group'] = df_computed['Country'].apply(assign_group)

print("Data preprocessing complete.")
# HELPER FUNCTION: Categorize education levels

print("Unique Country List:", df_computed['Country'].unique())

def categorize_education(attainment_level):
    if pd.isna(attainment_level):
        return 'Other'
    level_str = str(attainment_level)
    if any(x in level_str for x in ['ISCED11A_0', 'ISCED11A_1', 'ISCED11A_2']):
        return 'Primary'
    elif any(x in level_str for x in ['ISCED11A_3', 'ISCED11A_4']):
        return 'Secondary'
    elif any(x in level_str for x in ['ISCED11A_5', 'ISCED11A_6', 'ISCED11A_7', 'ISCED11A_8']):
        return 'Tertiary'
    return 'Other'

# HELPER FUNCTION: Clean facet labels
def clean_facet_labels(fig):
    """
    Finds all facet labels (annotations) and splits 'Column=Value' to just 'Value'.
    """
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    return fig
# ============================================================================
# VIZ 1: Education Attainment Trends Over Time
# ============================================================================
print("\n[1/7] Viz 1: Education Attainment Trends Over Time...")

# Filter data
trends = df_computed[
    (df_computed['EDUCATION_FIELD'] == '_T') &
    (df_computed['SEX'] == '_T') &
    (df_computed['AGE'] == 'Y25T64') &
    (df_computed['MEASURE'] == 'POP')
]

# Aggregate data
trends_agg = trends.groupby(['TIME_PERIOD', 'Education Level', 'Country'])['OBS_VALUE'].sum().reset_index()

# Get top 8 countries
top_countries = trends_agg.groupby('Country')['OBS_VALUE'].sum().nlargest(8).index.tolist()
trends_agg = trends_agg[trends_agg['Country'].isin(top_countries)]

# Calculate percentages
trends_agg['PERCENTAGE'] = trends_agg.groupby(['TIME_PERIOD', 'Country'])['OBS_VALUE'].transform(
    lambda x: (x / x.sum() * 100) if x.sum() > 0 else 0
)

fig1 = px.line(
    trends_agg,
    x='TIME_PERIOD',
    y='PERCENTAGE',
    color='Education Level',
    facet_col='Country',
    facet_col_wrap=4,
    title='Education Attainment Trends Over Time (Primary, Secondary, Tertiary)',
    labels={
        'TIME_PERIOD': 'Year',
        'PERCENTAGE': 'Percentage (%)',
        'Education Level': 'Education Level',
        'Country': 'Country'
    },
    height=1000,
    markers=True
)
fig1.update_layout(template='plotly_white', hovermode='x unified', font=dict(size=10))
fig1 = clean_facet_labels(fig1)  # Clean facet labels
fig1.write_html('viz_01_education_trends_over_time.html')
print("  ✓ Saved: viz_01_education_trends_over_time.html")

# ============================================================================
# VIZ 2: Age Group Performers (Top 2, Bottom 2, Averages) [REVISED]
# ============================================================================
print("\n[2/7] Viz 2: Education Attainment by Age Groups (Performers)...")

# Filter data for all countries
age_trends = df_computed[
    (df_computed['EDUCATION_FIELD'] == '_T') &
    (df_computed['SEX'] == '_T') &
    (df_computed['MEASURE'] == 'POP') &
    (df_computed['BIRTH_PLACE'] == '_T')
]

# Aggregate data
age_agg_all = age_trends.groupby(
    ['TIME_PERIOD', 'Age Group', 'Education Level', 'Country', 'Group']
)['OBS_VALUE'].sum().reset_index()

# Get latest year
latest_year = age_agg_all['TIME_PERIOD'].max()
if pd.notna(latest_year):
    print(f"  > Filtering Age Group data for latest available year: {latest_year}")
    age_agg_latest = age_agg_all[age_agg_all['TIME_PERIOD'] == latest_year]

    # --- NEW LOGIC: Calculate Averages and find Top/Bottom ---

    # 1. Calculate the average population for our custom groups
    avg_age_groups = age_agg_latest.groupby(
        ['Group', 'Age Group', 'Education Level']
    )['OBS_VALUE'].mean().reset_index()
    
    # 2. Get the pre-calculated 'OECD Average'
    oecd_avg_age = age_agg_latest[age_agg_latest['Country'] == 'OECD Average']
    
    # 3. Find Top 2 & Bottom 2 (by Tertiary, Ages 25-34 population)
    rank_metric = age_agg_latest[
        (age_agg_latest['Education Level'] == 'Tertiary') &
        (age_agg_latest['Age Group'] == 'Ages 25-34') &
        (~age_agg_latest['Group'].isin(AGGREGATE_GROUPS)) &
        (age_agg_latest['Group'] != 'Other')
    ].sort_values(by='OBS_VALUE')
    
    top_2_countries = rank_metric.nlargest(2, 'OBS_VALUE')['Country'].tolist()
    bottom_2_countries = rank_metric.nsmallest(2, 'OBS_VALUE')['Country'].tolist()

    # 4. Filter the main data for these selected countries
    top_bottom_2_data = age_agg_latest[
        age_agg_latest['Country'].isin(top_2_countries + bottom_2_countries)
    ]

    # 5. Combine everything for plotting
    plot_data = pd.concat([
        top_bottom_2_data,
        oecd_avg_age,
        avg_age_groups[avg_age_groups['Group'] == 'Advanced Economies'],
        avg_age_groups[avg_age_groups['Group'] == 'Emerging/Developing Economies']
    ], ignore_index=True)
    
    # Fill in 'Country' name from 'Group' for the averages
    plot_data['Country'] = plot_data['Country'].fillna(plot_data['Group'])

    # --- THIS IS THE NEW PLOT ---
    fig2 = px.bar(
        plot_data,
        x='Age Group',
        y='OBS_VALUE',
        color='Education Level',
        facet_col='Country', # Facet by Country/Group
        facet_col_wrap=3,
        title=f'Education Attainment by Age Group: Performers & Averages (Year {latest_year})',
        labels={
            'OBS_VALUE': 'Population',
            'Education Level': 'Education Level',
            'Country': 'Country / Group',
            'Age Group': 'Age Group'
        },
        height=800,
        barmode='group'
    )
    fig2.update_layout(template='plotly_white') # <-- Set to light theme
    fig2 = clean_facet_labels(fig2) # <-- Apply our facet label cleaner
    fig2.write_html('viz_02_education_by_age_groups.html')
    print("  ✓ Saved: viz_02_education_by_age_groups.html")
else:
    print("  ✗ Could not generate Viz 2: Latest year data not found.")

# ============================================================================
# VIZ 3: Education Field Trends (FIXED as 100% Stacked Bar)
# ============================================================================
print("\n[3/7] Viz 3: Education Field Trends...")

# Filter data
field_trends = df_computed[
    (df_computed['EDUCATION_FIELD'] != '_T') &
    (df_computed['SEX'] == '_T') &
    (df_computed['AGE'] == 'Y25T64') &
    (df_computed['MEASURE'] == 'POP')
]

# Aggregate data
field_agg = field_trends.groupby(['TIME_PERIOD', 'Field of Education', 'Country'])['OBS_VALUE'].sum().reset_index()

# --- NEW: Calculate Percentages ---
# This calculates the percentage of each field *within* its Time/Country group
field_agg['PERCENTAGE'] = field_agg.groupby(
    ['TIME_PERIOD', 'Country']
)['OBS_VALUE'].transform(lambda x: (x / x.sum() * 100) if x.sum() > 0 else 0)

# Get top countries
top_countries = field_agg.groupby('Country')['OBS_VALUE'].sum().nlargest(4).index.tolist()
field_agg = field_agg[field_agg['Country'].isin(top_countries)]

# --- UPDATED: Plot PERCENTAGE, not OBS_VALUE ---
fig3 = px.bar(
    field_agg,
    x='TIME_PERIOD',
    y='PERCENTAGE',  # <-- Use PERCENTAGE
    color='Field of Education',
    facet_col='Country',
    facet_col_wrap=2,
    title='Mix of Education Fields Over Time (Top 4 Countries)', # <-- New Title
    labels={
        'TIME_PERIOD': 'Year',
        'PERCENTAGE': 'Percentage of Population (%)', # <-- New Label
        'Field of Education': 'Field of Education',
        'Country': 'Country'
    },
    height=800,
    barmode='stack'
)
fig3.update_layout(template='plotly_white')
fig3 = clean_facet_labels(fig3) # <-- Apply our new label cleaner
fig3.write_html('viz_03_education_field_trends.html')
print("  ✓ Saved: viz_03_education_field_trends.html")
# ============================================================================
# VIZ 4: Country Growth Analysis (FIXED)
# ============================================================================
print("\n[4/7] Viz 4: Country Growth Analysis...")

# Filter data - ADDED BIRTH_PLACE FILTER to prevent stacking bug
growth_data = df_computed[
    (df_computed['EDUCATION_FIELD'] == '_T') &
    (df_computed['SEX'] == '_T') &
    (df_computed['AGE'] == 'Y25T64') &
    (df_computed['MEASURE'] == 'POP') &
    (df_computed['BIRTH_PLACE'] == '_T') # <--- ADDED FILTER
]

# Aggregate data
growth_agg = growth_data.groupby(['TIME_PERIOD', 'Country', 'Education Level'])['OBS_VALUE'].sum().reset_index()

years = sorted(growth_agg['TIME_PERIOD'].unique())
if len(years) >= 2:
    first_year = years[0]
    last_year = years[-1]

    growth_calc = growth_agg[
        (growth_agg['TIME_PERIOD'] == first_year) | (growth_agg['TIME_PERIOD'] == last_year)
    ].pivot_table(values='OBS_VALUE', index=['Country', 'Education Level'], columns='TIME_PERIOD').reset_index()

    # Handle potential missing years for a country
    if first_year in growth_calc.columns and last_year in growth_calc.columns:
        growth_calc['GROWTH_RATE'] = ((growth_calc[last_year] - growth_calc[first_year]) / growth_calc[first_year] * 100).fillna(0)
        growth_calc = growth_calc[growth_calc['GROWTH_RATE'] != 0] # Remove 0 growth

        # --- NEW LOGIC ---
        # 1. Find top 5 countries based on TERTIARY education growth
        top_tertiary_growth_countries = growth_calc[
            growth_calc['Education Level'] == 'Tertiary'
        ].nlargest(5, 'GROWTH_RATE')['Country'].tolist()

        # 2. Filter the main growth data to get ALL levels for those 5 countries
        growth_calc_top_countries = growth_calc[
            growth_calc['Country'].isin(top_tertiary_growth_countries)
        ]

        # 3. Plot this new, complete DataFrame
        fig4 = px.bar(
            growth_calc_top_countries, # <--- Use new DataFrame
            x='Country',
            y='GROWTH_RATE',
            color='Education Level',
            title=f'Education Growth Rates for Top 5 Countries (by Tertiary Growth, {first_year}-{last_year})',
            labels={
                'GROWTH_RATE': 'Growth Rate (%)',
                'Country': 'Country',
                'Education Level': 'Education Level'
            },
            height=600,
            barmode='group'
        )
        fig4.update_layout(template='plotly_white', hovermode='x unified')
        fig4.write_html('viz_04_country_growth_analysis.html')
        print("  ✓ Saved: viz_04_country_growth_analysis.html")
    else:
        print("  ✗ Could not generate Viz 4: Missing first or last year data in pivot.")
else:
    print("  ✗ Could not generate Viz 4: Not enough years of data for comparison.")


# ============================================================================
# VIZ 5: LFPR Performers (Top 2, Bottom 2, and Averages) [REVISED]
# ============================================================================
print("\n[5/7] Viz 5: Labor Force Participation (Performers)...")

# 1. Get Employment Data for ALL countries
common_filters = (
    (df_computed['EDUCATION_FIELD'] == '_T') &
    (df_computed['SEX'] == '_T') &
    (df_computed['AGE'] == 'Y25T64') &
    (df_computed['BIRTH_PLACE'] == '_T')
)

emp_data = df_computed[
    common_filters &
    (df_computed['LABOUR_FORCE_STATUS'] == 'EMP')
]
emp_agg = emp_data.groupby(
    ['Country', 'Group', 'Education Level', 'TIME_PERIOD']
)['OBS_VALUE'].sum().reset_index().rename(columns={'OBS_VALUE': 'Employed'})


# 2. Get Population Data for ALL countries
pop_data = df_computed[
    common_filters &
    (df_computed['MEASURE'] == 'POP')
]
pop_agg = pop_data.groupby(
    ['Country', 'Group', 'Education Level', 'TIME_PERIOD']
)['OBS_VALUE'].sum().reset_index().rename(columns={'OBS_VALUE': 'Population'})


# 3. Merge the two datasets
lfpr_merged = pd.merge(
    emp_agg,
    pop_agg,
    on=['Country', 'Group', 'Education Level', 'TIME_PERIOD'],
    how='left'
)

# 4. Calculate LFPR
if 'Employed' in lfpr_merged.columns and 'Population' in lfpr_merged.columns:
    lfpr_merged['LFPR'] = (lfpr_merged['Employed'] / lfpr_merged['Population'] * 100).fillna(0)
    
    lfpr_result_all = lfpr_merged[(lfpr_merged['LFPR'] > 0) & (lfpr_merged['LFPR'] <= 100)].copy()

    # Get latest year
    latest_year = lfpr_result_all['TIME_PERIOD'].max()
    if pd.notna(latest_year):
        print(f"  > Filtering LFPR data for latest available year: {latest_year}")
        lfpr_result_all = lfpr_result_all[lfpr_result_all['TIME_PERIOD'] == latest_year]

        # --- NEW LOGIC: Calculate Averages and find Top/Bottom ---

        # 1. Calculate the average LFPR for our custom groups
        avg_lfpr_groups = lfpr_result_all.groupby(
            ['Group', 'Education Level']
        )['LFPR'].mean().reset_index()
        
        # 2. Get the pre-calculated 'OECD Average'
        oecd_avg_lfpr = lfpr_result_all[lfpr_result_all['Country'] == 'OECD Average']
        
        # 3. Find Top 2 and Bottom 2 countries (based on Tertiary LFPR)
        tertiary_lfpr = lfpr_result_all[
            (lfpr_result_all['Education Level'] == 'Tertiary') &
            # Exclude aggregate groups from the individual ranking
            (~lfpr_result_all['Group'].isin(AGGREGATE_GROUPS)) &
            (lfpr_result_all['Group'] != 'Other')
        ].sort_values(by='LFPR')
        
        top_2_countries = tertiary_lfpr.nlargest(2, 'LFPR')['Country'].tolist()
        bottom_2_countries = tertiary_lfpr.nsmallest(2, 'LFPR')['Country'].tolist()

        # 4. Filter the main data for these selected countries
        top_bottom_2_data = lfpr_result_all[
            lfpr_result_all['Country'].isin(top_2_countries + bottom_2_countries)
        ]

        # 5. Combine everything into one DataFrame for plotting
        plot_data = pd.concat([
            top_bottom_2_data,
            oecd_avg_lfpr,
            avg_lfpr_groups[avg_lfpr_groups['Group'] == 'Advanced Economies'],
            avg_lfpr_groups[avg_lfpr_groups['Group'] == 'Emerging/Developing Economies']
        ], ignore_index=True)
        
        # In our combined data, the 'Country' for group averages is blank.
        # We fill it with the 'Group' name so it appears on the chart.
        plot_data['Country'] = plot_data['Country'].fillna(plot_data['Group'])

        # --- THIS IS THE NEW PLOT ---
        fig5 = px.bar(
            plot_data,
            x='Country',
            y='LFPR',
            color='Education Level',
            barmode='group',
            title=f'Labor Force Participation: Top 2, Bottom 2, and Averages (Year {latest_year})',
            labels={
                'Country': 'Country / Group',
                'LFPR': 'Labor Force Participation Rate (%)',
                'Education Level': 'Education Level'
            },
            height=600,
            # Order the bars by Tertiary LFPR
            category_orders={'Country': plot_data[plot_data['Education Level']=='Tertiary']
                                         .sort_values(by='LFPR')['Country'].tolist()}
        )
        fig5.update_layout(template='plotly_white')
        fig5.write_html('viz_05_education_vs_lfpr.html')
        print("  ✓ Saved: viz_05_education_vs_lfpr.html")
    else:
        print("  ✗ Could not generate Viz 5: Latest year data not found.")
else:
    print("  ✗ Could not generate Viz 5: 'Employed' or 'Population' data missing after merge.")

# ============================================================================
# VIZ 6: Migration Performers (Top 2, Bottom 2, Averages) [REVISED]
# ============================================================================
print("\n[6/7] Viz 6: Education Attainment vs Migration (Performers)...")

# Get latest year
latest_year = df_computed['TIME_PERIOD'].max()
print(f"  > Filtering Migration data for latest available year: {latest_year}")

# Filter data
migration_data = df_computed[
    (df_computed['EDUCATION_FIELD'] == '_T') &
    (df_computed['SEX'] == '_T') &
    (df_computed['AGE'] == 'Y25T64') &
    (df_computed['BIRTH_PLACE'] != '_T') & # Keep Native & Foreign
    (df_computed['TIME_PERIOD'] == latest_year)
]

# Aggregate data
migration_agg_all = migration_data.groupby(
    ['Country', 'Group', 'Birth Place', 'Education Level']
)['OBS_VALUE'].sum().reset_index()

# --- NEW LOGIC: Calculate Averages and find Top/Bottom ---

# 1. Calculate the average population for our custom groups
avg_mig_groups = migration_agg_all.groupby(
    ['Group', 'Birth Place', 'Education Level']
)['OBS_VALUE'].mean().reset_index()

# 2. Get the pre-calculated 'OECD Average'
oecd_avg_mig = migration_agg_all[migration_agg_all['Country'] == 'OECD Average']

# 3. Find Top 2 & Bottom 2 (by Foreign-Born, Tertiary population)
rank_metric = migration_agg_all[
    (migration_agg_all['Education Level'] == 'Tertiary') &
    (migration_agg_all['Birth Place'] == 'Foreign-Born') &
    (~migration_agg_all['Group'].isin(AGGREGATE_GROUPS)) &
    (migration_agg_all['Group'] != 'Other')
].sort_values(by='OBS_VALUE')

top_2_countries = rank_metric.nlargest(2, 'OBS_VALUE')['Country'].tolist()
bottom_2_countries = rank_metric.nsmallest(2, 'OBS_VALUE')['Country'].tolist()

# 4. Filter the main data for these selected countries
top_bottom_2_data = migration_agg_all[
    migration_agg_all['Country'].isin(top_2_countries + bottom_2_countries)
]

# 5. Combine everything for plotting
plot_data = pd.concat([
    top_bottom_2_data,
    oecd_avg_mig,
    avg_mig_groups[avg_mig_groups['Group'] == 'Advanced Economies'],
    avg_mig_groups[avg_mig_groups['Group'] == 'Emerging/Developing Economies']
], ignore_index=True)

# Fill in 'Country' name from 'Group' for the averages
plot_data['Country'] = plot_data['Country'].fillna(plot_data['Group'])
# Filter out 'Not Applicable' for a cleaner plot
plot_data = plot_data[plot_data['Birth Place'] != 'Not Applicable']

# --- THIS IS THE NEW PLOT ---
fig6 = px.bar(
    plot_data,
    x='Country',
    y='OBS_VALUE',
    color='Education Level',
    facet_col='Birth Place', # Facet by Native-Born vs Foreign-Born
    title=f'Education of Native vs Foreign-Born: Performers & Averages (Year {latest_year})',
    labels={
        'Country': 'Country / Group',
        'OBS_VALUE': 'Population',
        'Education Level': 'Education Level',
        'Birth Place': 'Birth Place'
    },
    height=700,
    barmode='group'
)
fig6.update_layout(template='plotly_white') # <-- Set to light theme
fig6 = clean_facet_labels(fig6) # <-- Apply our facet label cleaner
fig6.write_html('viz_06_education_vs_migration.html')
print("  ✓ Saved: viz_06_education_vs_migration.html")

# ============================================================================
# VIZ 7: Gender Distribution in Education (FIXED as 100% Stacked Bar)
# ============================================================================
print("\n[7/7] Viz 7: Gender Distribution in Education...")

# Filter data
gender_data = df_computed[
    (df_computed['EDUCATION_FIELD'] == '_T') &
    (df_computed['SEX'] != '_T') &               # <--- Keeps 'Female' and 'Male'
    (df_computed['AGE'] == 'Y25T64') &
    (df_computed['MEASURE'] == 'POP') &
    (df_computed['BIRTH_PLACE'] == '_T')         # <--- ADDED FILTER
]

# Aggregate data
gender_agg = gender_data.groupby(
    ['TIME_PERIOD', 'Education Level', 'Gender', 'Country']
)['OBS_VALUE'].sum().reset_index()

# --- NEW: Calculate Percentages ---
gender_agg['PERCENTAGE'] = gender_agg.groupby(
    ['TIME_PERIOD', 'Country', 'Education Level']
)['OBS_VALUE'].transform(lambda x: (x / x.sum() * 100) if x.sum() > 0 else 0)

# Get top countries
top_countries = gender_agg.groupby('Country')['OBS_VALUE'].sum().nlargest(4).index.tolist()
gender_agg = gender_agg[gender_agg['Country'].isin(top_countries)]

# --- UPDATED: Plot PERCENTAGE ---
fig7 = px.bar(
    gender_agg,
    x='TIME_PERIOD',
    y='PERCENTAGE',  # <--- Plot PERCENTAGE
    color='Gender',
    facet_col='Country',
    facet_row='Education Level',
    title='Gender Ratio in Education Levels (Top 4 Countries)', # <-- New Title
    labels={
        'TIME_PERIOD': 'Year',
        'PERCENTAGE': 'Percentage of Population (%)', # <-- New Label
        'Gender': 'Gender',
        'Country': 'Country',
        'Education Level': 'Education Level'
    },
    height=1000,
    barmode='stack' # <--- This makes it a 100% stacked bar
)
fig7.update_layout(template='plotly_white')
fig7 = clean_facet_labels(fig7) # <-- Apply our facet label cleaner
fig7.update_yaxes(range=[30, 70]) # <--- ADD THIS LINE
fig7.write_html('viz_07_gender_distribution.html')
print("  ✓ Saved: viz_07_gender_distribution.html")

# ============================================================================
# VIZ 8: Aggregate Group Trends (Advanced vs. Developing vs. OECD)
# ============================================================================
print("\n[8/9] Viz 8: Aggregate Group Trends...")

# Filter data (ensure all our "bug-fix" filters are here)
trends_data = df_computed[
    (df_computed['EDUCATION_FIELD'] == '_T') &
    (df_computed['SEX'] == '_T') &
    (df_computed['AGE'] == 'Y25T64') &
    (df_computed['MEASURE'] == 'POP') &
    (df_computed['BIRTH_PLACE'] == '_T')
]

# Calculate percentages for all individual countries
trends_agg = trends_data.groupby(
    ['TIME_PERIOD', 'Education Level', 'Country', 'Group']
)['OBS_VALUE'].sum().reset_index()

trends_agg['PERCENTAGE'] = trends_agg.groupby(
    ['TIME_PERIOD', 'Country']
)['OBS_VALUE'].transform(lambda x: (x / x.sum() * 100) if x.sum() > 0 else 0)

# Calculate the *average* percentage for our new custom groups
group_agg = trends_agg.groupby(
    ['TIME_PERIOD', 'Education Level', 'Group']
)['PERCENTAGE'].mean().reset_index()

# Get the 'OECD Average' data which is pre-calculated
# We filter for it by its 'Group' name, which we set in assign_group
oecd_avg = group_agg[group_agg['Group'] == 'OECD Average']

# Combine our new groups with the pre-calculated OECD average
final_agg_trends = pd.concat([
    group_agg[group_agg['Group'] == 'Advanced Economies'],
    group_agg[group_agg['Group'] == 'Emerging/Developing Economies'],
    oecd_avg
])

fig8 = px.line(
    final_agg_trends,
    x='TIME_PERIOD',
    y='PERCENTAGE',
    color='Education Level',
    facet_row='Group',  # Use rows to separate the 3 groups
    title='Education Attainment Trends: Group Averages',
    labels={
        'TIME_PERIOD': 'Year',
        'PERCENTAGE': 'Average Percentage of Population (%)',
        'Education Level': 'Education Level',
        'Group': 'Country Group'
    },
    height=900,
    markers=True
)
fig8.update_layout(template='plotly_white', hovermode='x unified')
fig8 = clean_facet_labels(fig8) # Apply our label cleaner
fig8.write_html('viz_08_aggregate_group_trends.html')
print("  ✓ Saved: viz_08_aggregate_group_trends.html")

# ============================================================================
# VIZ 9: Top 5 & Bottom 5 Performers (Tertiary Education Growth)
# ============================================================================
print("\n[9/9] Viz 9: Top 5 & Bottom 5 Performers...")

# This plot depends on 'growth_calc' from VIZ 4.
# We must ensure VIZ 4 has been run and 'growth_calc' was created.
if 'growth_calc' in locals():
    
    # Filter for Tertiary education
    tertiary_growth = growth_calc[
        (growth_calc['Education Level'] == 'Tertiary') &
        # Exclude all pre-calculated aggregates from the ranking
        (~growth_calc['Country'].isin(AGGREGATE_GROUPS))
    ].sort_values(by='GROWTH_RATE')

    # Get Top 5 and Bottom 5
    top_5 = tertiary_growth.nlargest(5, 'GROWTH_RATE')
    bottom_5 = tertiary_growth.nsmallest(5, 'GROWTH_RATE')
    
    # Add a column to identify them
    top_5['Performance'] = 'Top 5 Fastest Growth'
    bottom_5['Performance'] = 'Bottom 5 Slowest Growth'

    # Combine
    top_bottom_5 = pd.concat([top_5, bottom_5])

    # Plot
    fig9 = px.bar(
        top_bottom_5,
        x='GROWTH_RATE',
        y='Country',
        color='Performance',  # Color by Top vs Bottom
        color_discrete_map={
            'Top 5 Fastest Growth': 'green',
            'Bottom 5 Slowest Growth': 'red'
        },
        title=f'Top 5 & Bottom 5 Countries by Tertiary Education Growth ({first_year}-{last_year})',
        labels={
            'GROWTH_RATE': 'Growth Rate in Tertiary Education (%)',
            'Country': 'Country'
        },
        height=600,
        orientation='h' # Horizontal bar chart is easier to read
    )
    fig9.update_layout(template='plotly_white', 
                       yaxis={'categoryorder':'total ascending'})
    fig9.write_html('viz_09_top_bottom_5_performers.html')
    print("  ✓ Saved: viz_09_top_bottom_5_performers.html")
else:
    print("  ✗ Could not generate Viz 9: 'growth_calc' from VIZ 4 not found.")

print("\n" + "=" * 80)
print("VISUALIZATION GENERATION COMPLETE")
print("=" * 80)
print("Open the HTML files in your web browser to explore!")

# -- ADDITIONAL GENDER VISUALIZATIONS --
# gender_lfpr_analysis.py
# Advanced Gender & LFPR Analysis with Interactive Visualizations

import dask.dataframe as dd
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

print('=' * 80)
print('GENDER & LFPR ANALYSIS - INTERACTIVE VISUALIZATIONS')
print('=' * 80)

# Load Parquet
print('Loading Parquet dataset...')
ddf = dd.read_parquet('oecd_data.parquet')


# -- END ADDITIONAL VISUALS --
