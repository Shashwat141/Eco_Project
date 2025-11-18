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
    'FIELD001': 'Education', 'FIELD002': 'Arts/Humanities',
    'FIELD003': 'Social Sciences', 'FIELD004': 'Business/Law',
    'FIELD005': 'Natural Sciences', 'FIELD006': 'ICT',
    'FIELD007': 'Engineering', 'FIELD008': 'Agriculture',
    'FIELD009': 'Health/Welfare', 'FIELD010': 'Services',
    '_T': 'Total'
}

# Age group mapping
AGE_MAP = {
    'Y25T64': 'Ages 25-64', 'Y15T64': 'Ages 15-64', 'Y15T24': 'Ages 15-24',
    'Y25T34': 'Ages 25-34', 'Y35T44': 'Ages 35-44', 'Y45T54': 'Ages 45-54',
    'Y55T64': 'Ages 55-64', '_T': 'All Ages'
}

# Gender mapping
SEX_MAP = {'F': 'Female', 'M': 'Male', '_T': 'Total'}

# Birth place mapping
BIRTH_PLACE_MAP = {
    'NATIVE': 'Native-Born', 'FOREIGN': 'Foreign-Born',
    '_T': 'Total', '_Z': 'Not Applicable'
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
    'OECD': 'OECD Average (Broken)', # We will ignore this one
    'EU25': 'EU25 Average',
    'G20': 'G20 Average'
}

# --- NEW GROUPINGS ---

# List of all individual OECD countries in your dataset
OECD_MEMBER_COUNTRIES = [
    'Portugal', 'Ireland', 'Israel', 'Hungary', 'France', 'Netherlands',
    'Finland', 'Slovenia', 'Latvia', 'Iceland', 'Luxembourg', 'Czechia',
    'Italy', 'Spain', 'Germany', 'New Zealand', 'Norway', 'Sweden',
    'Slovak Republic', 'Greece', 'Australia', 'Denmark', 'Mexico', 'Türkiye',
    'United Kingdom', 'Switzerland', 'Lithuania', 'Estonia', 'Japan',
    'Poland', 'United States', 'Korea', 'Canada', 'Colombia', 'Austria',
    'Costa Rica', 'Belgium', 'Chile'
]

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
    'Chile', 'Colombia', 'Costa Rica', 'Mexico', 'Türkiye', 'Indonesia',
    'India', 'China', 'Brazil', 'Argentina', 'South Africa', 'Peru',
    'Saudi Arabia', 'Russia'
]

# List of aggregate groups *we will calculate* or *are pre-calculated*
AGGREGATE_GROUPS = [
    'OECD Average (Calculated)', # Our new, reliable group
    'Advanced Economies',
    'Emerging/Developing Economies',
    'EU25 Average',
    'G20 Average'
]

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

# HELPER FUNCTION: Categorize education levels
def categorize_education(attainment_level):
    if pd.isna(attainment_level): return 'Other'
    level_str = str(attainment_level)
    if any(x in level_str for x in ['ISCED11A_0', 'ISCED11A_1', 'ISCED11A_2']): return 'Primary'
    elif any(x in level_str for x in ['ISCED11A_3', 'ISCED11A_4']): return 'Secondary'
    elif any(x in level_str for x in ['ISCED11A_5', 'ISCED11A_6', 'ISCED11A_7', 'ISCED11A_8']): return 'Tertiary'
    return 'Other'

# HELPER FUNCTION: Assign country groups
def assign_group(country):
    if country in OECD_MEMBER_COUNTRIES:
        return 'OECD Average (Calculated)' # Assign to our new group
    if country in ADVANCED_ECONOMIES:
        return 'Advanced Economies'
    if country in EMERGING_DEVELOPING_ECONOMIES:
        return 'Emerging/Developing Economies'
    if country == 'EU25 Average' or country == 'G20 Average':
        return country
    if country == 'OECD Average (Broken)':
        return 'Other (Ignored)'
    if country == 'India': # Keep India separate
        return 'India'
    return 'Other'

# HELPER FUNCTION: Clean facet labels
def clean_facet_labels(fig):
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    return fig

# Apply all mappings
print("Applying human-readable labels...")
df_computed['Education Level'] = df_computed['ATTAINMENT_LEV'].apply(categorize_education)
df_computed['Field of Education'] = df_computed['EDUCATION_FIELD'].map(FIELD_MAP).fillna('Other/Total')
df_computed['Age Group'] = df_computed['AGE'].map(AGE_MAP).astype(object).fillna(df_computed['AGE'])
df_computed['Gender'] = df_computed['SEX'].map(SEX_MAP).astype(object).fillna(df_computed['SEX'])
df_computed['Birth Place'] = df_computed['BIRTH_PLACE'].map(BIRTH_PLACE_MAP).astype(object).fillna(df_computed['BIRTH_PLACE'])
df_computed['Country'] = df_computed['REF_AREA'].map(COUNTRY_MAP).astype(object).fillna(df_computed['REF_AREA'])
df_computed['Group'] = df_computed['Country'].apply(assign_group)

# Filter out all ignored/other data at the start
df_computed = df_computed[df_computed['Group'] != 'Other'].copy()

print("Data preprocessing complete.")

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
# VIZ 1: Attainment Trends (Performers & Averages) [REVISED]
# ============================================================================
print("\n[1/7] Viz 1: Education Attainment Trends (Performers)...")

# Filter data (we use the 'trends_agg' from VIZ 8)
if 'trends_agg' in locals():
    # Get latest year
    latest_year = trends_agg['TIME_PERIOD'].max()
    print(f"  > Ranking countries based on latest year: {latest_year}")
    
    # --- NEW LOGIC: Find Top/Bottom Performers ---
    
    # 1. Get data for the latest year
    latest_year_data = trends_agg[trends_agg['TIME_PERIOD'] == latest_year]

    # 2. Get the 'Advanced' & 'Emerging' group averages
    group_avg_data = trends_agg.groupby(
        ['TIME_PERIOD', 'Education Level', 'Group']
    )['PERCENTAGE'].mean().reset_index()
    
    adv_avg = group_avg_data[group_avg_data['Group'] == 'Advanced Economies']
    emg_avg = group_avg_data[group_avg_data['Group'] == 'Emerging/Developing Economies']
    
    # 3. Get the pre-calculated 'OECD Average'
    oecd_avg = trends_agg[trends_agg['Country'] == 'OECD Average']
    
    # 4. Find Top 2 & Bottom 2 (by Tertiary % in latest year)
    rank_metric = latest_year_data[
        (latest_year_data['Education Level'] == 'Tertiary') &
        (~latest_year_data['Group'].isin(AGGREGATE_GROUPS)) &
        (latest_year_data['Group'] != 'Other')
    ].sort_values(by='PERCENTAGE')
    
    top_2_countries = rank_metric.nlargest(2, 'PERCENTAGE')['Country'].tolist()
    bottom_2_countries = rank_metric.nsmallest(2, 'PERCENTAGE')['Country'].tolist()

    # 5. Filter the main trend data for these selected countries
    top_bottom_2_data = trends_agg[
        trends_agg['Country'].isin(top_2_countries + bottom_2_countries)
    ]
    
    # 6. Combine everything for plotting
    plot_data = pd.concat([
        top_bottom_2_data,
        oecd_avg,
        # We need the full trend data for the groups, not just one year
        trends_agg[trends_agg['Country'].isin(adv_avg['Country'])],
        trends_agg[trends_agg['Country'].isin(emg_avg['Country'])]
    ], ignore_index=True)
    
    # Fill 'Country' from 'Group' for averages
    plot_data['Country'] = plot_data['Country'].fillna(plot_data['Group'])

    # --- THIS IS THE NEW PLOT ---
    fig1 = px.line(
        plot_data,
        x='TIME_PERIOD',
        y='PERCENTAGE',
        color='Country',      # Each line is a Country/Group
        facet_row='Education Level', # Separate charts for Primary, Sec, Tert
        title='Education Attainment Trends: Performers & Averages',
        labels={
            'TIME_PERIOD': 'Year',
            'PERCENTAGE': 'Percentage of Population (%)',
            'Country': 'Country / Group'
        },
        height=900,
        markers=True
    )
    fig1.update_layout(template='plotly_white', hovermode='x unified')
    fig1 = clean_facet_labels(fig1)
    fig1.write_html('viz_01_education_trends_over_time.html')
    print("  ✓ Saved: viz_01_education_trends_over_time.html")
else:
    print("  ✗ Could not generate Viz 1: 'trends_agg' from VIZ 8 not found.")

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
# VIZ 3: Education Field Mix (Aggregate Groups) [REVISED]
# ============================================================================
print("\n[3/7] Viz 3: Education Field Mix by Group...")

# Get latest year
latest_year = df_computed['TIME_PERIOD'].max()
print(f"  > Filtering Field Mix data for latest available year: {latest_year}")

# Filter data
field_trends = df_computed[
    (df_computed['EDUCATION_FIELD'] != '_T') &
    (df_computed['SEX'] == '_T') &
    (df_computed['AGE'] == 'Y25T64') &
    (df_computed['MEASURE'] == 'POP') &
    (df_computed['BIRTH_PLACE'] == '_T') &
    (df_computed['TIME_PERIOD'] == latest_year)
]

# Aggregate data for all countries
field_agg_all = field_trends.groupby(
    ['Field of Education', 'Country', 'Group']
)['OBS_VALUE'].sum().reset_index()

# --- NEW LOGIC: Show Averages for Groups ---

# 1. Calculate the average population for our custom groups
avg_field_groups = field_agg_all.groupby(
    ['Group', 'Field of Education']
)['OBS_VALUE'].mean().reset_index()

# 2. Get the pre-calculated 'OECD Average'
oecd_avg_field = field_agg_all[field_agg_all['Country'] == 'OECD Average']

# 3. Combine the groups we want to plot
plot_data = pd.concat([
    avg_field_groups[avg_field_groups['Group'] == 'Advanced Economies'],
    avg_field_groups[avg_field_groups['Group'] == 'Emerging/Developing Economies'],
    oecd_avg_field
], ignore_index=True)

# Fill 'Country' from 'Group' for averages
plot_data['Country'] = plot_data['Country'].fillna(plot_data['Group'])

# 4. Calculate Percentage (for the 100% stack)
plot_data['PERCENTAGE'] = plot_data.groupby(
    ['Country']
)['OBS_VALUE'].transform(lambda x: (x / x.sum() * 100) if x.sum() > 0 else 0)


# --- THIS IS THE NEW PLOT ---
fig3 = px.bar(
    plot_data,
    x='Country',
    y='PERCENTAGE',
    color='Field of Education',
    title=f'Mix of Education Fields by Economic Group (Year {latest_year})',
    labels={
        'Country': 'Country Group',
        'PERCENTAGE': 'Percentage of Population (%)',
        'Field of Education': 'Field of Education'
    },
    height=600,
    barmode='stack'
)
# Add text labels on the bars
fig3.update_traces(texttemplate='%{y:.1f}%', textposition='inside')
fig3.update_layout(template='plotly_white')
fig3.write_html('viz_03_education_field_trends.html')
print("  ✓ Saved: viz_03_education_field_trends.html")
# ============================================================================
# VIZ 4: Growth Performers (Top 2, Bottom 2, Averages) [REVISED]
# ============================================================================
print("\n[4/7] Viz 4: Country Growth Analysis (Performers)...")

# Filter data
growth_data = df_computed[
    (df_computed['EDUCATION_FIELD'] == '_T') &
    (df_computed['SEX'] == '_T') &
    (df_computed['AGE'] == 'Y25T64') &
    (df_computed['MEASURE'] == 'POP') &
    (df_computed['BIRTH_PLACE'] == '_T')
]

# Aggregate data
growth_agg = growth_data.groupby(
    ['TIME_PERIOD', 'Country', 'Group', 'Education Level']
)['OBS_VALUE'].sum().reset_index()

years = sorted(growth_agg['TIME_PERIOD'].unique())
if len(years) >= 2:
    first_year = years[0]
    last_year = years[-1]

    growth_calc = growth_agg[
        (growth_agg['TIME_PERIOD'] == first_year) | (growth_agg['TIME_PERIOD'] == last_year)
    ].pivot_table(values='OBS_VALUE', index=['Country', 'Group', 'Education Level'], columns='TIME_PERIOD').reset_index()

    if first_year in growth_calc.columns and last_year in growth_calc.columns:
        growth_calc['GROWTH_RATE'] = ((growth_calc[last_year] - growth_calc[first_year]) / growth_calc[first_year] * 100).fillna(0)
        growth_calc = growth_calc[growth_calc['GROWTH_RATE'] != 0]

        # --- NEW LOGIC: Calculate Averages and find Top/Bottom ---

        # 1. Calculate the average growth rate for our custom groups
        avg_growth_groups = growth_calc.groupby(
            ['Group', 'Education Level']
        )['GROWTH_RATE'].mean().reset_index()
        
        # 2. Find Top 2 & Bottom 2 (by Tertiary Growth)
        rank_metric = growth_calc[
            (growth_calc['Education Level'] == 'Tertiary') &
            (growth_calc['Group'] != 'Other') &
            # Exclude all aggregates from ranking
            (~growth_calc['Group'].isin(AGGREGATE_GROUPS)) 
        ].sort_values(by='GROWTH_RATE')
        
        top_2_countries = rank_metric.nlargest(2, 'GROWTH_RATE')['Country'].tolist()
        bottom_2_countries = rank_metric.nsmallest(2, 'GROWTH_RATE')['Country'].tolist()

        # 3. Filter the main data for these
        top_bottom_2_data = growth_calc[
            growth_calc['Country'].isin(top_2_countries + bottom_2_countries)
        ]

        # 4. Combine everything for plotting
        plot_data = pd.concat([
            top_bottom_2_data,
            avg_growth_groups[avg_growth_groups['Group'] == 'OECD Average (Calculated)'],
            avg_growth_groups[avg_growth_groups['Group'] == 'Advanced Economies'],
            avg_growth_groups[avg_growth_groups['Group'] == 'Emerging/Developing Economies']
        ], ignore_index=True)
        
        plot_data['Country'] = plot_data['Country'].fillna(plot_data['Group'])

        # --- THIS IS THE NEW PLOT ---
        fig4 = px.bar(
            plot_data,
            x='Country',
            y='GROWTH_RATE',
            color='Education Level',
            title=f'Education Growth Rates: Performers & Averages ({first_year}-{last_year})',
            labels={
                'GROWTH_RATE': 'Growth Rate (%)',
                'Country': 'Country / Group',
                'Education Level': 'Education Level'
            },
            height=600,
            barmode='group',
            category_orders={'Country': plot_data[plot_data['Education Level']=='Tertiary']
                                         .sort_values(by='GROWTH_RATE')['Country'].tolist()}
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
# VIZ 7: Gender Distribution (Custom Comparison) [REVISED]
# ============================================================================
print("\n[7/7] Viz 7: Gender Distribution (Custom Comparison)...")

# Filter data
gender_data = df_computed[
    (df_computed['EDUCATION_FIELD'] == '_T') &
    (df_computed['SEX'] != '_T') &
    (df_computed['AGE'] == 'Y25T64') &
    (df_computed['MEASURE'] == 'POP') &
    (df_computed['BIRTH_PLACE'] == '_T')
]

# Aggregate data
gender_agg = gender_data.groupby(
    ['TIME_PERIOD', 'Education Level', 'Gender', 'Country', 'Group']
)['OBS_VALUE'].sum().reset_index()

# Calculate Percentages for all
gender_agg['PERCENTAGE'] = gender_agg.groupby(
    ['TIME_PERIOD', 'Country', 'Education Level']
)['OBS_VALUE'].transform(lambda x: (x / x.sum() * 100) if x.sum() > 0 else 0)

# --- NEW LOGIC: Get Custom Comparison Group ---

# 1. Calculate Averages for our groups
avg_gender_group = gender_agg.groupby(
    ['TIME_PERIOD', 'Education Level', 'Gender', 'Group']
)['PERCENTAGE'].mean().reset_index()

# Get 'OECD Average (Calculated)'
oecd_avg_gender = avg_gender_group[avg_gender_group['Group'] == 'OECD Average (Calculated)']

# Get 'Emerging/Developing Economies'
emerging_avg_gender = avg_gender_group[avg_gender_group['Group'] == 'Emerging/Developing Economies']

# 2. Get 'India'
india_gender = gender_agg[gender_agg['Country'] == 'India']

# 3. Find 'Worst OECD Performer'
# (Lowest % of Females in Tertiary Ed, latest year)
latest_year = gender_agg['TIME_PERIOD'].max()
rank_metric = gender_agg[
    (gender_agg['Education Level'] == 'Tertiary') &
    (gender_agg['Gender'] == 'Female') &
    (gender_agg['TIME_PERIOD'] == latest_year) &
    # Rank ONLY the individual OECD countries
    (gender_agg['Group'] == 'OECD Average (Calculated)')
].sort_values(by='PERCENTAGE')

if not rank_metric.empty:
    worst_country_name = rank_metric.nsmallest(1, 'PERCENTAGE')['Country'].iloc[0]
    print(f"  > Found 'Worst OECD Performer' (Female Tertiary %): {worst_country_name}")
    worst_country_gender = gender_agg[gender_agg['Country'] == worst_country_name]

    # 5. Combine all four datasets
    plot_data = pd.concat([
        oecd_avg_gender,
        emerging_avg_gender,
        india_gender,
        worst_country_gender
    ], ignore_index=True)
    
    # Fill 'Country' from 'Group' for the averages
    plot_data['Country'] = plot_data['Country'].fillna(plot_data['Group'])

    # --- THIS IS THE NEW PLOT ---
    fig7 = px.bar(
        plot_data,
        x='TIME_PERIOD',
        y='PERCENTAGE',
        color='Gender',
        facet_col='Country', # Facet by our 4 selected groups
        facet_row='Education Level',
        title='Gender Ratio: Custom Comparison (OECD Avg, Emerging Avg, India, Worst Performer)',
        labels={
            'TIME_PERIOD': 'Year',
            'PERCENTAGE': 'Percentage of Population (%)',
            'Gender': 'Gender',
            'Country': 'Country / Group',
            'Education Level': 'Education Level'
        },
        height=1000,
        barmode='stack'
    )
    
    fig7.update_layout(template='plotly_white') # <-- Set to light theme
    fig7 = clean_facet_labels(fig7) # <-- Apply facet label cleaner
    fig7.update_yaxes(range=[30, 70]) # <-- Apply Y-axis zoom
    
    fig7.write_html('viz_07_gender_distribution.html')
    print("  ✓ Saved: viz_07_gender_distribution.html")
else:
    print("  ✗ Could not generate Viz 7: Failed to find ranking metric for 'Worst Performer'.")

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
