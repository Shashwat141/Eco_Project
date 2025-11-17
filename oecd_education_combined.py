# oecd_education_visualizations.py
# OECD Education Data - Interactive Visualization Dashboard
# Uses Plotly for interactive charts with filters and legends

import dask.dataframe as dd
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np

def categorize_education(attainment_level):
    """Categorize education attainment levels into broader groups"""
    if pd.isna(attainment_level) or attainment_level == '_T':
        return 'Total'
    level = int(attainment_level[1:])  # Remove 'L' prefix and convert to int
    if level <= 2:
        return 'Below Upper Secondary'
    elif level <= 4:
        return 'Upper Secondary & Post-Secondary Non-Tertiary'
    else:
        return 'Tertiary Education'

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
    '_T': 'Total'
}
# Country / Reference Area mapping
COUNTRY_MAP = {
    'AUS': 'Australia', 'AUT': 'Austria', 'BEL': 'Belgium', 'CAN': 'Canada',
    'CHL': 'Chile', 'COL': 'Colombia', 'CRI': 'Costa Rica', 'CZE': 'Czechia',
    'DNK': 'Denmark', 'EST': 'Estonia', 'FIN': 'Finland', 'FRA': 'France',
    'DEU': 'Germany', 'GRC': 'Greece', 'HUN': 'Hungary', 'ISL': 'Iceland',
    'IRL': 'Ireland', 'ISR': 'Israel', 'ITA': 'Italy', 'JPN': 'Japan',
    'KOR': 'Korea', 'LVA': 'Latvia', 'LTU': 'Lithuania', 'LUX': 'Luxembourg',
    'MEX': 'Mexico', 'NLD': 'Netherlands', 'NZL': 'New Zealand', 'NOR': 'Norway',
    'POL': 'Poland', 'PRT': 'Portugal', 'SVK': 'Slovak Republic', 'SVN': 'Slovenia',
    'ESP': 'Spain', 'SWE': 'Sweden', 'CHE': 'Switzerland', 'TUR': 'Türkiye',
    'GBR': 'United Kingdom', 'USA': 'United States',
    'OECD': 'OECD Average'
}
# --- END OF NEW MAPPING ---
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
df_computed['Field of Education'] = df_computed['EDUCATION_FIELD'].map(FIELD_MAP).fillna('Other/Total')
df_computed['Age Group'] = df_computed['AGE'].map(AGE_MAP).fillna(df_computed['AGE'])
df_computed['Gender'] = df_computed['SEX'].map(SEX_MAP).fillna(df_computed['SEX'])
df_computed['Birth Place'] = df_computed['BIRTH_PLACE'].map(BIRTH_PLACE_MAP).fillna(df_computed['BIRTH_PLACE'])
df_computed['Country'] = df_computed['REF_AREA'].map(COUNTRY_MAP).fillna(df_computed['REF_AREA'])
print("Data preprocessing complete.")

# Remove the original function definition that was at the bottom of the file
# It has been moved to the top of the file

# HELPER FUNCTION: Categorize education levels

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
fig1.update_layout(template='plotly_dark', hovermode='x unified', font=dict(size=10))
fig1.write_html('viz_01_education_trends_over_time.html')
print("  ✓ Saved: viz_01_education_trends_over_time.html")

# ============================================================================
# VIZ 2: Education by Age Groups
# ============================================================================
print("\n[2/7] Viz 2: Education Attainment by Age Groups...")

# Filter data
age_trends = df_computed[
    (df_computed['EDUCATION_FIELD'] == '_T') &
    (df_computed['SEX'] == '_T') &
    (df_computed['MEASURE'] == 'POP')
]

# Aggregate data
age_agg = age_trends.groupby(['TIME_PERIOD', 'Age Group', 'Education Level', 'Country'])['OBS_VALUE'].sum().reset_index()

# Top 4 countries
top_countries = age_agg.groupby('Country')['OBS_VALUE'].sum().nlargest(4).index.tolist()
age_agg = age_agg[age_agg['Country'].isin(top_countries)]

fig2 = px.bar(
    age_agg,
    x='Age Group',
    y='OBS_VALUE',
    color='Education Level',
    facet_col='Country',
    facet_col_wrap=2,
    facet_row='TIME_PERIOD' if age_agg['TIME_PERIOD'].nunique() <= 3 else None,
    title='Education Attainment by Age Group (Top 4 Countries)',
    labels={
        'OBS_VALUE': 'Population',
        'Education Level': 'Education Level',
        'Country': 'Country',
        'Age Group': 'Age Group'
    },
    height=800,
    barmode='group'
)
fig2.update_layout(template='plotly_dark')
fig2.write_html('viz_02_education_by_age_groups.html')
print("  ✓ Saved: viz_02_education_by_age_groups.html")

# ============================================================================
# VIZ 3: Education Field Trends
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

# Get top countries
top_countries = field_agg.groupby('Country')['OBS_VALUE'].sum().nlargest(4).index.tolist()
field_agg = field_agg[field_agg['Country'].isin(top_countries)]

fig3 = px.bar(
    field_agg,
    x='TIME_PERIOD',
    y='OBS_VALUE',
    color='Field of Education',
    facet_col='Country',
    facet_col_wrap=2,
    title='Education Field Trends Over Time (Top 4 Countries)',
    labels={
        'TIME_PERIOD': 'Year',
        'OBS_VALUE': 'Population',
        'Field of Education': 'Field of Education',
        'Country': 'Country'
    },
    height=800,
    barmode='stack'
)
fig3.update_layout(template='plotly_dark')
fig3.write_html('viz_03_education_field_trends.html')
print("  ✓ Saved: viz_03_education_field_trends.html")

# ============================================================================
# VIZ 4: Country Growth Analysis
# ============================================================================
print("\n[4/7] Viz 4: Country Growth Analysis...")

# Filter data
growth_data = df_computed[
    (df_computed['EDUCATION_FIELD'] == '_T') &
    (df_computed['SEX'] == '_T') &
    (df_computed['AGE'] == 'Y25T64') &
    (df_computed['MEASURE'] == 'POP')
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

    growth_calc['GROWTH_RATE'] = ((growth_calc[last_year] - growth_calc[first_year]) / growth_calc[first_year] * 100).fillna(0)
    growth_calc = growth_calc[growth_calc['GROWTH_RATE'] != 0]

    fig4 = px.bar(
        growth_calc.nlargest(15, 'GROWTH_RATE'),
        x='Country',
        y='GROWTH_RATE',
        color='Education Level',
        title=f'Top Education Growth Rates by Country ({first_year}-{last_year})',
        labels={
            'GROWTH_RATE': 'Growth Rate (%)',
            'Country': 'Country',
            'Education Level': 'Education Level'
        },
        height=600,
        barmode='group'
    )
    fig4.update_layout(template='plotly_dark', hovermode='x unified')
    fig4.write_html('viz_04_country_growth_analysis.html')
    print("  ✓ Saved: viz_04_country_growth_analysis.html")

# ============================================================================
# VIZ 5: Education vs Labor Force Participation
# ============================================================================
# ============================================================================
# VIZ 5: Education vs Labor Force Participation
# ============================================================================
print("\n[5/7] Viz 5: Education vs Labor Force Participation...")

# --- NEW LOGIC ---

# 1. Get Employment Data
# We filter for 'EMP' (Employed) in the LABOUR_FORCE_STATUS column.
common_filters = (
    (df_computed['EDUCATION_FIELD'] == '_T') &
    (df_computed['SEX'] == '_T') &
    (df_computed['AGE'] == 'Y25T64')
)

emp_data = df_computed[
    common_filters &
    (df_computed['LABOUR_FORCE_STATUS'] == 'EMP')
]

# Select relevant columns and rename OBS_VALUE to 'Employed'
emp_agg = emp_data.groupby(
    ['Country', 'Education Level', 'TIME_PERIOD']
)['OBS_VALUE'].sum().reset_index().rename(columns={'OBS_VALUE': 'Employed'})


# 2. Get Population Data
# We filter for 'POP' (Population) in the MEASURE column.
pop_data = df_computed[
    common_filters &
    (df_computed['MEASURE'] == 'POP')
]

# Select relevant columns and rename OBS_VALUE to 'Population'
pop_agg = pop_data.groupby(
    ['Country', 'Education Level', 'TIME_PERIOD']
)['OBS_VALUE'].sum().reset_index().rename(columns={'OBS_VALUE': 'Population'})


# 3. Merge the two datasets
# We merge on the common keys. Now each row will have 'Employed' and 'Population'
lfpr_merged = pd.merge(
    emp_agg,
    pop_agg,
    on=['Country', 'Education Level', 'TIME_PERIOD'],
    how='left' # Use 'left' or 'inner' join
)

# 4. Calculate LFPR
# Now we can safely calculate the Labor Force Participation Rate
if 'Employed' in lfpr_merged.columns and 'Population' in lfpr_merged.columns:
    lfpr_merged['LFPR'] = (lfpr_merged['Employed'] / lfpr_merged['Population'] * 100).fillna(0)
    
    # Filter out invalid LFPR values
    lfpr_result = lfpr_merged[(lfpr_merged['LFPR'] > 0) & (lfpr_merged['LFPR'] <= 100)].copy()

    # Filter for a specific year to make the chart cleaner
    # (Plotting all years can be messy, let's pick the latest one)
    latest_year = lfpr_result['TIME_PERIOD'].max()
    if pd.notna(latest_year):
        print(f"  > Filtering LFPR data for latest available year: {latest_year}")
        lfpr_result = lfpr_result[lfpr_result['TIME_PERIOD'] == latest_year]

    # Filter for top 10 countries by population to keep the chart clean
    top_countries_lfpr = pop_agg.groupby('Country')['Population'].sum().nlargest(10).index
    lfpr_result = lfpr_result[lfpr_result['Country'].isin(top_countries_lfpr)]

    # --- THIS IS THE PLOT ---
    fig5 = px.bar(
        lfpr_result,
        x='Country',
        y='LFPR',
        color='Education Level',
        barmode='group',
        facet_col='Education Level',  # Facet by category to make it easy to compare
        title=f'Labor Force Participation Rate by Education Level (Ages 25-64, Year {latest_year})',
        labels={
            'Country': 'Country',
            'LFPR': 'Labor Force Participation Rate (%)',
            'Education Level': 'Education Level'
        },
        height=600
    )
    fig5.update_layout(template='plotly_dark')
    fig5.write_html('viz_05_education_vs_lfpr.html')
    print("  ✓ Saved: viz_05_education_vs_lfpr.html")
else:
    print("  ✗ Could not generate Viz 5: 'Employed' or 'Population' data missing after merge.")

# --- END OF REVISED BLOCK ---#
# ============================================================================
# VIZ 6: Education vs Migration
# ============================================================================
print("\n[6/7] Viz 6: Education Attainment vs Migration...")

# Filter data
migration_data = df_computed[
    (df_computed['EDUCATION_FIELD'] == '_T') &
    (df_computed['SEX'] == '_T') &
    (df_computed['AGE'] == 'Y25T64') &
    (df_computed['BIRTH_PLACE'] != '_T')
]

# Aggregate data
migration_agg = migration_data.groupby(['Country', 'Birth Place', 'Education Level'])['OBS_VALUE'].sum().reset_index()

# Get top countries
top_countries = migration_agg.groupby('Country')['OBS_VALUE'].sum().nlargest(5).index.tolist()
migration_agg = migration_agg[migration_agg['Country'].isin(top_countries)]

fig6 = px.bar(
    migration_agg,
    x='Country',
    y='OBS_VALUE',
    color='Education Level',
    facet_col='Birth Place',
    title='Education Attainment by Birth Place (Native vs Foreign-Born)',
    labels={
        'Country': 'Country',
        'OBS_VALUE': 'Population',
        'Education Level': 'Education Level',
        'Birth Place': 'Birth Place'
    },
    height=600,
    barmode='group'
)
fig6.update_layout(template='plotly_dark')
fig6.write_html('viz_06_education_vs_migration.html')
print("  ✓ Saved: viz_06_education_vs_migration.html")

# ============================================================================
# VIZ 7: Gender Distribution in Education
# ============================================================================
print("\n[7/7] Viz 7: Gender Distribution in Education...")

# Filter data
gender_data = df_computed[
    (df_computed['EDUCATION_FIELD'] == '_T') &
    (df_computed['SEX'] != '_T') &
    (df_computed['AGE'] == 'Y25T64') &
    (df_computed['MEASURE'] == 'POP')
]

# Aggregate data
gender_agg = gender_data.groupby(['TIME_PERIOD', 'Education Level', 'Gender', 'Country'])['OBS_VALUE'].sum().reset_index()

# Get top countries
top_countries = gender_agg.groupby('Country')['OBS_VALUE'].sum().nlargest(4).index.tolist()
gender_agg = gender_agg[gender_agg['Country'].isin(top_countries)]

fig7 = px.bar(
    gender_agg,
    x='TIME_PERIOD',
    y='OBS_VALUE',
    color='Gender',
    facet_col='Country',
    facet_row='Education Level',
    title='Gender Distribution in Education Levels (Top 4 Countries)',
    labels={
        'TIME_PERIOD': 'Year',
        'OBS_VALUE': 'Population',
        'Gender': 'Gender',
        'Country': 'Country',
        'Education Level': 'Education Level'
    },
    height=1000
)
fig7.update_layout(template='plotly_dark')
fig7.write_html('viz_07_gender_distribution.html')
print("  ✓ Saved: viz_07_gender_distribution.html")

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
