# oecd_education_visualizations.py
# OECD Education Data - Interactive Visualization Dashboard
# Uses Plotly for interactive charts with filters and legends

import dask.dataframe as dd
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np

print("=" * 80)
print("OECD EDUCATION DATA - INTERACTIVE VISUALIZATIONS")
print("=" * 80)

# Load the Parquet dataset
print("\nLoading Parquet dataset...")
ddf = dd.read_parquet('oecd_data.parquet')

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

trends = ddf[
    (ddf['EDUCATION_FIELD'] == '_T') &
    (ddf['SEX'] == '_T') &
    (ddf['AGE'] == 'Y25T64') &
    (ddf['MEASURE'] == 'POP')
].compute()

trends['EDUCATION_CATEGORY'] = trends['ATTAINMENT_LEV'].apply(categorize_education)
trends_agg = trends.groupby(['TIME_PERIOD', 'EDUCATION_CATEGORY', 'REF_AREA'])['OBS_VALUE'].sum().reset_index()

# Get top 8 countries
top_countries = trends_agg.groupby('REF_AREA')['OBS_VALUE'].sum().nlargest(8).index.tolist()
trends_agg = trends_agg[trends_agg['REF_AREA'].isin(top_countries)]

# Calculate percentages
trends_agg['PERCENTAGE'] = trends_agg.groupby(['TIME_PERIOD', 'REF_AREA'])['OBS_VALUE'].transform(
    lambda x: (x / x.sum() * 100) if x.sum() > 0 else 0
)

fig1 = px.line(
    trends_agg,
    x='TIME_PERIOD',
    y='PERCENTAGE',
    color='EDUCATION_CATEGORY',
    facet_col='REF_AREA',
    facet_col_wrap=4,
    title='Education Attainment Trends Over Time (Primary, Secondary, Tertiary)',
    labels={'TIME_PERIOD': 'Year', 'PERCENTAGE': 'Percentage (%)', 'EDUCATION_CATEGORY': 'Education Level'},
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

age_trends = ddf[
    (ddf['EDUCATION_FIELD'] == '_T') &
    (ddf['SEX'] == '_T') &
    (ddf['MEASURE'] == 'POP')
].compute()

age_trends['EDUCATION_CATEGORY'] = age_trends['ATTAINMENT_LEV'].apply(categorize_education)
age_agg = age_trends.groupby(['TIME_PERIOD', 'AGE', 'EDUCATION_CATEGORY', 'REF_AREA'])['OBS_VALUE'].sum().reset_index()

# Top 4 countries
top_countries = age_agg.groupby('REF_AREA')['OBS_VALUE'].sum().nlargest(4).index.tolist()
age_agg = age_agg[age_agg['REF_AREA'].isin(top_countries)]

fig2 = px.bar(
    age_agg,
    x='AGE',
    y='OBS_VALUE',
    color='EDUCATION_CATEGORY',
    facet_col='REF_AREA',
    facet_col_wrap=2,
    facet_row='TIME_PERIOD' if age_agg['TIME_PERIOD'].nunique() <= 3 else None,
    title='Education Attainment by Age Group (Top 4 Countries)',
    labels={'OBS_VALUE': 'Population', 'EDUCATION_CATEGORY': 'Education Level'},
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

field_trends = ddf[
    (ddf['EDUCATION_FIELD'] != '_T') &
    (ddf['SEX'] == '_T') &
    (ddf['AGE'] == 'Y25T64') &
    (ddf['MEASURE'] == 'POP')
].compute()

field_mapping = {
    'FIELD001': 'Education', 'FIELD002': 'Arts/Humanities', 'FIELD003': 'Social Sciences',
    'FIELD004': 'Business/Law', 'FIELD005': 'Natural Sciences', 'FIELD006': 'ICT',
    'FIELD007': 'Engineering', 'FIELD008': 'Agriculture', 'FIELD009': 'Health/Welfare', 'FIELD010': 'Services'
}
field_trends['FIELD_NAME'] = field_trends['EDUCATION_FIELD'].map(field_mapping).fillna(field_trends['EDUCATION_FIELD'])

field_agg = field_trends.groupby(['TIME_PERIOD', 'FIELD_NAME', 'REF_AREA'])['OBS_VALUE'].sum().reset_index()
top_countries = field_agg.groupby('REF_AREA')['OBS_VALUE'].sum().nlargest(4).index.tolist()
field_agg = field_agg[field_agg['REF_AREA'].isin(top_countries)]

fig3 = px.bar(
    field_agg,
    x='TIME_PERIOD',
    y='OBS_VALUE',
    color='FIELD_NAME',
    facet_col='REF_AREA',
    facet_col_wrap=2,
    title='Education Field Trends Over Time (Top 4 Countries)',
    labels={'TIME_PERIOD': 'Year', 'OBS_VALUE': 'Population', 'FIELD_NAME': 'Field of Education'},
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

growth_data = ddf[
    (ddf['EDUCATION_FIELD'] == '_T') &
    (ddf['SEX'] == '_T') &
    (ddf['AGE'] == 'Y25T64') &
    (ddf['MEASURE'] == 'POP')
].compute()

growth_data['EDUCATION_CATEGORY'] = growth_data['ATTAINMENT_LEV'].apply(categorize_education)
growth_agg = growth_data.groupby(['TIME_PERIOD', 'REF_AREA', 'EDUCATION_CATEGORY'])['OBS_VALUE'].sum().reset_index()

years = sorted(growth_agg['TIME_PERIOD'].unique())
if len(years) >= 2:
    first_year = years[0]
    last_year = years[-1]

    growth_calc = growth_agg[
        (growth_agg['TIME_PERIOD'] == first_year) | (growth_agg['TIME_PERIOD'] == last_year)
    ].pivot_table(values='OBS_VALUE', index=['REF_AREA', 'EDUCATION_CATEGORY'], columns='TIME_PERIOD').reset_index()

    growth_calc['GROWTH_RATE'] = ((growth_calc[last_year] - growth_calc[first_year]) / growth_calc[first_year] * 100).fillna(0)
    growth_calc = growth_calc[growth_calc['GROWTH_RATE'] != 0]

    fig4 = px.bar(
        growth_calc.nlargest(15, 'GROWTH_RATE'),
        x='REF_AREA',
        y='GROWTH_RATE',
        color='EDUCATION_CATEGORY',
        title=f'Top Education Growth Rates by Country ({first_year}-{last_year})',
        labels={'GROWTH_RATE': 'Growth Rate (%)', 'REF_AREA': 'Country'},
        height=600,
        barmode='group'
    )
    fig4.update_layout(template='plotly_dark', hovermode='x unified')
    fig4.write_html('viz_04_country_growth_analysis.html')
    print("  ✓ Saved: viz_04_country_growth_analysis.html")

# ============================================================================
# VIZ 5: Education vs Labor Force Participation
# ============================================================================
print("\n[5/7] Viz 5: Education vs Labor Force Participation...")

lfpr_data = ddf[
    (ddf['EDUCATION_FIELD'] == '_T') &
    (ddf['SEX'] == '_T') &
    (ddf['AGE'] == 'Y25T64')
].compute()

lfpr_data['EDUCATION_CATEGORY'] = lfpr_data['ATTAINMENT_LEV'].apply(categorize_education)

# Calculate LFPR by education category
lfpr_agg = lfpr_data.groupby(['REF_AREA', 'LABOUR_FORCE_STATUS', 'EDUCATION_CATEGORY'])['OBS_VALUE'].sum().reset_index()
lfpr_pivot = lfpr_agg.pivot_table(values='OBS_VALUE', index=['REF_AREA', 'EDUCATION_CATEGORY'], columns='LABOUR_FORCE_STATUS')

if 'EMP' in lfpr_pivot.columns and 'POP' in lfpr_pivot.columns:
    lfpr_pivot['LFPR'] = (lfpr_pivot['EMP'] / lfpr_pivot['POP'] * 100).fillna(0)
    lfpr_result = lfpr_pivot.reset_index()
    lfpr_result = lfpr_result[lfpr_result['LFPR'] > 0]

    fig5 = px.box(
        lfpr_data[lfpr_data['LABOUR_FORCE_STATUS'] == 'EMP'].groupby(['EDUCATION_CATEGORY', 'REF_AREA'])['OBS_VALUE'].sum().reset_index(),
        x='EDUCATION_CATEGORY',
        y='OBS_VALUE',
        color='EDUCATION_CATEGORY',
        title='Employment by Education Level',
        labels={'EDUCATION_CATEGORY': 'Education Level', 'OBS_VALUE': 'Employed Population'},
        height=600
    )
    fig5.update_layout(template='plotly_dark')
    fig5.write_html('viz_05_education_vs_lfpr.html')
    print("  ✓ Saved: viz_05_education_vs_lfpr.html")

# ============================================================================
# VIZ 6: Education vs Migration
# ============================================================================
print("\n[6/7] Viz 6: Education Attainment vs Migration...")

migration_data = ddf[
    (ddf['EDUCATION_FIELD'] == '_T') &
    (ddf['SEX'] == '_T') &
    (ddf['AGE'] == 'Y25T64') &
    (ddf['BIRTH_PLACE'] != '_T')
].compute()

migration_data['EDUCATION_CATEGORY'] = migration_data['ATTAINMENT_LEV'].apply(categorize_education)
migration_agg = migration_data.groupby(['REF_AREA', 'BIRTH_PLACE', 'EDUCATION_CATEGORY'])['OBS_VALUE'].sum().reset_index()

top_countries = migration_agg.groupby('REF_AREA')['OBS_VALUE'].sum().nlargest(5).index.tolist()
migration_agg = migration_agg[migration_agg['REF_AREA'].isin(top_countries)]

fig6 = px.bar(
    migration_agg,
    x='REF_AREA',
    y='OBS_VALUE',
    color='EDUCATION_CATEGORY',
    facet_col='BIRTH_PLACE',
    title='Education Attainment by Birth Place (Native vs Foreign-Born)',
    labels={'REF_AREA': 'Country', 'OBS_VALUE': 'Population', 'EDUCATION_CATEGORY': 'Education Level'},
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

gender_data = ddf[
    (ddf['EDUCATION_FIELD'] == '_T') &
    (ddf['SEX'] != '_T') &
    (ddf['AGE'] == 'Y25T64') &
    (ddf['MEASURE'] == 'POP')
].compute()

gender_data['EDUCATION_CATEGORY'] = gender_data['ATTAINMENT_LEV'].apply(categorize_education)
gender_mapping = {'F': 'Female', 'M': 'Male'}
gender_data['SEX_NAME'] = gender_data['SEX'].map(gender_mapping)

gender_agg = gender_data.groupby(['TIME_PERIOD', 'EDUCATION_CATEGORY', 'SEX_NAME', 'REF_AREA'])['OBS_VALUE'].sum().reset_index()

top_countries = gender_agg.groupby('REF_AREA')['OBS_VALUE'].sum().nlargest(4).index.tolist()
gender_agg = gender_agg[gender_agg['REF_AREA'].isin(top_countries)]

fig7 = px.bar(
    gender_agg,
    x='TIME_PERIOD',
    y='OBS_VALUE',
    color='SEX_NAME',
    facet_col='REF_AREA',
    facet_row='EDUCATION_CATEGORY',
    title='Gender Distribution in Education Levels (Top 4 Countries)',
    labels={'TIME_PERIOD': 'Year', 'OBS_VALUE': 'Population', 'SEX_NAME': 'Gender'},
    height=1000
)
fig7.update_layout(template='plotly_dark')
fig7.write_html('viz_07_gender_distribution.html')
print("  ✓ Saved: viz_07_gender_distribution.html")

print("\n" + "=" * 80)
print("VISUALIZATION GENERATION COMPLETE")
print("=" * 80)
print("Open the HTML files in your web browser to explore!")
