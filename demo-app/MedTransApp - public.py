# MEDICAL RECORDS ANALYSIS WITH WATSON ASSISTANT - DASHBOARD
# SEPTEMBER 2021

# Import data
import streamlit as st
import pandas as pd
import geopandas as gpd
import csv
from collections import Counter
import plotly.express as px
from debater_python_api.api.debater_api import DebaterApi
from austin_utils import init_logger
from bokeh.models import (ColorBar,GeoJSONDataSource, HoverTool,
                          LinearColorMapper)  ## WE NEED VERSION 2.2.2 OF BOKEH, HIGHER VERSIONS DO NOT RENDER PROPERLY ON STREAMLIT
from bokeh import palettes 
from bokeh.plotting import figure

data = '.'

# FUNCTIONS
@st.cache
def read_data():
    with open(data + '/mtsamples_descriptions_clean.csv') as csv_file:
        reader = csv.DictReader(csv_file)
        sentences = list(reader)
    
    return sentences

@st.cache
def read_original():
    _df = pd.read_csv(data + '/mtsamples_descriptions_clean.csv')
    return _df

@st.cache
def read_pollution_map():
    _df = gpd.read_file(data + '/pollution_levels.json')
    return _df 

@st.cache
def read_smoking_map():
    _df = gpd.read_file(data + '/smoking_rate.json')
    return _df 

@st.cache
def get_boroughs():
    return gpd.read_file("https://skgrange.github.io/www/data/london_boroughs.json")


@st.cache
def init_debater():
    init_logger()
    api_key = '--YOUR API KEY--'
    debater_api = DebaterApi(apikey=api_key)
    keypoints_client = debater_api.get_keypoints_client()
    domain = 'medical_demo'
    
    return keypoints_client, domain, debater_api

@st.cache
def get_top_quality_sentences(sentences, top_k, topic):  
    '''
    Ranks the sentences in a series based on how they relate to a given topic and returns the top K.
    '''  
    arg_quality_client = debater_api.get_argument_quality_client()
    sentences_topic = [{'sentence': sentence['text'], 'topic': topic} for sentence in sentences]
    arg_quality_scores = arg_quality_client.run(sentences_topic)
    sentences_and_scores = zip(sentences, arg_quality_scores)
    sentences_and_scores_sorted = sorted(sentences_and_scores, key=lambda x: x[1], reverse=True)
    sentences_sorted = [sentence for sentence, _ in sentences_and_scores_sorted]
    # print_top_and_bottom_k_sentences(sentences_sorted, 10)
    return sentences_sorted[:top_k]

@st.cache
def run_kpa(sentences, run_params, key_points_by_job_id=None):
    '''
    Runs key point analysis.
    '''

    sentences_texts = [sentence['text'] for sentence in sentences]
    sentences_ids = [sentence['id'] for sentence in sentences]

    keypoints_client.delete_domain_cannot_be_undone(domain) # Clear domain in case it existed already

    keypoints_client.upload_comments(domain=domain, 
                                     comments_ids=sentences_ids, 
                                     comments_texts=sentences_texts, 
                                     dont_split=True)

    keypoints_client.wait_till_all_comments_are_processed(domain=domain)

    future = keypoints_client.start_kp_analysis_job(domain=domain, 
                                                    comments_ids=sentences_ids, 
                                                    run_params=run_params,
                                                    key_points_by_job_id=key_points_by_job_id)

    kpa_result = future.get_result(high_verbosity=False, 
                                   polling_timout_secs=5)
    
    return kpa_result, future.get_job_id()
    

@st.cache(allow_output_mutation=True)
def result_to_df(result):  
    '''
    Converts the results of KPA to a pandas dataframe.
    '''
    matchings_rows = []
    for keypoint_matching in result['keypoint_matchings']:
        kp = keypoint_matching['keypoint']
        for match in keypoint_matching['matching']:
            match_row = [kp, match["sentence_text"], match["score"], match["comment_id"], match["sentence_id"],
                            match["sents_in_comment"], match["span_start"], match["span_end"], match["num_tokens"],
                            match["argument_quality"]]

            matchings_rows.append(match_row)

    cols = ["kp", "sentence_text", "match_score", 'comment_id', 'sentence_id', 'sents_in_comment', 'span_start',
            'span_end', 'num_tokens', 'argument_quality']
    match_df = pd.DataFrame(matchings_rows, columns=cols)
    
    return match_df

@st.cache
def merge_dfs(df_results, df_sentences):
    df_results['comment_id'] = df_results['comment_id'].astype(int)
    df_merge = df_results.merge(df_sentences[['id', 'id_description', 'medical_specialty_new', 'year', 'borough']], left_on='comment_id', right_on='id', validate = 'one_to_one')
    
    return df_merge

@st.cache
def get_sentence_to_mentions(sentences_texts):
    '''
    Extracts wikipedia terms from a sentence.
    '''
    term_wikifier_client = debater_api.get_term_wikifier_client()
    mentions_list = term_wikifier_client.run(sentences_texts)
    sentence_to_mentions = {}
    for sentence_text, mentions in zip(sentences_texts,    
                                       mentions_list):
        sentence_to_mentions[sentence_text] = set([mention['concept']['title'] for mention in mentions])
    
    return sentence_to_mentions

@st.cache
def get_wiki_terms(_df, _filter_var):
    '''
    Extracts the wikipedia terms and their frequency from series of sentences in a dataframe. It filters on a variable whose values 
    are the keys to the output dictionary
    '''
    terms = {}
    for val in set(_df[_filter_var].values):
        sentence_to_mentions = get_sentence_to_mentions(_df['sentence_text'][_df[_filter_var]==val].values) # Extract Wikipedia terms
        all_mentions = [mention for sentence in sentence_to_mentions for mention in sentence_to_mentions[sentence]] # Put terms in list
        term_count = dict(Counter(all_mentions)) # Count terms and put in dictionary
        if 'Patient' in term_count.keys():
            term_count.pop('Patient') 
        if 'History' in term_count.keys():
            term_count.pop('History')
        terms[val] = term_count
    
    return terms

def plot_indicators(_gdf, _sel_year, _sel_var, title = ''):
    '''
    Plots an indicator such as number of KP mentions or smoking rate in a map of the boroughs of London
    '''

    geosource = GeoJSONDataSource(geojson = _gdf[_gdf['Year'] == int(_sel_year)].to_json())
    palette = palettes.Magma11
    palette = palette[::-1] # reverse order of colors so higher values have darker colors# Instantiate LinearColorMapper that linearly maps numbers in a range, into a sequence of colors.
    color_mapper = LinearColorMapper(palette = palette, low = 0, high = _gdf[_sel_var].max()) 
    
    # Define custom tick labels for color bar.
    color_bar = ColorBar(
                        color_mapper = color_mapper, 
                        label_standoff = 8,
                        width = 500, height = 20,
                        border_line_color = None,
                        location = (0,0), 
                        orientation = 'horizontal'
                        )
    # Create figure object.
    p = figure(title = title + ' by London borough', 
            plot_height = 600, plot_width = 800, 
            toolbar_location = 'below',
            tools = 'pan, wheel_zoom, box_zoom, reset')
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None 
    
    # Add patch renderer to figure.
    states = p.patches('xs','ys', source = geosource,
                    fill_color = {'field': _sel_var,
                                    'transform' : color_mapper},
                    line_color = 'black', 
                    line_width = 0.25, 
                    fill_alpha = 1)
                    
    # Create hover tool
    p.add_tools(HoverTool(renderers = [states],
                        tooltips = [('Borough','@Borough'),
                                (str(_sel_var), '@' + str(_sel_var))]))
                                
    # Specify layout
    p.add_layout(color_bar, 'below')
    p.axis.visible = False

    return p

@st.cache
def group_kps(_df_merge, _boroughs, _all_years = [2010, 2013, 2016]):
    '''
    Creates a dataframe that contains an observation for each combination of year-borough-kp. It then merges it to the results dataframe.
    This way, we can plot maps and calculate correlations even if there was no occurrence of a KP in a particular year-borough. 
    '''
    # List all the years, boroughs and KPs that we need to combine
    all_kps = list(set(_df_merge[_df_merge['kp']!='none']['kp'].values))
    all_boroughs = list(set(_boroughs['name'].values))
    all_years = _all_years

    # Create a dataframe with every possible combination of borough-kp-year
    _all_years_kp_boroughs = pd.DataFrame(columns = ['Borough', 'KP', 'Year'])
    for borough in all_boroughs:
        for kp in all_kps:
            for year in all_years:
                _all_years_kp_boroughs = _all_years_kp_boroughs.append({
                                                                    'Borough': borough,
                                                                    'KP': kp,
                                                                    'Year': year
                                                                    }, ignore_index=True)
                                                                # Count the occurrence of each kp by borough and year
    _df_counts = _df_merge[_df_merge['kp']!='none'].groupby(by = ['year', 'kp', 'borough']).agg({'comment_id':'count'}).reset_index()
    _df_counts.rename(columns = {'comment_id':'count', 'borough':'Borough', 'year':'Year', 'kp':'KP'}, inplace = True)

    # Merge the occurrences dataset with the dataset containing all combinations of borough-kp-year. We have lots of combinations for which the count of kp is 0, but that's expected
    _all_merge = _all_years_kp_boroughs.merge(_df_counts, how = 'left', on = ['Year', 'KP', 'Borough'])
    _all_merge['count'].fillna(0, inplace = True)
    
    return _all_merge

@st.cache
def corr_smoking(year, _df) -> pd.DataFrame:
    
    """Create a dataframe with key points and their correlation to the smoking rate in a given year"""
    
    corr = _df[_df['Year'] == int(year)].drop('Year',axis=1).groupby('KP').corr()
    corr = corr[corr['smoking_rate']!=1].reset_index()
    corr.rename({'smoking_rate':'Correlation','KP':'Key Point'},axis=1,inplace=True)
    corr = corr[['Key Point','Correlation']]
    
    return corr
    

# UI SET UP - 1
st.sidebar.write(" ## **Navigation**")
tab = st.sidebar.radio(label = 'Go to:', options = ['Home - Key Point Classification and Distribution', 
                                                    'Analysis by Key Point',
                                                    'Public Health Indicators'])
st.sidebar.write(" ## **Model Parameters**")
number_kp = st.sidebar.slider(label = 'How many key points do you think there are in your sample?', min_value = 2, max_value= 20, value = 20, key = 'number_kp')
sel_year = st.sidebar.selectbox(label = 'Select the year that you want to visualise:', options = ['All', '2010', '2013', '2016'], key = 'viz_months')

# ANALYSIS
# Read data in
sentences = read_data()

# Initialise debater
keypoints_client, domain, debater_api = init_debater()

# Select top sentences
sentences_top_1000_aq = get_top_quality_sentences(sentences, 1000, "The patient is a 30-year-old who was admitted with symptoms including obstructions, failures and pain that started four days ago.")

# Get KPA results
kpa_result, job_id = run_kpa(sentences_top_1000_aq, {'n_top_kps': number_kp,
                                                     'mapping_threshold': 0.95})

# Export results to dataframe and merge back to original dataset
df_results = result_to_df(kpa_result)

df_sentences = read_original()

df_merge_all = merge_dfs(df_results, df_sentences)
share_classified = round(100*(1 - df_merge_all['kp'].value_counts(normalize = True)['none']), 2)

# Filter data for tabs based on year
if sel_year != 'All':
    df_merge = df_merge_all[df_merge_all['year'] == int(sel_year)].copy()
elif sel_year == 'All':
    df_merge = df_merge_all.copy()

# UI SET UP - 2
if tab == 'Home - Key Point Classification and Distribution':
    # Initial header
    st.title('Medical Transcriptions Analysis with IBM Project Debater')
    st.write("In this demo app, we show how [IBM's Project Debater](https://research.ibm.com/interactive/project-debater/) can assist **Population Health Management** (PHM) experts and practitioners uncover and exploit insights \
            from unstructured data in the form of medical transcriptions. \
            \n \n This demo uses the Medical Transcriptions dataset from Kaggle, which you can access [using this link](https://www.kaggle.com/tboyle10/medicaltranscriptions). \
            The dataset contains sample medical transcriptions for various medical specialties, and was downloaded from [mtsamples.com](https://mtsamples.com/)")

    st.write("This tool **is not meant to give medical advice or to replace a professional doctor's judgement**. On the contrary, its intended users are PHM managers or analysts who \
            want to understand macro trends and correlations in a given area or are interested in governance practices.")
    st.write(' ## **Key point classification and distribution**')  
    st.write("Let's start by breaking down the sample in a set of key points. Each sentence of the sample will be assigned to a bucket or key point. \
            The idea is that similar sentences should be clustered together.")



    st.write('We have trained the model using the top 1000 sentences from our sample, which we can classify in',  
            df_merge_all['kp'].nunique() - 1, 'key points, plus a "none" (meaningless) key point.', 
            share_classified, '% were assigned a key point. The key points are:')
    st.write(list(set(df_merge_all['kp'].values)))


    st.write('Select a year in the sidebar and hover over the pie chart to see how many sentences are classified in each key point:')

    # Pie chart
    fig1 = px.pie(df_merge['kp'].value_counts(), 
                values = df_merge['kp'].value_counts(), 
                names = df_merge['kp'].value_counts().index,
                color_discrete_sequence=px.colors.qualitative.Light24
                # title = 'Key points distribution'
                )

    fig1.layout.update(showlegend = False, template = 'ggplot2', width = 600, height = 500)

    st.plotly_chart(fig1)

elif tab == 'Analysis by Key Point':

    # Explore sentences in each key point
    st.write(' ## **Analysis by key point** ')
    st.write("Let's explore what is inside each key point")
    viz_kp = st.selectbox(label = 'Select the key point that you want to explore:', options = list(set(df_merge['kp'].values)), key = 'viz_kp_sentences')

    st.write(' ### **Sentences by key point**')
    st.write('We can see the sentences that have been classified within each key point:')
    st.write('The key point', viz_kp, 'contains', len(df_merge[df_merge['kp'] == viz_kp]), 'sentences in', sel_year + ":")
    st.write(list(df_merge['sentence_text'][df_merge['kp'] == viz_kp].values))


    # Count Wikipedia terms in each key point and Visualise term wikifier results
    st.write(' ### **Wikipedia terms by key point**')
    st.write('We can also associate Wikipedia articles to terms that appear in the sentences within each keypoint. \
            This allows us to interpret the key point by pointing us to actual concepts that constitute the core of what the topic is about.')
    
    terms_kp = get_wiki_terms(df_merge, 'kp')
    _df_viz = pd.DataFrame(list(terms_kp[viz_kp].items()),columns = ['Term','Count']).sort_values(by = 'Count', ascending=True)

    fig2 = px.bar(x = _df_viz['Count'].tail(10),
            y = _df_viz['Term'].tail(10),
            color=_df_viz['Term'].tail(10),
            color_discrete_sequence=px.colors.sequential.GnBu_r,
            orientation = 'h',
            title = 'Key Point:' + viz_kp
            )

    fig2.layout.update(showlegend = False, template = 'ggplot2', width = 700, height = 500,
                yaxis = dict(title_text = 'Top 10 Wikipedia Terms',showline = True, showticklabels = True, color = 'black'),
                xaxis = dict(title_text = 'Number of Mentions')
                )
    st.plotly_chart(fig2)

elif tab == 'Public Health Indicators':

    st.write(" ## **Public Health Indicators**")
    st.write("In this section we can cross-examine the results of our key point analysis with other public health indicators.")
    
    if sel_year == 'All':
        st.error("Please select a valid year on the sidebar menu to continue")
    else:
        # Health indicators
        sel_var = st.selectbox(label = 'Select the health indicator that you want to explore:', options = ['smoking_rate'], key = 'viz_var')
        st.write(" ### **Correlation of Key Point incidence and Health Indicators**")       
        st.write("The table below shows the correlation between how many cases of each key point there were in the different \
                London boroughs and the health indicator", sel_var, "in year", sel_year)
        
        boroughs = get_boroughs()
        df_group_kps = group_kps(df_merge_all, _boroughs = boroughs)
        df_group_kps_geo = df_group_kps.merge(boroughs[['name', 'geometry']], how = 'left', left_on = 'Borough', right_on = 'name')
        df_group_kps_geo = gpd.GeoDataFrame(df_group_kps_geo)

        smokers = read_smoking_map()

        smoke_kp = pd.merge(smokers[smokers['smoking_rate'].isna()==False], 
                    df_group_kps, 
                    on=['Year','Borough'])
        corr = corr_smoking(year = sel_year, _df = smoke_kp).sort_values(by = 'Correlation', ascending = False)
        st.dataframe(corr)

        st.write(" ### **Geographical analysis**")

        # Keypoints map
        st.write(" #### **Distribution of key points across London boroughs**")
        _kp_viz = st.selectbox(label = 'Select which keypoint you want to visualise on a map:', options = list(set(df_group_kps_geo['KP'].values)), key = '_kp_viz')
        p1 = plot_indicators(df_group_kps_geo[df_group_kps_geo['KP'] == _kp_viz], int(sel_year), 'count', 'Distribution of key point '+ _kp_viz)
        st.bokeh_chart(p1)

        # Smokers map
        st.write(" #### **Distribution of other Health Indicators across London boroughs**")
        p2 = plot_indicators(smokers, sel_year, sel_var, 'Distribution of smoking rates')
        st.bokeh_chart(p2)


        # Term wikifier by borough Visualise term wikifier results
        st.write(' ### **Wikipedia terms by borough**')
        st.write('The chart below shows the most common Wikipedia terms across the London boroughs.')
        viz_borough = st.selectbox(label = 'Select a London borough:', options = list(set(df_merge['borough'].values)), key = 'viz_kp_boroughs')
        terms_borough = get_wiki_terms(df_merge, 'borough')
        _df_viz_borough = pd.DataFrame(list(terms_borough[viz_borough].items()),columns = ['Term','Count']).sort_values(by = 'Count', ascending=True)

        fig2 = px.bar(x = _df_viz_borough['Count'].tail(10),
                y = _df_viz_borough['Term'].tail(10),
                color=_df_viz_borough['Term'].tail(10),
                color_discrete_sequence=px.colors.sequential.GnBu_r,
                orientation = 'h',
                title = 'Borough: ' + viz_borough
                )

        fig2.layout.update(showlegend = False, template = 'ggplot2', width = 700, height = 500,
                    yaxis = dict(title_text = 'Top 10 Wikipedia Terms',showline = True, showticklabels = True, color = 'black'),
                    xaxis = dict(title_text = 'Number of Mentions')
                    )
        st.plotly_chart(fig2)
