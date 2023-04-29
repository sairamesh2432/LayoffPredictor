import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# add title
st.title('Tech Layoff Predictor')

# load headcount dicts
with open('layoff_headcounts.pickle', 'rb') as f:
    layoff_headcounts = pickle.load(f)
with open("non_layoff_headcounts.pickle", "rb") as f:
    non_layoff_headcounts = pickle.load(f)

# merge dicts
headcounts = {**layoff_headcounts, **non_layoff_headcounts}

# Load pre-existing machine learning model
with open('rf.pkl', 'rb') as f:
    model = pickle.load(f)

# Load tech company data
company_data = pd.read_csv('companies.csv')

# Create dropdown list of top tech companies
top_companies = pd.read_csv("Top 50 US Tech Companies 2022 - 2023.csv")["Stock Name"]

company_list = company_data[company_data["Ticker"].isin(top_companies)]["Company Name"].unique()
# company_list = company_data['Company Name'].unique()


# Allow users to search for a company name
search_query = st.text_input('Search for a tech company:')
if search_query:
    search_results = company_data[company_data['Company Name'].str.contains(search_query, case=False)]
    if len(search_results) > 0:
        selected_company = st.selectbox('Select a tech company:', search_results['Company Name'].unique())
    else:
        st.warning('No results found for search query: {}. Please enter another company.'.format(search_query))
        selected_company = None
else:
    selected_company = st.selectbox('Select a tech company:', company_list)

# Get layoff probability for selected company
with open("selected_columns.pkl", "rb") as f:
    selected_columns = pickle.load(f)

if selected_company:
    data_df = company_data[selected_columns + ['Company Name']]
    input_data = data_df[data_df['Company Name']==selected_company].drop(['Company Name'], axis=1)
    layoff_probability = model.predict_proba(input_data.values)[0][1]
    is_layoff = company_data[company_data["Company Name"] == selected_company]["Layoffs"].values[0]
    if is_layoff != 1:
        st.write(f"{selected_company} has not had any layoffs in the past 3 years.")
        st.write('The probability of future layoffs at {} is {:.2f}%.'.format(selected_company, layoff_probability*100))

        # dot_data = export_graphviz(model.estimators_[0], out_file=None, feature_names=selected_columns, 
        #                 class_names=["No Layoffs", "Layoffs"], rounded=True, filled=True)


        # cols_new = st.columns(1, gap='small')
        # cols_new[0].graphviz_chart(dot_data)
    else:
        st.write(f"{selected_company} has had layoffs in the past 3 years. Proceed with caution.")
    # plot headcount growth
    fig, ax = plt.subplots()
    company_ticker = company_data[company_data["Company Name"] == selected_company]["Ticker"].values[0]
    ax.plot(headcounts[company_ticker]['Year'], headcounts[company_ticker]['Number of Employees'])
    ax.set_title(f"{selected_company} Headcount Growth")
    ax.set_xlabel("Year")
    ax.set_ylabel("Headcount")
    st.pyplot(fig)
# Display layoff probability to user

# st.write('The actual layoff status is {}.'.format(is_layoff))
