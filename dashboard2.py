import dash
import dash_html_components as html
import dash_core_components as dcc
from dash_html_components.Title import Title
import pandas as pd 
import plotly.express as px
import numpy as np
from dash.dependencies import Input, Output
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import itertools
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


#--------------------------------Setting up dataframes------------------------------------------------------#

df_all = pd.read_csv("owid-covid-data.csv")
malaysia = df_all[df_all["location"] == "Malaysia"]
malaysia = malaysia.loc[:,["location", "date", "total_cases", "new_cases", "total_deaths", "new_deaths"]]
death_data = pd.read_csv("death_data_translated.csv", index_col = [0])


my_state = pd.read_csv("daily_data.csv", index_col = [0])
google_my_data = pd.read_csv("2021_MY_Region_Mobility_Report.csv")
state_d = {"Wp\xa0Putrajaya" : "WP Putrajaya", "Wp\xa0Kuala Lumpur" : "Wp Kuala Lumpur", "W.P. Putrajaya" : "WP Putrajaya",\
          "W.P. Kuala Lumpur" : "Wp Kuala Lumpur", "Wp\xa0Labuan" : "WP Labuan", "W.P. Labuan"  :"WP Labuan"}

for old, new in state_d.items():
    my_state["States"] = my_state["States"].str.replace(old,new, regex = True)
    
my_state["States"] = my_state["States"].str.title()
clean_google = google_my_data[google_my_data["sub_region_1"].notnull()]
clean_google = clean_google.iloc[:, [2,5,8,9,10,11,12,13,14]]
clean_google["sub_region_1"] = clean_google["sub_region_1"].str.replace("Federal Territory of Kuala Lumpur", "Wp Kuala Lumpur")
clean_google["sub_region_1"] = clean_google["sub_region_1"].str.replace("Malacca", "Melaka")
clean_google["sub_region_1"] = clean_google["sub_region_1"].str.replace("Labuan Federal Territory", "WP Labuan")
clean_google["sub_region_1"] = clean_google["sub_region_1"].str.replace("Penang", "Pulau Pinang")
clean_google["sub_region_1"] = clean_google["sub_region_1"].str.replace("Putrajaya", "WP Putrajaya")

clean_google_dailycases = pd.merge(clean_google, malaysia, how = "inner", on = "date")
clean_google_dailycases = clean_google_dailycases[["sub_region_1", "date", "total_cases", "total_deaths", "retail_and_recreation_percent_change_from_baseline", \
                        "grocery_and_pharmacy_percent_change_from_baseline", "parks_percent_change_from_baseline", "transit_stations_percent_change_from_baseline", \
                        "workplaces_percent_change_from_baseline", "residential_percent_change_from_baseline"]]
clean_google_dailycases = clean_google_dailycases.rename(columns={'sub_region_1': 'States'})
clean_google_dailycases["States"] = clean_google_dailycases["States"].str.title()

df3 = pd.merge(my_state, clean_google_dailycases, how = "inner", on = ["States", "date"]) #Combined google data


df2 = death_data.copy()
df2 = df2.drop_duplicates()
# df2["state"].values[3910] = "Sabah"
# df2["gender"].values[3910] = "male"
# #df2["age"].values[3910] = "0.3"
df2["age"] = df2["age"].str.replace(r".*bulan","0.3", regex = True)
df2["age"] = df2.age.str.strip()
df2["age"] = pd.to_numeric(df2.age)
df2.loc[(df2.age >= 65),  'age_range'] = '65+'
df2.loc[(df2["age"] >= 25) & (df2["age"] <= 40), 'age_range'] = "25-40"
df2.loc[(df2["age"] >= 41) & (df2["age"] <= 64), 'age_range'] = "41-64"
df2.loc[(df2.age < 25),  'age_range'] = '<25'

ls_cause = list(death_data['cause'])
ls_cause = [x for x in ls_cause if type(x) == str]

ls_cause = [x.replace(", ",",") for x in ls_cause]
ls_cause = [x.replace(",,",",") for x in ls_cause]
ls_cause = [x.split(",") for x in ls_cause]
ls_cause = list(itertools.chain(*ls_cause))
ls_cause = [l.strip() for l in ls_cause]
unique_cause = list(set(ls_cause))
cause_count = { cause : death_data["cause"].str.contains(cause).sum() for cause in unique_cause }
cause_count_df = pd.DataFrame.from_dict(cause_count,orient='index', columns = ["count"])
cause_count_df = cause_count_df.sort_values(by = "count")
t20 = cause_count_df.tail(20)
t20 = t20.drop(['a','g', 'm', 's', 'l', 's', 'e', "", "r", "(pid)", "p"])
t20 = t20.reset_index()

t10ls = list(t20["index"])
for sick in t10ls:
    df2[f"{sick}"] = df2["cause"].str.contains(f"{sick}")
gender_age = df2.groupby(["gender", "age_range"]).size().reset_index(name='count')
gender_state = df2.groupby(["state", "gender"]).size().reset_index(name = 'count')
gender_state_age = df2.groupby(["state", "gender", "age_range"]).size().reset_index(name = 'count')

disease_grouped = df2.groupby(["gender", "age_range"])[["high blood pressure",'cancer', 'diabetes', 'kidney', 'na', "heart", "obesity","stroke","dyslipidemia"]].sum()

disease_age = df2.groupby(["state","gender", "age"])[["lung", "high blood pressure",'cancer', 'diabetes', 'kidney', 'na', "heart", "obesity","stroke","dyslipidemia"]].sum()
# disease_age["count"] = disease_age.sum(axis = 1)
disease_age = disease_age.astype(int)
w = disease_age.reset_index()

pca_df = disease_age.reset_index()
X = pca_df.iloc[:,3:]  # all rows, all the features and no labels
y = pca_df.iloc[:, 2]  # all rows, label only

scaler = StandardScaler()
X=scaler.fit_transform(X)

pca = PCA()
pca.fit_transform(X)
pca.get_covariance()
explained_variance=pca.explained_variance_ratio_
pca=PCA(n_components=5)
X_new=pca.fit_transform(X)
explained_variance=pca.explained_variance_ratio_
loadings = pca.components_
num_pc = pca.n_features_
pc_list = ["PC"+str(i) for i in list(range(1, num_pc+1))]
loadings_df = pd.DataFrame.from_dict(dict(zip(pc_list, loadings)))
loadings_df['variable'] = pca_df.columns.values[3:]
loadings_df = loadings_df.set_index('variable')
pcaa = pca_df.iloc[:,3:]
x = StandardScaler().fit_transform(pcaa)
x = pd.DataFrame(x, columns=pcaa.columns)
pcamodel = PCA(n_components=5)
pca = pcamodel.fit_transform(x)



#----------------------------Dash layout ------------------------------------------------#

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.title= "Malaysia's COVID-19 Situation Dashboard"


tab0_layout = [
    html.Div([
        dcc.RadioItems(
            
            id='items',
            options=[{'label': "gender", 'value': "gender"},
                     {"label": "age", "value":"age"},
                     {"label": "disease by age", "value":"disease"}],
            value="gender", labelStyle = {"display" : "block"}
        ),
        html.H2(id='header'), html.Div(id='ddd-output-container'),
        dcc.Dropdown(
        id='disease-drop',
        options=[{'label' : x , 'value' : x} for x in t10ls],
        value='high blood pressure',
        placeholder = "Choose two disease...", multi = False
    )
    ])
]

tab5_layout = [
    html.Div([
        html.H4("Common pre-existing health condition and breakdown of death cases by gender (national aggregate)")
    ])
]

tab6_layout = [
    html.Div([
        html.H2("Big picture "),
        dcc.RadioItems(
            id = "pca-type",
            options = [ { "label" : "loading", "value" : "loading"},
            { "label" : "bar-chart", 'value' : "bar"}],
            value = "loading"
        )
    ])
]

mastertab_layout = [
    html.Div([
        html.H2("Malaysia COVID cases by states"),
        html.P("Visualizing Malaysia's covid-19 daily cases by states to understand trend and conditions in each state", style = {'fontsize' : "16"}),
        dcc.RadioItems(
            id = "graph-type",
            options = [ { "label" : "time-series", "value" : "line"},
            { "label" : "bar-chart", 'value' : "bar"}],
            value = "line", labelStyle={'display': 'block'}
        )
    ])
]

tab2_layout = [
    html.Div([
        html.H3("Google's mobility data \n for each state"),
        html.P("Using Google's mobility data, we could see how mobility in each sector has changed over time", style = {"fontsize" : "16"}),
        dcc.Dropdown(
            id = "google-drop",
            options = [{'label' : x , 'value' : x} for x in df3["States"].unique()],
            value = "Selangor",
            placeholder = "Choose a state for google drop..."
        )
    ])
]

tab3_layout = [
    html.Div([
        html.H3("State level data from Google and OWID"),
        html.P("Visualizing daily cases and google's data over time side-by-side, we could see whether trends are similar across both variables", style = {"fontsize": "16"}),
        dcc.Dropdown(
            id = "stacked_drop",
            options = [{'label' : x , 'value' : x} for x in df3["States"].unique()],
            value = "Selangor",
            placeholder = "Choose a state for stacked plot..."
        )
    ])
]

tab4_layout = [
    html.Div([
        html.H3("Regression of daily cases and residential movements"),
        html.P("Using linear regression, we explore if there is any correlation between daily cases and residential mobility in each states, and are the trends \
        in each states consistent"),
        dcc.Dropdown(
            id = "regression_drop",
            options = [{'label' : x , 'value' : x} for x in df3["States"].unique()],
            value = "Selangor",
            placeholder = "Choose a state for regression plot..."
        )
    ])
]

app.layout = html.Div([
    dcc.Tabs(id='tabs-example', value='tab-m', children=[
        dcc.Tab(label = "State-resolved covid-cases data", value = "tab-m", children = mastertab_layout),
        dcc.Tab(label='Mobility data from Google', value='tab-2', children = tab2_layout),
        dcc.Tab(label="Daily cases and residential mobility", value='tab-3' , children = tab3_layout),
        dcc.Tab(label='State level daily cases vs residential movement changes', value='tab-1', children = tab4_layout),
        dcc.Tab(label = "Overview of death cases", value = "tab-5", children = tab5_layout),
        dcc.Tab(label= "Pre-existing health conditions by gender and age (national aggregate)", value = 'tab-0', children = tab0_layout),
        dcc.Tab(label = "PCA", value = "tab-6", children = tab6_layout),
        

        
    ]),
    html.Div(id='tabs-example-content'),
    # dcc.Dropdown(
    #     id='dropdown',
    #     options=[{'label' : x , 'value' : x} for x in df3["States"].unique()],
    #     value='Selangor',
    #     placeholder = "Choose a state...", clearable = False
    # ),
    # html.Div(id='dd-output-container'),
    dcc.Graph(id='graph-court')
])

@app.callback(Output('graph-court', 'figure'),
              [Input('tabs-example', 'value'), 
            #   Input('dropdown', 'value'),
              Input("items", "value"),
              Input("graph-type", "value"),
              Input("disease-drop","value"),
              Input("pca-type","value")],
              Input("google-drop", "value"),
              Input("stacked_drop", "value"),
              Input("regression_drop", "value"))


#--------------------------------------------------Updating plots in tabs ----------------------------------------------#
# def render_content(tab, state, var, chart,disease,pca_type, google_drop, residential_drop, regress_drop):
def render_content(tab, var, chart,disease,pca_type, google_drop, residential_drop, regress_drop):
    if tab == 'tab-1':
        #print(regress_drop) 
        #dff = df3[df3["States"] == regress_drop]
        #fig = px.scatter(df3[df3["States"] == regress_drop], x="residential_percent_change_from_baseline", y="Daily cases", height=600, trendline = "ols")
        fig = px.scatter(df3[df3["States"] == regress_drop], x="residential_percent_change_from_baseline", y="Daily cases")
        
        fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)'}, yaxis={'title':'daily cases'},
                      title={'text':f'covid cases against residential movement changes in {regress_drop}',
                      'font':{'size':28},'x':0.5,'xanchor':'center'}, xaxis = {"title" : "residential change from baseline"})
        
        return fig
    elif tab == 'tab-2':
        #print(google_drop)
        fig = px.scatter(df3[df3["States"] == google_drop], x = "date", y=["retail_and_recreation_percent_change_from_baseline", \
                        "grocery_and_pharmacy_percent_change_from_baseline", "parks_percent_change_from_baseline", "transit_stations_percent_change_from_baseline", \
                        "workplaces_percent_change_from_baseline", "residential_percent_change_from_baseline"])

        fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)'})
        fig.update_yaxes(nticks=10)
        #fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror = True)
        #fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror = True)


        return fig
    elif tab == 'tab-3':

        fig = make_subplots(rows=1, cols=2)

        fig.append_trace(go.Scatter(
            x=df3[df3["States"] == residential_drop]["date"],
            y=df3[df3["States"] == residential_drop]["Daily cases"], name="Daily cases"
        ), row=1, col=1)

        fig.append_trace(go.Scatter(
            x=df3[df3["States"] == residential_drop]["date"],
            y=df3[df3["States"] == residential_drop]["residential_percent_change_from_baseline"], name = "% residential movements"
        ), row=1, col=2)

        fig.update_layout({"plot_bgcolor" : "rgba(0,0,0,0)"},height=500, width=1300, title_text="Stacked Subplots", autosize=False, showlegend = True)
        return fig

    elif tab == "tab-0":
        ##### change here 
        if var == "gender":
            fig = px.bar(disease_grouped.reset_index(), x = "age_range", y = ["high blood pressure", "dyslipidemia", "kidney", "heart", "obesity", "cancer", "stroke", "diabetes"], facet_col = "gender")
            fig.update_layout({
            'plot_bgcolor': 'rgba(0, 0, 0, 0)',
            'paper_bgcolor': 'rgba(0, 0, 0, 0)',
            })
            fig.update_layout(barmode = 'group', yaxis_tickangle = 45, uniformtext_minsize=15)
            for annotation in fig['layout']['annotations']: 
                annotation['textangle']= 0
            return fig
        elif var == "age":
            fig = px.bar(disease_grouped.reset_index(), x = "gender", y = ["high blood pressure", "dyslipidemia", "kidney", "heart", "obesity", "cancer", "stroke", "diabetes"], facet_col = "age_range")
            fig.update_layout({
            'plot_bgcolor': 'rgba(0, 0, 0, 0)',
            'paper_bgcolor': 'rgba(0, 0, 0, 0)',
            })
            fig.update_layout(barmode = 'group', yaxis_tickangle = 45, uniformtext_minsize=15)
            for annotation in fig['layout']['annotations']: 
                annotation['textangle']= 0
            return fig
        else:
            #change this
            print(disease)
            age_hbp = w.groupby(["age", "gender"])[[disease]].sum().reset_index()
            age_hbp[[disease]] = age_hbp[[disease]].astype(int)
            #age_hbp[disease[1]] = age_hbp[disease[1]].astype(int)
            fig = px.scatter(age_hbp, x = "age", y = [disease], color = "gender")

            fig.update_layout({
                "plot_bgcolor" : "rgba(0,0,0,0)"
            })

            return fig


    elif tab == "tab-5":
        fig = make_subplots(rows = 1, cols = 2, specs =[ [{"type" : "bar"}, {"type":"pie"}]], \
        subplot_titles=("Gender breakdown of death cases","Top underlying health conditions"))

        g = death_data.groupby("gender").size().reset_index(name = "count")
        fig.append_trace(go.Bar(
            x = g.gender,
            y = g["count"],marker=dict(color = g["count"]),  showlegend=False
        ), row = 1 , col = 1 )

    

        fig.add_trace(go.Pie(
            values = t20["count"], labels = t20["index"]
        ), row = 1, col = 2)
        fig.update_layout({"plot_bgcolor" : "rgba(0,0,0,0)"}, title_text=" ")
        
        return fig
    
    elif tab == "tab-6":
        if pca_type == "loading": 
            score = pca[:,0:2]
            coeff = np.transpose(pcamodel.components_[0:2, :])
            xs = score[:,0]
            ys = score[:,1]
            n = coeff.shape[0]
            scalex = 1.0/(xs.max() - xs.min())
            scaley = 1.0/(ys.max() - ys.min())

            fig = px.scatter(x = xs * scalex,y = ys * scaley, title = "PCA loadings plot of first two principal")

            fig.update_layout({"plot_bgcolor":"rgba(0,0,0,0)"}, yaxis = {"title" : f"PC2 {explained_variance[1]*100:.2f}%"},
                            xaxis = {"title" : f"PC1 {explained_variance[0]*100:.2f}%"})

            for i in range(n):
                
                fig.add_trace(go.Scatter(x=[0,coeff[i,0]], y=[0,coeff[i,1]],
                                    mode='lines',
                                    name=f"{x.columns[i]}", text = f"{x.columns[i]}", textposition="bottom left"))

            #fig.update_traces(textposition='top center')
            return fig
        else:
            fig = make_subplots(rows = 1, cols = 2,
                   subplot_titles = ("Dimension 1", "Dimension 2"))
        # Dimension indexing
        dimensions = dimensions = ['Dimension {}'.format(i) for i in range(1,len(pcamodel.components_)+1)]

        # PCA components
        components = pd.DataFrame(np.round(pcamodel.components_, 4), columns = list(pcaa.keys()))
        components.index = dimensions
        for_plotting = components[0:2].reset_index()
        for_plotting = for_plotting.rename(columns = {"index" : "dim"})

        # PCA explained variance
        ratios = pcamodel.explained_variance_ratio_.reshape(len(pcamodel.components_), 1)
        variance_ratios = pd.DataFrame(np.round(ratios, 4), columns = ['Explained Variance'])
        variance_ratios.index = dimensions
        xxx = for_plotting
        xxx = xxx.melt(id_vars='dim')

        fig.append_trace(go.Bar(name = "PC1",
            x = xxx[xxx["dim"] == "Dimension 1"].variable,
            y = xxx[xxx["dim"] == "Dimension 1"].value,marker=dict(color = xxx[xxx["dim"] == "Dimension 1"].value, colorscale="viridis", colorbar=dict(x=0.45,thickness=10)),  showlegend=False
        ), row = 1 , col = 1 )


        fig.append_trace(go.Bar(name = "PC2",
            x = xxx[xxx["dim"] == "Dimension 2"].variable,
            y = xxx[xxx["dim"] == "Dimension 2"].value,marker=dict(color = xxx[xxx["dim"] == "Dimension 2"].value, colorscale="inferno", colorbar=dict(thickness=10)),  showlegend=False
        ), row = 1 , col = 2 )
        fig.update_layout({"plot_bgcolor":"rgba(0,0,0,0)"}, barmode='group')
        return fig
    elif tab == "tab-m":
        if chart == "line":
            fig = px.line(my_state, x="date", y="Daily cases", color='States')

            fig.update_layout({
            'plot_bgcolor': 'rgba(0, 0, 0, 0)',
            'paper_bgcolor': 'rgba(0, 0, 0, 0)'})
            return fig
        else:

            fig = make_subplots(rows = 2, cols = 1)

            date = my_state.sort_values("date")["date"].values[0]
            f = my_state.groupby("States").sum("Daily cases").reset_index().sort_values("Daily cases")
            f.rename(columns = {"Daily cases" : "Cumulative cases"}, inplace = True)

            fig.append_trace(go.Bar(
                x = f.States,
                y = f["Cumulative cases"],marker=dict(color = f["Cumulative cases"])), row = 2, col = 1)

            death_data["state"] = death_data["state"].str.strip()
            k = death_data.groupby("state").size().reset_index(name="count")
            k = k.sort_values("count")
            fig.append_trace(go.Bar(
                x = k["state"],
                y = k["count"],marker=dict(color = k["count"],
                        colorscale='blugrn')), row = 1, col = 1)

            fig.update_layout({"plot_bgcolor" : "rgba(0,0,0,0)"}, showlegend = False)
            fig.update_xaxes(title_text="States", row=1, col=1)
            fig.update_xaxes(title_text="States", row=2, col=1)
            fig.update_yaxes(title_text="Cumulative cases", row=1, col=1)
            fig.update_yaxes(title_text="Cumulative deaths", row=2, col=1)

            return fig

if __name__ == '__main__':
    app.run_server(debug=True, port = "8080")
