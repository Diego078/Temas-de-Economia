"""SpaceX Falcon 9 Capstone — interactive Plotly Dash application.

Requirements: pip install dash plotly pandas
Run from the folder containing spacex_model_compact.csv:
    python SpaceX_Dash.py
Then open http://127.0.0.1:8050

Data: independently reconstructed, 90 historical observations from a public copy
of IBM course dataset_part_2.csv; PayloadMass rounded to the nearest kilogram.
"""
from pathlib import Path
import pandas as pd
import plotly.express as px
from dash import Dash, Input, Output, dcc, html

DATA = Path(__file__).with_name('spacex_model_compact.csv')
cols = ['FlightNumber','Year','PayloadMass','OrbitCode','SiteCode','Flights',
        'GridFins','Reused','Legs','Block','ReusedCount','Class']
df = pd.read_csv(DATA, header=None, names=cols)
site_names = {0:'CCAFS SLC 40', 1:'VAFB SLC 4E', 2:'KSC LC 39A'}
orbit_names = ['LEO','ISS','PO','GTO','ES-L1','SSO','HEO','MEO','VLEO','SO','GEO']
df['LaunchSite'] = df.SiteCode.map(site_names)
df['Orbit'] = df.OrbitCode.map(lambda x: orbit_names[x] if x >= 0 else orbit_names[10])
df['LandingStatus'] = df.Class.map({0:'Not successful',1:'Successful'})

app = Dash(__name__)
app.title = 'Falcon 9 Launch Analytics | IBM Capstone'
NAVY = '#10243A'; TEAL = '#12A69A'
style_box = {'padding':'20px','backgroundColor':'white','borderRadius':'10px'}
app.layout = html.Div([
    html.H1('Falcon 9 Landing Outcomes', style={'textAlign':'center','backgroundColor':NAVY,'color':'white','padding':'23px','margin':'0'}),
    html.Div([
        html.Div([html.Label('Launch site'),dcc.Dropdown(id='site',options=[{'label':'All launch sites','value':'ALL'}]+[{'label':v,'value':v} for v in site_names.values()],value='ALL',clearable=False)],style={'flex':'1'}),
        html.Div([html.Label('Payload mass (kg)'),dcc.RangeSlider(id='payload',min=0,max=16000,step=500,value=[0,16000],marks={i:str(i) for i in range(0,16001,4000)})],style={'flex':'2'})
    ],style={'display':'flex','gap':'45px','padding':'25px','backgroundColor':'#EFF5F8'}),
    html.Div([dcc.Graph(id='site-pie'),dcc.Graph(id='payload-scatter'),dcc.Graph(id='year-line'),dcc.Graph(id='orbit-bars')],style={'display':'grid','gridTemplateColumns':'1fr 1fr','gap':'8px','margin':'12px'})
],style={'fontFamily':'Arial','backgroundColor':'#EFF5F8','minHeight':'100vh'})

@app.callback(Output('site-pie','figure'),Output('payload-scatter','figure'),Output('year-line','figure'),Output('orbit-bars','figure'),Input('site','value'),Input('payload','value'))
def update_graphs(site, payload):
    d = df[(df.PayloadMass >= payload[0]) & (df.PayloadMass <= payload[1])]
    if site != 'ALL': d = d[d.LaunchSite == site]
    if d.empty:
        return tuple(px.scatter(title='No records match these filters') for _ in range(4))
    pie=px.pie(d,names='LandingStatus',title='Landing outcome distribution',color='LandingStatus',color_discrete_map={'Successful':TEAL,'Not successful':'#F1A15A'})
    scatter=px.scatter(d,x='PayloadMass',y='FlightNumber',color='LandingStatus',hover_data=['Year','LaunchSite','Orbit'],title='Payload mass and flight sequence',color_discrete_map={'Successful':TEAL,'Not successful':'#F1A15A'})
    annual=d.groupby('Year',as_index=False).Class.mean();annual['Class']*=100
    line=px.line(annual,x='Year',y='Class',markers=True,title='Success rate by year (%)');line.update_traces(line_color=TEAL)
    orb=d.groupby('Orbit',as_index=False).Class.mean();orb['Class']*=100
    bars=px.bar(orb.sort_values('Class'),x='Class',y='Orbit',orientation='h',title='Landing success rate by orbit (%)');bars.update_traces(marker_color=TEAL)
    for fig in (pie,scatter,line,bars):fig.update_layout(template='plotly_white',margin=dict(t=60,l=40,r=20,b=40),font=dict(family='Arial'))
    return pie,scatter,line,bars

if __name__ == '__main__':
    app.run(debug=False,port=8050)