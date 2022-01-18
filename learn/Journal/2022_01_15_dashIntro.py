# %%
import dash
import dash_html_components as html

import pandas as pd

app = dash.Dash()

app.layout = html.Div('My Dashboard 2')

if __name__ == '__main__':
    app.run_server(debug=True)

    