import numpy as np
import plotly.graph_objs as go
from plotly.offline import iplot, plot

import cufflinks
cufflinks.go_offline()
 

cufflinks.set_config_file(world_readable=True, theme='pearl', offline=True)



def makeData(func):
    """
    makes data to plot in 3D with given function
    """
    x = np.arange (-10, 10, 0.1)
    y = np.arange (-10, 10, 0.1)
    xgrid, ygrid = np.meshgrid(x, y)

    #zgrid = xgrid**2+2*ygrid**2+0.012*xgrid*ygrid-2*ygrid+xgrid
    
    zgrid = func(xgrid, ygrid)
    
    return xgrid, ygrid, zgrid

    
def plot_function(func, title, only_save=False):
    """
    plots given function with setting the title
    """
    x, y, z = makeData(func)
    
    surface = go.Surface(x=x, y=y, z=z)
    
    data = [surface]

    layout = go.Layout(
        title=title,
        scene=dict(
            xaxis=dict(
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)'
            ),
            yaxis=dict(
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)'
            ),
            zaxis=dict(
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)'
            )
        )
    )

    fig = go.Figure(data=data, layout=layout)
    if only_save:
        plot(fig, filename=title)
    else:
        iplot(fig, filename=title)
    
    
def plot_contour(func, title, only_save=False):
    """
    plots contour of a given function with setting a title
    """
    x, y, z = makeData(func)
    
    Z = [func(el, el1) for el, el1 in zip(x[0], y[:, 0])]
    
    trace1 = go.Contour(
    z=Z,
    y=y[:, 0],
    x=x[0],
    contours=dict(
            coloring ='heatmap',
            showlabels = True,
            labelfont = dict(
                family = 'Raleway',
                size = 12,
                color = 'white')))
    
    lay = go.Layout(title=title)
    
    fig=go.Figure([trace1], lay)
    
    if only_save:
        plot(fig, filename=title)
    else:
        iplot(fig, filename=title)

def plot_contour_and_scatter_of_descent(func, results_of_method, title, only_save=False):
    """
    plots contour of a function and a scatter plot of the steps of the method with setting a title
    """
    x, y, z = makeData(func)
    
    Z = [func(el, el1) for el, el1 in zip(x[0], y[:, 0])]
    
    trace1 = go.Contour(
    z=Z,
    y=y[:, 0],
    x=x[0],
    contours=dict(
            coloring ='heatmap',
            showlabels = True,
            labelfont = dict(
                family = 'Raleway',
                size = 12,
                color = 'white')))

    trace2 = go.Scatter(
        x=[el[0] for el in results_of_method],
        y=[el[1] for el in results_of_method],
        mode='markers+lines',
        name='steepest',
        line=dict(
            color='black'))
    lay = go.Layout(title=title)
    
    fig=go.Figure([trace1,trace2], lay)
    
    if only_save:
        plot(fig, filename=title)
    else:
        iplot(fig, filename=title)
