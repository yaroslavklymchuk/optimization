import numpy as np
import itertools
import matplotlib.pyplot as plt
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

    
def plot_function(functions, title, only_save=False):
    """
    plots given function with setting the title
    """
    #data = list(itertools.chain(*[[go.Surface(x=x, y=y, z=z) for x, y, z in makeData(func)] for func in functions]))
    data = []
    for i,func in enumerate(functions):
        x, y, z = makeData(func)
        
        if i>0:
            surface = go.Surface(x=x, y=y, z=z, showscale=False)
        else:
            surface = go.Surface(x=x, y=y, z=z)
        data.append(surface)
    
    """
    x, y, z = makeData(func)
    
    surface = go.Surface(x=x, y=y, z=z)
    
    data = [surface]
    """

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
        
        
        
def plot_implicit(fn, bbox=(-2.5,2.5)):
    ''' create a plot of an implicit function
    fn  ...implicit function (plot where fn==0)
    bbox ..the x,y,and z limits of plotted interval'''
    xmin, xmax, ymin, ymax, zmin, zmax = bbox*3
    fig = plt.figure(figsize=(50,50))
    ax = fig.add_subplot(111, projection='3d')
    A = np.linspace(xmin, xmax, 100) # resolution of the contour
    B = np.linspace(xmin, xmax, 15) # number of slices
    A1,A2 = np.meshgrid(A,A) # grid on which the contour is plotted

    for z in B: # plot contours in the XY plane
        X,Y = A1,A2
        Z = fn(X,Y,z)
        cset = ax.contour(X, Y, Z+z, [z], zdir='z')
        # [z] defines the only level to plot for this contour for this value of z

    for y in B: # plot contours in the XZ plane
        X,Z = A1,A2
        Y = fn(X,y,Z)
        cset = ax.contour(X, Y+y, Z, [y], zdir='y')

    for x in B: # plot contours in the YZ plane
        Y,Z = A1,A2
        X = fn(x,Y,Z)
        cset = ax.contour(X+x, Y, Z, [x], zdir='x')

    # must set plot limits because the contour will likely extend
    # way beyond the displayed level.  Otherwise matplotlib extends the plot limits
    # to encompass all values in the contour.
    ax.set_zlim3d(zmin,zmax)
    ax.set_xlim3d(xmin,xmax)
    ax.set_ylim3d(ymin,ymax)

    plt.show()
