import numpy as np
from ipywidgets import *
from mpl_toolkits.axes_grid1 import make_axes_locatable

# import matplotlib.pyplot as plt
# import plotly.graph_objects as go

# import bqplot
# from bqplot import pyplot as plt
from bqplot import *

from bqplot.interacts import (
    FastIntervalSelector, IndexSelector, BrushIntervalSelector,
    BrushSelector, MultiSelector, LassoSelector,
)


def browse4D(data4D, figsize=(16,8)):
    
    """
    Single image visualization of 4D datacube controlled by sliders

    Accepts:
        data4D - py4DSTEM datacube
        figsize - 2 element list giving figure size

    Returns:

    """

    # default inputs
    cmap_diff = 'inferno'
    cmap_real = 'viridis'


    # initialize image slice selection
    xy = np.array((np.round(data4D.data.shape[0]/2).astype(int),
    				np.round(data4D.data.shape[1]/2).astype(int)))
    qxy = np.array((np.round(data4D.data.shape[2]/2).astype(int),
    				np.round(data4D.data.shape[3]/2).astype(int)))

    # define coordinate axes
    x = np.arange(data4D.data.shape[1])
    y = np.arange(data4D.data.shape[0])
    qx = np.arange(data4D.data.shape[3])
    qy = np.arange(data4D.data.shape[2])


    # Get image slices
    real_im = data4D.data[:,:,qxy[0],qxy[1]]
    diff_im = data4D.data[xy[0],xy[1],:,:]
    real_im = np.flip(real_im, axis=0)
    diff_im = np.flip(diff_im, axis=0)
    diff_im = np.maximum(diff_im,0)**0.1

    # create_panel(BrushSelector, orientation='vertical', scale=scales['y'])
    # create_panel(BrushSelector)

    # Make marks for the 2 panels
    fig = [];



    # Diffraction
    # selector = BrushSelector
    # sel = selector(marks=[scatter], x_scale=scales['x'], y_scale=scales['y'])
    # text_brush = HTML()
    # if selector != LassoSelector:
    #     def update_text(*args):
    #         text_brush.value = '{}.selected = {}'.format(selector.__name__, sel.selected)
    #     sel.observe(update_text, 'selected')
    #     update_text()

    diff_x_sc = LinearScale()
    diff_y_sc = LinearScale()
    diff_col_sc = ColorScale(scheme=cmap_diff)
    diff_heat = HeatMap(x=qx, 
                        y=qy, 
                        color=diff_im,
                        scales={'x': diff_x_sc, 'y': diff_y_sc, 'color': diff_col_sc})
    diff_ax_x = Axis(scale=diff_x_sc, visible=False)
    diff_ax_y = Axis(scale=diff_y_sc, 
                     orientation='vertical', visible=False)
    diff_ax_c = ColorAxis(scale=diff_col_sc)


    # selector = BrushSelector
    # sel = selector(marks=[diff_heat], x_scale=diff_x_sc, y_scale=diff_y_sc)
    # # sel = BrushSelector(marks=[diff_heat], x_scale=scales['x'], y_scale=scales['y'])
    # text_brush = HTML()
    # def update_text(*args):
    #     text_brush.value = '{}.selected = {}'.format(selector.__name__, sel.selected)
    # sel.observe(update_text, 'selected')
    # update_text()


    fig.append(Figure(marks=[diff_heat], axes=[diff_ax_x, diff_ax_y, diff_ax_c],
                     title='Mean Diffraction Pattern',
                     layout=Layout(width='500px', height='500px'),
                     min_aspect_ratio=1, max_aspect_ratio=1, padding_y=0))


    # display(diff_fig)
    # display(widgets.HBox(fig))

    real_x_sc = LinearScale()
    real_y_sc = LinearScale()
    real_col_sc = ColorScale(scheme=cmap_real)
    real_heat = HeatMap(x=x, 
                        y=y, 
                        color=real_im,
                        scales={'x': real_x_sc, 'y': real_y_sc, 'color': real_col_sc})
    real_ax_x = Axis(scale=real_x_sc, visible=False)
    real_ax_y = Axis(scale=real_y_sc, 
                orientation='vertical', visible=False)
    real_ax_c = ColorAxis(scale=real_col_sc)
    fig.append(Figure(marks=[real_heat], axes=[real_ax_x, real_ax_y, real_ax_c],
                     title='Virtual Detector Image',
                     layout=Layout(width='500px', height='500px'),
                     min_aspect_ratio=1, max_aspect_ratio=1, padding_y=0))

    display(widgets.HBox(fig))

