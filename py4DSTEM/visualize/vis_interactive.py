
import numpy as np

from bqplot import *
from bqplot_image_gl import ImageGL
from bqplot_image_gl.interacts import MouseInteraction

from IPython.display import display
# from ipywidgets import *
import ipywidgets as widgets


# import numpy as np
# from mpl_toolkits.axes_grid1 import make_axes_locatable

# # import matplotlib.pyplot as plt
# # import plotly.graph_objects as go

# # import bqplot
# # from bqplot import pyplot as plt
# from bqplot import *

# from bqplot.interacts import (
#     FastIntervalSelector, IndexSelector, BrushIntervalSelector,
#     BrushSelector, MultiSelector, LassoSelector,
# )





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


    # # define coordinate axes
    # x = np.arange(data4D.data.shape[1])
    # y = np.arange(data4D.data.shape[0])
    # qx = np.arange(data4D.data.shape[3])
    # qy = np.arange(data4D.data.shape[2])


    # Initial image slices
    real_im = data4D.data[:,:,qxy[0],qxy[1]]
    diff_im = data4D.data[xy[0],xy[1],:,:]
    # real_im = np.flip(real_im, axis=0)
    # diff_im = np.flip(diff_im, axis=0)
    # diff_im = np.maximum(diff_im,0)**0.1

    # Initial color ranges
    real_int_mean = np.mean(real_im)
    real_int_std = np.std(real_im)
    real_int_range = np.array((real_int_mean - 3*real_int_std, real_int_mean + 3*real_int_std))
    diff_int_std = np.std(diff_im)
    diff_int_range = np.array((0, 8*diff_int_std))

    # Scale images from 0 to 1 for plotting
    real_im_scale = (real_im - real_int_range[0]) / (real_int_range[1] - real_int_range[0])
    diff_im_scale = (diff_im - diff_int_range[0]) / (diff_int_range[1] - diff_int_range[0])



    # instantiate real space view
    # scale_x = LinearScale(min=0, max=data4D.data.shape[1])
    # scale_y = LinearScale(min=0, max=data4D.data.shape[0])
    scales = {'x': LinearScale(min=0, max=data4D.data.shape[1]),
              'y': LinearScale(min=0, max=data4D.data.shape[0])}
    scales = {'x': LinearScale(allow_padding=False), 'y': LinearScale(allow_padding=False)}
    x_axis = Axis(scale=scales['x'])
    y_axis = Axis(scale=scales['y'], orientation='vertical')
    fig = Figure(scales=scales,
                 axes=[x_axis, y_axis],
                 min_aspect_ratio=1,
                 max_aspect_ratio=1,
                 fig_margin={'top': 10, 'bottom': 50, 'left': 50, 'right': 10})
    color_scale = ColorScale(min=real_int_range[0],
                             max=real_int_range[1], 
                             scheme=cmap_real)

    image_real = ImageGL(image=real_im_scale, 
                         scales = {'x': fig.axes[0].scale, 'y': fig.axes[1].scale, 'image': color_scale})
    fig.marks = [image_real]




        
    # # fig
    display(fig)
    # # print(image_real.image.shape)
    # image_real.image = real_im_scale
    # image_real.x = [0, real_im.shape[0]]
    # image_real.y = [0, real_im.shape[1]]

    # print(np.min(image_real.image),np.max(image_real.image))
    # # print(image_real.image.shape)
    # display(widgets.HBox([fig]))



    # return fig
    # display(widgets.HBox([fig]))
    # fig
    # display(fig)

    # Figure(axes=[ax_x, ax_y], marks=[image_real])


    # Add image data to axes
    # image_real.image = real_im


    # scales = {'x': LinearScale(allow_padding=False), 'y': LinearScale(allow_padding=False)}

    # x_axis = Axis(scale=scales['x'])
    # y_axis = Axis(scale=scales['y'], orientation='vertical')

    # fig = Figure(scales=scales,
    #             axes=[x_axis, y_axis],
    #             min_aspect_ratio=1,
    #             max_aspect_ratio=1,
    #             fig_margin={'top': 10, 'bottom': 50, 'left': 50, 'right': 10})

    # color_scale = ColorScale(min=0, max=1, scheme='viridis')

    # image = ImageGL(image=np.zeros((0,0)), # we set empty array to instantiate
    #                 scales = {'x': fig.axes[0].scale, 'y': fig.axes[1].scale, 'image': color_scale})

    # fig.marks = [image]


    # display(fig)

    

    # create_panel(BrushSelector, orientation='vertical', scale=scales['y'])
    # create_panel(BrushSelector)

    # # Make marks for the 2 panels
    # fig = [];



    # # Diffraction
    # # selector = BrushSelector
    # # sel = selector(marks=[scatter], x_scale=scales['x'], y_scale=scales['y'])
    # # text_brush = HTML()
    # # if selector != LassoSelector:
    # #     def update_text(*args):
    # #         text_brush.value = '{}.selected = {}'.format(selector.__name__, sel.selected)
    # #     sel.observe(update_text, 'selected')
    # #     update_text()

    # diff_x_sc = LinearScale()
    # diff_y_sc = LinearScale()
    # diff_col_sc = ColorScale(scheme=cmap_diff)
    # diff_heat = HeatMap(x=qx, 
    #                     y=qy, 
    #                     color=diff_im,
    #                     scales={'x': diff_x_sc, 'y': diff_y_sc, 'color': diff_col_sc})
    # diff_ax_x = Axis(scale=diff_x_sc, visible=False)
    # diff_ax_y = Axis(scale=diff_y_sc, 
    #                  orientation='vertical', visible=False)
    # diff_ax_c = ColorAxis(scale=diff_col_sc)


    # # selector = BrushSelector
    # # sel = selector(marks=[diff_heat], x_scale=diff_x_sc, y_scale=diff_y_sc)
    # # # sel = BrushSelector(marks=[diff_heat], x_scale=scales['x'], y_scale=scales['y'])
    # # text_brush = HTML()
    # # def update_text(*args):
    # #     text_brush.value = '{}.selected = {}'.format(selector.__name__, sel.selected)
    # # sel.observe(update_text, 'selected')
    # # update_text()


    # fig.append(Figure(marks=[diff_heat], axes=[diff_ax_x, diff_ax_y, diff_ax_c],
    #                  title='Mean Diffraction Pattern',
    #                  layout=Layout(width='500px', height='500px'),
    #                  min_aspect_ratio=1, max_aspect_ratio=1, padding_y=0))


    # # display(diff_fig)
    # # display(widgets.HBox(fig))

    # real_x_sc = LinearScale()
    # real_y_sc = LinearScale()
    # real_col_sc = ColorScale(scheme=cmap_real)
    # real_heat = HeatMap(x=x, 
    #                     y=y, 
    #                     color=real_im,
    #                     scales={'x': real_x_sc, 'y': real_y_sc, 'color': real_col_sc})
    # real_ax_x = Axis(scale=real_x_sc, visible=False)
    # real_ax_y = Axis(scale=real_y_sc, 
    #             orientation='vertical', visible=False)
    # real_ax_c = ColorAxis(scale=real_col_sc)
    # fig.append(Figure(marks=[real_heat], axes=[real_ax_x, real_ax_y, real_ax_c],
    #                  title='Virtual Detector Image',
    #                  layout=Layout(width='500px', height='500px'),
    #                  min_aspect_ratio=1, max_aspect_ratio=1, padding_y=0))

    # display(widgets.HBox(fig))

