import numpy as np
from ipywidgets import *
from mpl_toolkits.axes_grid1 import make_axes_locatable

# import matplotlib.pyplot as plt
# import plotly.graph_objects as go

# import bqplot
# from bqplot import pyplot as plt
from bqplot import *






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
    x = np.arange(data4D.data.shape[0])
    y = np.arange(data4D.data.shape[1])
    qx = np.arange(data4D.data.shape[2])
    qy = np.arange(data4D.data.shape[3])


    # Get image slices
    real_im = data4D.data[:,:,qxy[0],qxy[1]]
    diff_im = data4D.data[xy[0],xy[1],:,:]

    fig = [];

    diff_x_sc = LinearScale()
    diff_y_sc = LinearScale()
    diff_col_sc = ColorScale(scheme=cmap_diff)
    diff_heat = HeatMap(x=qx, 
                        y=qy, 
                        color=diff_im,
                        scales={'x': diff_x_sc, 'y': diff_y_sc, 'color': diff_col_sc})
    diff_ax_x = Axis(scale=diff_x_sc)
    diff_ax_y = Axis(scale=diff_y_sc, 
                     orientation='vertical')
    diff_ax_c = ColorAxis(scale=diff_col_sc)
    fig.append(Figure(marks=[diff_heat], axes=[diff_ax_x, diff_ax_y, diff_ax_c],
                     title='Mean Diffraction Pattern',
                     layout=Layout(width='450px', height='450px'),
                     min_aspect_ratio=1, max_aspect_ratio=1, padding_y=0))

    real_x_sc = LinearScale()
    real_y_sc = LinearScale()
    real_col_sc = ColorScale(scheme=cmap_real)
    real_heat = HeatMap(x=x, 
                        y=y, 
                        color=real_im,
                        scales={'x': real_x_sc, 'y': real_y_sc, 'color': real_col_sc})
    real_ax_x = Axis(scale=real_x_sc)
    real_ax_y = Axis(scale=real_y_sc, 
                orientation='vertical')
    real_ax_c = ColorAxis(scale=real_col_sc)
    fig.append(Figure(marks=[real_heat], axes=[real_ax_x, real_ax_y, real_ax_c],
                     title='Virtual Detector Image',
                     layout=Layout(width='450px', height='450px'),
                     min_aspect_ratio=1, max_aspect_ratio=1, padding_y=0))


    display(widgets.HBox(fig, align_content = 'fill'))

    fig

    # fig = []
    # fig.append(Figure(layout=fig_layout, 
    #               axes=[ax_x, ax_y], 
    #               marks=[line], 
    #               title='Example 1', 
    #               fig_margin = dict(top=30, bottom=10, left=20, right=20)
    #               ))

    # size = 100
    # scale = 100.
    # np.random.seed(0)
    # x_data = np.arange(size)
    # y_data = np.cumsum(np.random.randn(size)  * scale)

    # x_sc = LinearScale()
    # y_sc = LinearScale()

    # ax_x = Axis(label='X', scale=x_sc, grid_lines='solid')
    # ax_y = Axis(label='Y', scale=y_sc, orientation='vertical', grid_lines='solid')

    # line = Lines(x=x_data, y=x_data, scales={'x': x_sc, 'y': y_sc})

    # fig_layout = widgets.Layout(width='auto', height='auto')


    # figx = list()
    # figx[0] = Figure(layout=fig_layout, 
    #               axes=[ax_x, ax_y], 
    #               marks=[line], 
    #               title='Example 1', 
    #               fig_margin = dict(top=30, bottom=10, left=20, right=20)
    #               )
    # figx[1] = Figure(layout=fig_layout, 
    #               axes=[ax_x, ax_y], 
    #               marks=[line], 
    #               title='Example 2', 
    #               fig_margin = dict(top=30, bottom=10, left=20, right=20)
    #               )
    # display(widgets.HBox(figx, align_content = 'stretch'))
    # figy=[]
    # for i in range(2):
    #     figx=[]
    #     for j in range(3):
        #     figx.append(Figure(layout=fig_layout, axes=[ax_x, ax_y], marks=[line], title='Example' + str(i*3+j), fig_margin = dict(top=30, bottom=10, left=20, right=20)))
        # figy.append(widgets.HBox(figx))
    # display(widgets.VBox(figy, align_content = 'stretch'))

    # Generate plots
    # diff_axes_options = {'color': {'orientation': "vertical","side":"left"}}
    # fig = plt.figure(title='4D Browser')
    # plt.heatmap(diff_im, axes_options=diff_axes_options)

    # bar_chart = plt.bar([1, 2, 3, 4, 5], [2, 9, 10, 40, 40])
    # line_chart = plt.plot([1, 2, 3, 4, 5], [10, 5, 30, 60, 20], 'rs-')
    # fig

    # fig  = plt.figure(title="Mean Diffraction Pattern",
    #                        padding_y=0)
    # fig.layout.width = "900px"
    # fig.layout.height = "450px"

    # diff_axes_options = {'color': {'orientation': "vertical","side":"left"},
    #                     'scales': {}
    #                     }
    # diff_plot = plt.heatmap(diff_im, axes_options=diff_axes_options)

    # real_axes_options = {'color': {'orientation': "vertical","side":"right"}}
    # real_plot = plt.heatmap(real_im, axes_options=real_axes_options)

    # plt.show()
# fig

    # diff_fig  = plt.figure(title="Mean Diffraction Pattern",
    #                        padding_y=0)
    # diff_fig.layout.width = "900px"
    # diff_fig.layout.height = "450px"
    # diff_axes_options = {'color': {'orientation': "vertical","side":"left"}}
    # diff_plot = plt.heatmap(diff_im, axes_options=diff_axes_options)


    
    # plt.show()


    # real_fig  = plt.figure(title="Mean Diffraction Pattern",padding_y=0)
    # real_fig.layout.width = "200px"
    # real_fig.layout.height = "200px"
    # real_axes_options = {'color': {'orientation': "vertical","side":"left"}}
    # plt.heatmap(real_im, axes_options=real_axes_options)
    # plt.show()





    # # initialize figure and axes
    # fig,ax = plt.subplots(1,2,figsize=figsize)

    # # divider = make_axes_locatable(ax)
    # # cax = divider.append_axes('right', size='5%', pad=0.05)

    # # im = ax.imshow(im_real, cmap='bone')

    # # fig.colorbar(im, cax=cax, orientation='vertical')
    # # plt.show()

    # # # plot the initial images
    # h0 = ax[0].imshow(im_diff, cmap=cmap_diff)
    # h1 = ax[1].imshow(im_real, cmap=cmap_real)

    # # appearance of diffraction image
    # ax[0].axis('off')
    # divider0 = make_axes_locatable(ax[0])
    # cax0 = divider0.append_axes('left', size='5%', pad=0.10)
    # fig.colorbar(h0, cax=cax0, orientation='vertical')
    # cax0.yaxis.set_ticks_position('left')

    # # appearance of real space image
    # ax[1].axis('off')
    # divider1 = make_axes_locatable(ax[1])
    # cax1 = divider1.append_axes('right', size='5%', pad=0.10)
    # fig.colorbar(h1, cax=cax1, orientation='vertical')


    # # Interactive elements
    # def update_qx(qx):
    #     f.layout.xaxis.range = [start, end]

    # # Add diffraction slider bar selectors
    # interact(draw_browse_4D, qx=(0,data4D.data.shape[0]));


    # # initial axis 0 appearance
    # ax[0].axis('off')
    # divider = make_axes_locatable(ax[0])
    # cax = divider.append_axes('left', size='5%', pad=0.10)
    # ax[0].colorbar(h0, cax=cax, orientation='vertical')

    # ax[1].axis('off')


    # for col in ax:
    #     im = col.imshow(data, cmap='bone')
    #     divider = make_axes_locatable(col)
    #     cax = divider.append_axes('right', size='5%', pad=0.05)
    #     fig.colorbar(im, cax=cax, orientation='vertical')
    # plt.show()

    # return True
    # cax = ax.matshow(_ar,vmin=vmin,vmax=vmax,cmap=cmap,**kwargs)
    #     if np.any(_ar.mask==True):
    #         mask_display = np.ma.array(data=_ar.mask,mask=_ar.mask==False)
    #         cmap = ListedColormap([mask_color,'w'])
    #         ax.matshow(mask_display,cmap=cmap)


# def draw_browse_4D(qx):
# 	print(qx)
