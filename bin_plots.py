import numpy as np
import matplotlib.pyplot as plt

def plot_combined_binmap_from_my_list(my_list,binmap_savefig_location):
    """
    Combine and plot a single binmap from my_list, where each element contains
    bin number, coordinates of spaxels in the bin, S/N, coadded flux, coadded variance, and flag.

    Parameters
    ----------
    my_list : list
        A list where each element contains [bin_no, Coordinates, S/N, coadded flux, coadded variance, flag].
    """
    bins = []
    coordinates = []
    sn_values_line_1 = []  # To store the S/N values
    sn_values_line_2 = []
    sn_values_line_3 = []
    sn_values_line_4 = []
    # Extract bin number, coordinates, and S/N values from my_list
    for row in my_list:
        bin_number = row[0]  # Bin number
        coords = row[1]      # Coordinates of spaxels in the bin
        sn_line_1 = row[3]          # Signal-to-noise ratio
        sn_line_2 = row[4]
        sn_line_3 = row[5]
        sn_line_4 = row[6]
        bins.append(bin_number)
        coordinates.append(coords)
        sn_values_line_1.append(sn_line_1)
        sn_values_line_2.append(sn_line_2)
        sn_values_line_3.append(sn_line_3)
        sn_values_line_4.append(sn_line_4)

    # Flatten the list of coordinates to get the x and y coordinates separately
    x_coords = np.array([coord[0] for coord_set in coordinates for coord in coord_set])
    y_coords = np.array([coord[1] for coord_set in coordinates for coord in coord_set])

    x_min, x_max = int(np.min(x_coords)), int(np.max(x_coords))
    y_min, y_max = int(np.min(y_coords)), int(np.max(y_coords))

    # Initialize the binmap with -1 (indicating no bin assigned)
    binmap = np.full((y_max - y_min + 1, x_max - x_min + 1), -1, dtype=int)

    # Assign bin numbers to the binmap based on coordinates
    for bin_number, coord_set in zip(bins, coordinates):
        coords = np.array(coord_set)
        x, y = coords[:, 0], coords[:, 1]
        binmap[(y - y_min).astype(int), (x - x_min).astype(int)] = bin_number

    # Plot the binmap using imshow
    fig, ax = plt.subplots(figsize=(5, 5))
    cax = ax.imshow(binmap, origin='lower', cmap='tab20', extent=[x_min, x_max + 1, y_min, y_max + 1])

    # Add a colorbar
    fig.colorbar(cax, ax=ax, label='Bin Number')

    # Labeling the plot
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    #ax.set_title('Combined Binmap from my_list')
    ax.grid(True)

    # Function to update the displayed value when hovering over the plot
    def on_move(event):
        # Only show the annotation if the mouse is over the image
        if event.inaxes == ax:
            # Get the mouse position (in pixels)
            mouse_x, mouse_y = event.x, event.y

            # Convert mouse position to data coordinates
            data_x, data_y = ax.transData.inverted().transform([mouse_x, mouse_y])

            # Find the nearest bin number in the binmap
            bin_x = int(np.clip(data_x - x_min, 0, binmap.shape[1] - 1))
            bin_y = int(np.clip(data_y - y_min, 0, binmap.shape[0] - 1))

            # Check if we're on a valid bin
            bin_value = binmap[bin_y, bin_x]

            # Find the corresponding index in my_list to get the sn_unbin value
            if bin_value != -1:
                # Find the coordinates that correspond to this bin
                idx = np.where(bins == bin_value)[0][0]
                sn_value_line_1 = sn_values_line_1[idx]
                sn_value_line_2 = sn_values_line_2[idx]
                sn_value_line_3 = sn_values_line_3[idx]
                sn_value_line_4 = sn_values_line_4[idx]

                # Create a label showing the sn_unbin value
                ax.set_title(f"Bin ID: {bin_value}; S/N values: \nOII: {round(sn_value_line_1,2)}, H$\\gamma$: {round(sn_value_line_2,2)}, \n H$\\beta$: {round(sn_value_line_3,2)}, OIII: {round(sn_value_line_4,2)}")
            else:
                ax.set_title('Binmap')

            # Redraw the plot to update the title
            fig.canvas.draw_idle()

    # Connect the event to the plot
    fig.canvas.mpl_connect('motion_notify_event', on_move)
    plt.savefig(binmap_savefig_location+'binmap.jpeg',dpi=300)
    # Display the plot
    plt.close()

def plot_combined_binmap_no_interactive(my_list, ax=None, alpha=0.5):
    """
    Plot a combined binmap from my_list, overlaying it on an existing figure if provided.

    Parameters
    ----------
    my_list : list
        A list where each element contains [bin_no, Coordinates, S/N, coadded flux, coadded variance, flag].
    ax : matplotlib.axes._subplots.Axes, optional
        If provided, the function will plot on this axis. If None, it will use the current active axis.
    alpha : float, optional
        Transparency level for overlaying plots (default is 0.5).
    """
    bins = []
    coordinates = []

    for row in my_list:
        bins.append(row[0])        # Bin number
        coordinates.append(row[1]) # Coordinates of spaxels in the bin

    # Flatten coordinates to get x and y values
    x_coords = np.array([coord[0] for coord_set in coordinates for coord in coord_set])
    y_coords = np.array([coord[1] for coord_set in coordinates for coord in coord_set])

    x_min, x_max = int(np.min(x_coords)), int(np.max(x_coords))
    y_min, y_max = int(np.min(y_coords)), int(np.max(y_coords))

    # Initialize binmap
    binmap = np.full((y_max - y_min + 1, x_max - x_min + 1), -1, dtype=int)

    # Assign bin numbers
    for bin_number, coord_set in zip(bins, coordinates):
        coords = np.array(coord_set)
        x, y = coords[:, 0], coords[:, 1]
        binmap[(y - y_min).astype(int), (x - x_min).astype(int)] = bin_number

    # Use provided axis or get the current one
    if ax is None:
        ax = plt.gca()

    # Plot binmap overlay
    cax = ax.imshow(binmap, origin='lower', cmap='tab20', extent=[x_min, x_max + 1, y_min, y_max + 1], alpha=alpha)

    # Remove interactivity
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')

    return cax  # Return the color map in case further modifications are needed
