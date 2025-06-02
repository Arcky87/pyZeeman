import astropy.io.fits as pyfits
import numpy as np

import logging

from numpy.polynomial.chebyshev import chebfit, chebval

from scipy.optimize import curve_fit
from scipy.signal import find_peaks

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from matplotlib.colors import ListedColormap
import seaborn as sns

from sklearn.cluster import AgglomerativeClustering
from skimage.exposure import equalize_hist

import warnings
warnings.simplefilter("ignore")
##################################################################

def order_tracer(dir_name, file_name, X_half_width, step, min_height, aperture, adaptive, view):
    poly_order = 2 # order of polynomial fit

    #read image
    hdulist = pyfits.open(dir_name.joinpath(file_name))
    ordim = hdulist[0].data.copy()
    hdulist.close()

    ny, nx = ordim.shape

    trace_im = np.zeros((ordim.shape[0], ordim.shape[1]), dtype=float)

    x_positions = np.arange(X_half_width, nx - X_half_width, X_half_width)

    # Search for local maxima in the cross sections and construct the mask of orders. 1 - max, 0 - rest of pixels
    for x_t in range(X_half_width, ordim.shape[1]-X_half_width, X_half_width):
        slic = np.median(ordim[:,x_t-X_half_width:x_t+X_half_width-1], 1)
        peaks_coord,peak_props = find_peaks(slic, height=min_height, width=3,prominence=1.8)

        if len(peaks_coord) > 14:
            # Take the strongest peaks
            peak_heights = peak_props['peak_heights']
            sorted_idx = np.argsort(peak_heights)[::-1]
            peaks_coord = peaks_coord[sorted_idx[:14]]

        if len(peaks_coord) > 0:
            trace_im[peaks_coord, x_t] = 1

 #       trace_im[peaks_coord[:14], x_t] = 1 # ES original

    # Now rearrange the data for the futher clustering
    total_points = np.sum(trace_im) # check if we have enough trace points
    print(f"Found {total_points} trace points")

    if total_points < 14 * 3:
        print("WARNING: Too few trace points detected. Adjusting parameters...")
        logging.warning("Insufficient trace points detected")
        return None, None

    ones = np.array(np.where(trace_im==1))
    ord_mask = ones.T
    try:
        model = AgglomerativeClustering(n_clusters=14,metric='euclidean',
                                        linkage='single',
                                        compute_full_tree=False,
                                        #distance_threshold=None
        )
        
        model.fit(ord_mask)
        labels = model.labels_
    
    except Exception as e:
        print(f"Clustering failed: {e}")
        return None, None
    
    n_clusters = len(np.unique(labels))
    print(f"Agglomerative clustering created {n_clusters} clusters")
    plot_cluster_maps(ordim, ord_mask, labels, n_clusters, trace_im,   # Plot cluster maps
                     save_plots=True, output_dir="./cluster_analysis")

 #   order_tab = [] # output table of traces
    width_tab = [] # output table of sizes

    order_tab = [] # output table of traces
    center_x = nx // 2
    min_points_required = max(3, len(x_positions) // 3)  # Require at least 1/3 of sampling points

 #   center_x = ordim.shape[1]/2 - 1
    for i in range(n_clusters):
        cluster_mask = labels == i
        cluster_points = ord_mask[cluster_mask]

        # Check if we have enough points spanning sufficient x-range
        x_coords = cluster_points[:, 1]  # x coordinates
        y_coords = cluster_points[:, 0]

        print(f"\nCluster {i}:")
        print(f"  Points: {len(x_coords)}")
        print(f"  X range: {np.min(x_coords)} to {np.max(x_coords)} (span: {np.max(x_coords) - np.min(x_coords)})")
        print(f"  Y range: {np.min(y_coords)} to {np.max(y_coords)} (span: {np.max(y_coords) - np.min(y_coords)})")
        print(f"  Y mean: {np.mean(y_coords):.1f}, Y std: {np.std(y_coords):.1f}")

        x_span = np.max(x_coords) - np.min(x_coords)
        min_x_span = nx * 0.95  # Require order to span at least 70% of detector


        if len(x_coords) >= min_points_required and x_span >= min_x_span:
            try:
                # FIX 8: Proper Chebyshev fitting with error handling
                # Sort by x coordinate for proper fitting
                sort_idx = np.argsort(x_coords)
                x_sorted = x_coords[sort_idx]
                y_sorted = y_coords[sort_idx]
                
                cheb_coef = chebfit(x_sorted, y_sorted, poly_order)
                
                # Store center position and coefficients
                center_y = chebval(center_x, cheb_coef)
                order_info = np.concatenate([[center_y], cheb_coef])
                order_tab.append(order_info)
                
            except np.linalg.LinAlgError:
                print(f"Fitting failed for cluster {i}")
                continue

    if len(order_tab) == 0:
        print("ERROR: No valid orders found")
        logging.error("No valid orders found after clustering")
        return None, None

        # if len(ord_mask[labels==i,1]) >= (ordim.shape[1]/X_half_width-2)*0.5:
        #     cheb_coef = chebfit(ord_mask[labels==i,1], ord_mask[labels==i,0], poly_order)
        #     cheb_coef = np.insert(cheb_coef, 0, chebval(center_x, cheb_coef))
        #     order_tab.append(cheb_coef)
 
    order_tab = np.asarray(order_tab) # convert list into array
    order_tab = order_tab[order_tab[:,0].argsort()]  # Sort array by ascending
    order_tab[:, 0] = np.arange(len(order_tab[:, 0]), dtype=int) # insert the number of order in the 0th column
    n_orders = int(order_tab[-1, 0]) + 1
    print(f"Found {n_orders} orders")
    logging.info(f"Found {n_orders} orders")

    
    # Recentering orders and the final tracing
    #get points for order tracing, start from center column of image
    n_points = np.floor((ordim.shape[1]/2-X_half_width)/step)
    print(f"{1+2*n_points} points in each order for fitting")
    logging.info(f"{1+2*n_points} points in each order for fitting")
    trace_x = np.arange(ordim.shape[1]//2-X_half_width - n_points*step, ordim.shape[1]-step, step, dtype=int)

    x_coord = np.arange(ordim.shape[1])
    orders = []

    for i in range(n_orders):
        print(f"Re-trace order {i}")
        logging.info(f"Re-trace order {i}")
        xfit = []
        centr = []
        width = []
        for x in trace_x:
            xt = np.arange(x-step, x+step, 1, dtype=int)
            yc = chebval(x, order_tab[i, 1:])
            if yc > 0 and yc+X_half_width+2 < ordim.shape[0]:
                yy = np.arange(yc-X_half_width, yc+X_half_width+2, 1, dtype=int)
                prof = np.median(ordim[yy[0]:yy[-1]+1, xt], axis=1)
                if prof.shape[0] != 0 and max(prof) > 15.: # Fit only well-exposured part of the order
            # Re-fit the cross-sections
                    moffat = lambda x, A, B, C, D, x0: A*(1 + ((x-x0)/B)**2)**(-C)+D
                    Y = yy-yc
                    p0 = np.array([max(prof), 3.0, 3.0, 0.0, 3.0])
                    try:
                        popt, pcov = curve_fit(moffat, Y, prof, p0, maxfev=10000,
                                               bounds = ([0,0.5,0.5,0,-10],
                                                         [np.inf,20,20,np.inf,10])
                                               )
                        fwhm = 2 * popt[1] * np.sqrt(2**(1/popt[2]) - 1)

                        if (np.isfinite(fwhm) and 1 < fwhm < 15 and
                            np.all(np.isfinite(popt)) and np.sqrt(np.diag(pcov))[4] < 2): # Check center uncertainty

                            xfit.append(x)
                            centr.append(popt[4]+yc)
                            width.append(fwhm)

                    except (RuntimeError, ValueError, TypeError):
                        continue

            if len(xfit) < 3:
                print(f"Insufficient good fits for order {i}")
                continue

            med_fwhm = np.median(width)
            print(f"Fit {len(xfit)} points, median FWHM {med_fwhm:.3f}")
            logging.info(f"Fit {len(xfit)} points, median FWHM {med_fwhm:.3f}")
                    
                    # else:
                    #     fwhm = 2*popt[1]*np.sqrt(2**(1/(popt[2]))-1)
                    #     if np.isfinite(fwhm) and fwhm > 1 and fwhm < 15:
                    #         xfit.append(x)
                    #         centr.append(popt[4]+yc)
                    #         width.append(fwhm)
                    #     med_fwhm = np.median(width)
                    # print(f"Order {i}, median FWHM: {med_fwhm:.2f}, mean FWHM: {np.mean(width):.2f}")
        # print(f"Fit {len(xfit)} points, median FWHM {med_fwhm:.3f}")
        # logging.info(f"Fit {len(xfit)} points, median FWHM {med_fwhm:.3f}")
            try:
                coef_center = chebfit(xfit, centr, poly_order)
                coef_width = chebfit(xfit, width, poly_order)
                ## Check the limits of the orders
                if adaptive:
                    width_arr = chebval(x_coord, coef_width)
                    width_arr = np.clip(width_arr, med_fwhm * 0.5, med_fwhm * 2.0) # clip unreasonable width (re-check the line)
                else:
                    width_arr = np.full(nx, med_fwhm)
                    #width = np.repeat(med_fwhm, ordim.shape[1])

                            # Check order boundaries
                order_center = chebval(x_coord, coef_center)
                order_bottom = order_center - width_arr
                order_top = order_center + width_arr

                if np.min(order_bottom) >= 1 and np.max(order_top) < ny - 1:
                    width_tab.append(width_arr)
                    orders.append(coef_center)
                else:
                    print(f"Skip incomplete order #{i}")
                    logging.info(f"Skip incomplete order #{i}")

                # if np.min(chebval(x_coord, coef_center) - width) < 1. or np.max(chebval(x_coord, coef_center) + width) >= ordim.shape[0]-1:
                #     print(f"Skip incomplete order #{i}")
                #     logging.info(f"Skip incomplete order #{i}")
                # else:
                #     width_tab.append(width)
                #     orders.append(coef_center)
            except np.linalg.LinAlgError:
                print(f"Final fitting failed for order {i}")
                continue

        print(f"Successfully traced {len(orders)} orders")
        return orders, width_tab

    # Output data
    width_tab = np.asarray(width_tab)
    orders = np.asarray(orders)
    text_file = open(dir_name.joinpath('TEMP', 'traces.txt'), "w") # Save results
    for i in range(orders.shape[0]):
        text_file.write("Order" + '\t' + str(i) + '\n')
        for j in x_coord:
            text_file.write(format('%.2f' % chebval(j, orders[i, :])) + '\t' + format('%.2f' % width[j]) + '\n')
    text_file.close()
    print(f"Data saved to {dir_name.joinpath('temp/traces.txt')}")
    logging.info(f"Data saved to {dir_name.joinpath('temp/traces.txt')}")

    # Display the results
    fig = plt.figure(figsize=(15, 15/(ordim.shape[1]/ordim.shape[0])), tight_layout=True)
    ax0 = fig.add_subplot(1,1,1)
    ax0.imshow(equalize_hist(ordim), cmap='gist_gray')
    ax0.set_xlabel("CCD X")
    ax0.set_ylabel("CCD Y")
    for i in range(orders.shape[0]):
        ax0.plot(x_coord, chebval(x_coord, orders[i, :]) + aperture * width_tab[i, :], 'b-', lw=0.4)
        ax0.plot(x_coord, chebval(x_coord, orders[i, :]) - aperture * width_tab[i, :], 'r-', lw=0.4)
        ax0.text(x_coord[15], chebval(x_coord[15], orders[i, :]), i+1, color='k', backgroundcolor='yellow', fontsize=8)
    plt.gca().invert_yaxis()
    fig.savefig(dir_name.joinpath('orders_map.pdf'), dpi=350)
    if view:
        plt.show()

    return None

def plot_cluster_maps(ordim, ord_mask, labels, n_clusters, trace_im=None, save_plots=False, output_dir=None):
    """
    Create comprehensive visualizations of detected clusters on the spectral image.
    
    Parameters:
    -----------
    ordim : numpy.ndarray
        Original image data
    ord_mask : numpy.ndarray
        Array of detected points [y, x] coordinates
    labels : numpy.ndarray
        Cluster labels from AgglomerativeClustering
    n_clusters : int
        Number of clusters
    trace_im : numpy.ndarray, optional
        Binary trace image
    save_plots : bool
        Whether to save plots to files
    output_dir : str or Path
        Directory to save plots
    """
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 12))
    
    # Plot 1: Original image with cluster overlay
    ax1 = plt.subplot(2, 3, 1)
    
    # Show original image
    vmin, vmax = np.percentile(ordim, [1, 99])
    im1 = ax1.imshow(ordim, aspect='auto', cmap='gray', vmin=vmin, vmax=vmax, 
                     origin='lower', alpha=0.8)
    
    # Overlay clusters with different colors
    colors = plt.cm.tab20(np.linspace(0, 1, n_clusters))
    
    for i in range(n_clusters):
        cluster_mask = labels == i
        cluster_points = ord_mask[cluster_mask]
        if len(cluster_points) > 0:
            ax1.scatter(cluster_points[:, 1], cluster_points[:, 0], 
                       c=[colors[i]], label=f'Order {i}', alpha=0.8, s=2)
    
    ax1.set_xlabel('X pixel (dispersion)')
    ax1.set_ylabel('Y pixel (spatial)')
    ax1.set_title('Original Image + Detected Clusters')
    plt.colorbar(im1, ax=ax1, shrink=0.8)
    
    # Plot 2: Clusters only (clean view)
    ax2 = plt.subplot(2, 3, 2)
    
    # Create empty image for cluster map
    cluster_map = np.zeros_like(ordim)
    
    for i in range(n_clusters):
        cluster_mask = labels == i
        cluster_points = ord_mask[cluster_mask]
        if len(cluster_points) > 0:
            cluster_map[cluster_points[:, 0], cluster_points[:, 1]] = i + 1
    
    # Create custom colormap
    cluster_colors = ['black'] + [plt.cm.tab20(i/n_clusters) for i in range(n_clusters)]
    custom_cmap = ListedColormap(cluster_colors)
    
    im2 = ax2.imshow(cluster_map, aspect='auto', cmap=custom_cmap, 
                     vmin=0, vmax=n_clusters, origin='lower')
    
    ax2.set_xlabel('X pixel (dispersion)')
    ax2.set_ylabel('Y pixel (spatial)')
    ax2.set_title('Cluster Map')
    
    # Custom colorbar with cluster labels
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
    cbar2.set_ticks(np.arange(n_clusters + 1))
    cbar2.set_ticklabels(['Background'] + [f'Cluster {i}' for i in range(n_clusters)])
    
    # Plot 3: Individual cluster traces
    ax3 = plt.subplot(2, 3, 3)
    
    # Plot each cluster as a separate trace
    for i in range(n_clusters):
        cluster_mask = labels == i
        cluster_points = ord_mask[cluster_mask]
        if len(cluster_points) > 0:
            # Sort by x coordinate for better line plotting
            sort_idx = np.argsort(cluster_points[:, 1])
            x_sorted = cluster_points[sort_idx, 1]
            y_sorted = cluster_points[sort_idx, 0]
            
            ax3.plot(x_sorted, y_sorted, 'o-', color=colors[i], 
                    label=f'Order {i}', markersize=3, linewidth=1, alpha=0.7)
    
    ax3.set_xlabel('X pixel (dispersion)')
    ax3.set_ylabel('Y pixel (spatial)')
    ax3.set_title('Individual Order Traces')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Cross-section at detector center
    ax4 = plt.subplot(2, 3, 4)
    
    center_x = ordim.shape[1] // 2
    x_window = 20  # Average over ±20 pixels
    
    # Extract cross-section
    cross_section = np.median(ordim[:, center_x-x_window:center_x+x_window], axis=1)
    y_coords = np.arange(len(cross_section))
    
    ax4.plot(cross_section, y_coords, 'b-', linewidth=1, label='Data')
    
    # Mark cluster positions at this x location
    for i in range(n_clusters):
        cluster_mask = labels == i
        cluster_points = ord_mask[cluster_mask]
        if len(cluster_points) > 0:
            # Find points near center_x
            x_coords = cluster_points[:, 1]
            y_coords_cluster = cluster_points[:, 0]
            
            near_center = np.abs(x_coords - center_x) < x_window * 2
            if np.any(near_center):
                y_mean = np.mean(y_coords_cluster[near_center])
                ax4.axhline(y=y_mean, color=colors[i], linestyle='--', 
                           alpha=0.8, linewidth=2, label=f'Order {i}')
    
    ax4.set_xlabel('Intensity')
    ax4.set_ylabel('Y pixel (spatial)')
    ax4.set_title(f'Cross-section at X={center_x}±{x_window}')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Cluster statistics
    ax5 = plt.subplot(2, 3, 5)
    
    cluster_stats = []
    for i in range(n_clusters):
        cluster_mask = labels == i
        cluster_points = ord_mask[cluster_mask]
        if len(cluster_points) > 0:
            stats = {
                'cluster': i,
                'n_points': len(cluster_points),
                'x_span': np.max(cluster_points[:, 1]) - np.min(cluster_points[:, 1]),
                'y_mean': np.mean(cluster_points[:, 0]),
                'y_std': np.std(cluster_points[:, 0])
            }
            cluster_stats.append(stats)
    
    # Bar plot of cluster statistics
    if cluster_stats:
        clusters = [s['cluster'] for s in cluster_stats]
        n_points = [s['n_points'] for s in cluster_stats]
        x_spans = [s['x_span'] for s in cluster_stats]
        
        ax5_twin = ax5.twinx()
        
        bars1 = ax5.bar([c - 0.2 for c in clusters], n_points, width=0.4, 
                       alpha=0.7, label='N points', color='skyblue')
        bars2 = ax5_twin.bar([c + 0.2 for c in clusters], x_spans, width=0.4, 
                            alpha=0.7, label='X span', color='lightcoral')
        
        ax5.set_xlabel('Cluster Number')
        ax5.set_ylabel('Number of Points', color='skyblue')
        ax5_twin.set_ylabel('X Span (pixels)', color='lightcoral')
        ax5.set_title('Cluster Statistics')
        ax5.set_xticks(clusters)
        
        # Add value labels on bars
        for bar, val in zip(bars1, n_points):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(n_points)*0.01,
                    f'{val}', ha='center', va='bottom', fontsize=8)
        
        for bar, val in zip(bars2, x_spans):
            ax5_twin.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(x_spans)*0.01,
                         f'{val:.0f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 6: Y position vs cluster number
    ax6 = plt.subplot(2, 3, 6)
    
    if cluster_stats:
        y_means = [s['y_mean'] for s in cluster_stats]
        y_stds = [s['y_std'] for s in cluster_stats]
        
        ax6.errorbar(clusters, y_means, yerr=y_stds, fmt='o-', 
                    capsize=5, capthick=2, linewidth=2, markersize=8)
        
        ax6.set_xlabel('Cluster Number')
        ax6.set_ylabel('Y Position (pixels)')
        ax6.set_title('Order Positions')
        ax6.grid(True, alpha=0.3)
        ax6.set_xticks(clusters)
        
        # Add value labels
        for i, (y_mean, y_std) in enumerate(zip(y_means, y_stds)):
            ax6.text(clusters[i], y_mean + y_std + max(y_means)*0.02, 
                    f'{y_mean:.1f}±{y_std:.1f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    if save_plots and output_dir:
        from pathlib import Path
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        plt.savefig(output_dir / 'cluster_analysis.png', dpi=300, bbox_inches='tight')
        print(f"Cluster analysis plot saved to {output_dir / 'cluster_analysis.png'}")
    
    plt.show()
    
    # Additional detailed plot for each cluster
    plot_individual_clusters(ordim, ord_mask, labels, n_clusters, save_plots, output_dir)

def plot_individual_clusters(ordim, ord_mask, labels, n_clusters, save_plots=False, output_dir=None):
    """
    Create detailed plots for each individual cluster.
    """
    
    # Calculate grid dimensions
    ncols = min(4, n_clusters)
    nrows = int(np.ceil(n_clusters / ncols))
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
    if nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)
    
    colors = plt.cm.tab20(np.linspace(0, 1, n_clusters))
    
    for i in range(n_clusters):
        row = i // ncols
        col = i % ncols
        ax = axes[row, col]
        
        # Show zoomed region around this cluster
        cluster_mask = labels == i
        cluster_points = ord_mask[cluster_mask]
        
        if len(cluster_points) > 0:
            # Define zoom region
            x_coords = cluster_points[:, 1]
            y_coords = cluster_points[:, 0]
            
            x_min = max(0, np.min(x_coords) - 50)
            x_max = min(ordim.shape[1], np.max(x_coords) + 50)
            y_min = max(0, np.min(y_coords) - 20)
            y_max = min(ordim.shape[0], np.max(y_coords) + 20)
            
            # Extract and show zoomed image
            zoom_img = ordim[y_min:y_max, x_min:x_max]
            vmin, vmax = np.percentile(zoom_img, [5, 95])
            
            im = ax.imshow(zoom_img, aspect='auto', cmap='gray', 
                          vmin=vmin, vmax=vmax, origin='lower',
                          extent=[x_min, x_max, y_min, y_max])
            
            # Overlay cluster points
            ax.scatter(x_coords, y_coords, c=[colors[i]], s=10, alpha=0.8, edgecolor='white', linewidth=0.5)
            
            # Fit and overlay polynomial if possible
            try:
                if len(set(x_coords)) >= 3:  # Need at least 3 unique x values
                    sort_idx = np.argsort(x_coords)
                    x_sorted = x_coords[sort_idx]
                    y_sorted = y_coords[sort_idx]
                    
                    # Simple polynomial fit for visualization
                    poly_coef = np.polyfit(x_sorted, y_sorted, min(2, len(x_sorted)-1))
                    x_fit = np.linspace(x_min, x_max, 100)
                    y_fit = np.polyval(poly_coef, x_fit)
                    
                    ax.plot(x_fit, y_fit, 'r-', linewidth=2, alpha=0.8, label='Poly fit')
            except:
                pass
            
            ax.set_xlabel('X pixel')
            ax.set_ylabel('Y pixel')
            ax.set_title(f'Cluster {i} ({len(cluster_points)} points)')
            
        else:
            ax.text(0.5, 0.5, f'Cluster {i}\nNo points', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
    
    # Hide empty subplots
    for i in range(n_clusters, nrows * ncols):
        row = i // ncols
        col = i % ncols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    
    if save_plots and output_dir:
        from pathlib import Path
        output_dir = Path(output_dir)
        plt.savefig(output_dir / 'individual_clusters.png', dpi=300, bbox_inches='tight')
        print(f"Individual clusters plot saved to {output_dir / 'individual_clusters.png'}")
    
    plt.show()

if __name__ == '__main__':
    from pathlib import Path
    
    # Directories containing the data
    temp_directory = Path('./TEMP')
    data_directory = Path('.')
    
    # Parameters for order tracing
    params = {
        'file_name': 's_flat.fits',      # Median flat file
        'X_half_width': 10,              # Half width for order detection
        'step': 50,                      # Step size for tracing
        'min_height': 10,               # Minimum peak height
        'aperture': 2.0,                 # Aperture size in FWHM units
        'adaptive': True,                # Use adaptive width
        'view': True                     # Show plots
    }
    
    # First plot the vertical averaging of the median flat
    print('Reading median flat...')
    with pyfits.open(data_directory / params['file_name']) as hdul:
        flat_data = hdul[0].data
        
        # Calculate vertical average
        vertical_avg = np.median(flat_data, axis=1)  # Using median for better noise reduction
        
        # Create plot
        plt.figure(figsize=(15, 5))
        plt.plot(np.arange(len(vertical_avg)), vertical_avg, 'k-', label='Vertical median')
        plt.xlabel('Y pixel')
        plt.ylabel('Median intensity')
        plt.title('Vertical averaging of median flat')
        plt.legend()
        
        peaks, _ = find_peaks(vertical_avg, height=params['min_height'], width=3, prominence=1.8)
        plt.plot(peaks, vertical_avg[peaks], "rx", label='Detected orders')
        plt.legend()
        
        # if params['view']:
        #     plt.show()
        # plt.close()
    
    print('Starting order detection and tracing...')
    ords, widths = order_tracer(
        dir_name=data_directory,
        **params
    )

    if ords is not None:
        print(f"Tracing successful: {len(ords)} orders found")
    else:
        print("Tracing failed")    
    
    print('Order tracing completed')
    print('Results saved in:')
    print('1. TEMP/traces.txt - order traces')
    print('2. TEMP/orders_map.pdf - visualization')
