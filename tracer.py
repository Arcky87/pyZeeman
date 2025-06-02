import astropy.io.fits as pyfits
import numpy as np

import logging

from numpy.polynomial.chebyshev import chebfit, chebval

from scipy.optimize import curve_fit
from scipy.signal import find_peaks

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

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
                                        linkage='ward',
                                        compute_full_tree=False)#distance_threshold=None
        
        model.fit(ord_mask)
        labels = model.labels_
    
    except Exception as e:
        print(f"Clustering failed: {e}")
        return None, None
    
    n_clusters = len(np.unique(labels))
    print(f"Agglomerative clustering created {n_clusters} clusters")

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
        min_x_span = nx * 0.7  # Require order to span at least 70% of detector


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

                        if (np.isinfinite(fwhm) and 1 < fwhm < 15 and
                            np.all(np.isinfinite(popt)) and np.sqrt(np.diag(pcov))[4] < 2): # Check center uncertainty

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
                    width = chebval(x_coord, coef_width)
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
            except np.Linalg.LinAlgError:
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

if __name__ == '__main__':
    from pathlib import Path
    
    # Directories containing the data
    temp_directory = Path('./TEMP')
    data_directory = Path('.')
    
    # Parameters for order tracing
    params = {
        'file_name': 's_flat.fits',      # Median flat file
        'X_half_width': 15,              # Half width for order detection
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
