import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from cartopy import crs as ccrs
from cartopy import feature as cfeature


def plot_confusion_matrix(y_true, y_pred, classes, model_name, normalize=False, title=None, cmap=plt.cm.Blues, filename=None):
    """
    Function to plot a confusion matrix. 
    """
    if not title:
        if normalize:
            title = 'Confusion Matrix (normalized)'
        else:
            title = 'Confusion Matrix'

    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(5,5))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax, shrink=0.80)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]))
    ax.set_xticklabels(classes, fontsize=10)
    ax.set_yticklabels(classes, fontsize=10)
    ax.set_title(title, fontsize=12)
    ax.set_ylabel('True label', fontsize=10)
    ax.set_xlabel('Predicted label', fontsize=10)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                   fontsize=10)
            
    if filename:
        path = f'/glade/u/home/jwillson/winter-ptype/images/{model_name}/'
        plt.savefig(path + filename, dpi=300, bbox_inches="tight")
        
    return ax


def conus_plot(df, 
               dataset = "mping", 
               column = "pred_label", 
               title = "Predicted", 
               save_path = False):
    
    lat_n = 54.0
    lat_s = 20.0
    lon_w = -63.0
    lon_e = -125.0
    c_lat = (lat_n + lat_s)/2
    c_lon = (lon_w + lon_e )/2
    colors = {0:'lime', 1:'dodgerblue', 2:'red', 3:'black'}
    scale = 10
    proj = ccrs.LambertConformal(central_longitude=c_lon, central_latitude=c_lat)
    res = '50m'  # Coarsest and quickest to display; other options are '10m' (slowest) and '50m'.
    fig = plt.figure(figsize=(18, 12))
    ax = plt.subplot(1, 1, 1, projection=proj)
    ax.set_extent([lon_w, lon_e, lat_s, lat_n])
    ax.add_feature(cfeature.LAND.with_scale(res))
    ax.add_feature(cfeature.OCEAN.with_scale(res))
    ax.add_feature(cfeature.COASTLINE.with_scale(res))
    ax.add_feature(cfeature.LAKES.with_scale(res), alpha=0.5)
    ax.add_feature(cfeature.STATES.with_scale(res))

    zorder = [1,2,4,3]
    if dataset == 'ASOS':
        df['rand_lon'] = [df['lon'].to_numpy()[i]+np.random.normal(scale=scale) for i in range(len(df['lon']))]
        df['rand_lat'] = [df['lat'].to_numpy()[i]+np.random.normal(scale=scale) for i in range(len(df['lat']))]
        for i in range(4):
            ax.scatter(df["rand_lon"][df[column] == i]-360,
                       df["rand_lat"][df[column] == i],
                       c=df["true_label"][df[column] == i].map(colors),
                       s=3, transform=ccrs.PlateCarree(), zorder=zorder[i], alpha = 0.2)
    else:
        for i in range(4):
            ax.scatter(df["lon"][df[column] == i]-360,
                       df["lat"][df[column] == i],
                       c=df[column][df[column] == i].map(colors),
                       s=60, transform=ccrs.PlateCarree(), zorder=zorder[i], alpha = 0.2)

    first_day = str(min(df['datetime'])).split(' ')[0]
    last_day = str(max(df['datetime'])).split(' ')[0]
    plt.legend(colors.values(), labels=["Rain", "Snow", "Ice Pellets", "Freezing Rain"], fontsize=24, markerscale=3, loc="lower right")
    plt.title(f"{dataset} {first_day} to {last_day} {title} Labels", fontsize=30)
    if save_path is not False:
        fn = os.path.join(save_path, f'{first_day}_{last_day}_truelabels.png')
        plt.savefig(fn, dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.show()
