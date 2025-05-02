from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import numpy as np
import os
import h5py
from scipy.stats import pearsonr
from nilearn.maskers import NiftiLabelsMasker
from matplotlib.gridspec import GridSpec
from nilearn import plotting
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import r2_score
import nibabel as nib
import matplotlib.pyplot as plt
import math
from nilearn.maskers import NiftiLabelsMasker
from torch.utils.data import Dataset
from collections import defaultdict

def preprocess_features(features):
    """
    Rplaces NaN values in the stimulus features with zeros, and z-score the
    features.

    Parameters
    ----------
    features : float
        Stimulus features.

    Returns
    -------
    prepr_features : float
        Preprocessed stimulus features.

    """

    ### Convert NaN values to zeros ###
    features = np.nan_to_num(features)

    ### Z-score the features ###
    scaler = StandardScaler()
    prepr_features = scaler.fit_transform(features)

    ### Output ###
    return prepr_features


def perform_pca(prepr_features, n_components):
    """
    Perform PCA on the standardized features.

    Parameters
    ----------
    prepr_features : float
        Preprocessed stimulus features.
    n_components : int
        Number of components to keep

    Returns
    -------
    features_pca : float
        PCA-downsampled stimulus features.

    """

    ### Set the number of principal components to keep ###
    # If number of PCs is larger than the number of features, set the PC number
    # to the number of features
    if n_components > prepr_features.shape[1]:
        n_components = prepr_features.shape[1]

    ### Perform PCA ###n_init=4, max_iter=300
    pca = PCA(n_components, random_state=42)
    features_pca = pca.fit_transform(prepr_features)

    ### Output ###
    return features_pca

def count_total_files(features_dir, movie_list):
    total_files = 0
    for movie in movie_list:
        if 'friends' in movie:
            season = movie.split('-')[1]
            pattern = f"{season}e"
        else:
            movie_name = movie.replace('movie10-', '')
            pattern = movie_name
            
        files = [f for f in os.listdir(features_dir + 'audio') 
                if '_features_' in f and pattern in f]
        total_files += len(files)
            
    return total_files

def get_fmri_files(subject, task_type, fmri_dir):
    subject_dir = os.path.join(fmri_dir, f'sub-{subject}/func/')
    files = [f for f in os.listdir(subject_dir) if f'sub-{subject}_task-{task_type}' in f]
    if len(files) != 1:
        raise ValueError(f"Expected 1 fMRI file for subject {subject} and task {task_type}, found {len(files)}")
    return os.path.join(subject_dir, files[0])

def load_fmri(root_data_dir, subject):
    fmri = {}

    fmri_file = f'sub-0{subject}_task-friends_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_desc-s123456_bold.h5'
    fmri_dir = os.path.join(root_data_dir, f'sub-0{subject}', 'func', fmri_file)
    fmri_friends = h5py.File(fmri_dir, 'r')
    for key, val in fmri_friends.items():
        fmri[str(key[13:])] = val[:].astype(np.float32)
    del fmri_friends

    fmri_file = f'sub-0{subject}_task-movie10_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_bold.h5'
    fmri_dir = os.path.join(root_data_dir, f'sub-0{subject}', 'func', fmri_file)
    fmri_movie10 = h5py.File(fmri_dir, 'r')
    for key, val in fmri_movie10.items():
        fmri[key[13:]] = val[:].astype(np.float32)
    del fmri_movie10

    keys_all = fmri.keys()
    figures_splits = 12
    for s in range(figures_splits):
        movie = 'figures' + format(s+1, '02')
        keys_movie = [rep for rep in keys_all if movie in rep]
        fmri[movie] = ((fmri[keys_movie[0]] + fmri[keys_movie[1]]) / 2).astype(np.float32)
        del fmri[keys_movie[0]]
        del fmri[keys_movie[1]]

    keys_all = fmri.keys()
    life_splits = 5
    for s in range(life_splits):
        movie = 'life' + format(s+1, '02')
        keys_movie = [rep for rep in keys_all if movie in rep]
        fmri[movie] = ((fmri[keys_movie[0]] + fmri[keys_movie[1]]) / 2).astype(np.float32)
        del fmri[keys_movie[0]]
        del fmri[keys_movie[1]]

    return fmri

def train_baseline_encoding(features_train, fmri_train):
    """
    Train a linear-regression-based encoding model to predict fMRI responses
    using movie features.

    Parameters
    ----------
    features_train : float
        Stimulus features for the training movies.
    fmri_train : float
        fMRI responses for the training movies.

    Returns
    -------
    model : object
        Trained regression model.

    """

    ### Train the linear regression model ###
    model = LinearRegression().fit(features_train, fmri_train)

    ### Output ###
    return model

def load_fmri(root_data_dir, subject):
    """
    Load the fMRI responses for the selected subject.

    Parameters
    ----------
    root_data_dir : str
        Root data directory.
    subject : int
        Subject used to train and validate the encoding model.

    Returns
    -------
    fmri : dict
        Dictionary containing the  fMRI responses.

    """

    fmri = {}

    ### Load the fMRI responses for Friends ###
    # Data directory
    fmri_file = f'sub-0{subject}_task-friends_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_desc-s123456_bold.h5'
    fmri_dir = os.path.join(root_data_dir, f'sub-0{subject}', 'func', fmri_file)
    # Load the the fMRI responses
    fmri_friends = h5py.File(fmri_dir, 'r')
    for key, val in fmri_friends.items():
        fmri[str(key[13:])] = val[:].astype(np.float32)
    del fmri_friends

    ### Load the fMRI responses for Movie10 ###
    # Data directory
    fmri_file = f'sub-0{subject}_task-movie10_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_bold.h5'
    fmri_dir = os.path.join(root_data_dir, f'sub-0{subject}', 'func', fmri_file)
    # Load the the fMRI responses
    fmri_movie10 = h5py.File(fmri_dir, 'r')
    for key, val in fmri_movie10.items():
        fmri[key[13:]] = val[:].astype(np.float32)
    del fmri_movie10
    # Average the fMRI responses across the two repeats for 'figures'
    keys_all = fmri.keys()
    figures_splits = 12
    for s in range(figures_splits):
        movie = 'figures' + format(s+1, '02')
        keys_movie = [rep for rep in keys_all if movie in rep]
        fmri[movie] = ((fmri[keys_movie[0]] + fmri[keys_movie[1]]) / 2).astype(np.float32)
        del fmri[keys_movie[0]]
        del fmri[keys_movie[1]]
    # Average the fMRI responses across the two repeats for 'life'
    keys_all = fmri.keys()
    life_splits = 5
    for s in range(life_splits):
        movie = 'life' + format(s+1, '02')
        keys_movie = [rep for rep in keys_all if movie in rep]
        fmri[movie] = ((fmri[keys_movie[0]] + fmri[keys_movie[1]]) / 2).astype(np.float32)
        del fmri[keys_movie[0]]
        del fmri[keys_movie[1]]

    ### Output ###
    return fmri

def align_features_and_fmri_samples(features, fmri, excluded_samples_start,
    excluded_samples_end, hrf_delay, stimulus_window, movies):
    """
    Align the stimulus feature with the fMRI response samples for the selected
    movies, later used to train and validate the encoding models.

    Parameters
    ----------
    features : dict
        Dictionary containing the stimulus features.
    fmri : dict
        Dictionary containing the fMRI responses.
    excluded_trs_start : int
        Integer indicating the first N fMRI TRs that will be excluded and not
        used for model training. The reason for excluding these TRs is that due
        to the latency of the hemodynamic response the fMRI responses of first
        few fMRI TRs do not yet contain stimulus-related information.
    excluded_trs_end : int
        Integer indicating the last N fMRI TRs that will be excluded and not
        used for model training. The reason for excluding these TRs is that
        stimulus feature samples (i.e., the stimulus chunks) can be shorter than
        the fMRI samples (i.e., the fMRI TRs), since in some cases the fMRI
        run ran longer than the actual movie. However, keep in mind that the fMRI
        timeseries onset is ALWAYS SYNCHRONIZED with movie onset (i.e., the
        first fMRI TR is always synchronized with the first stimulus chunk).
    hrf_delay : int
        fMRI detects the BOLD (Blood Oxygen Level Dependent) response, a signal
        that reflects changes in blood oxygenation levels in response to
        activity in the brain. Blood flow increases to a given brain region in
        response to its activity. This vascular response, which follows the
        hemodynamic response function (HRF), takes time. Typically, the HRF
        peaks around 5–6 seconds after a neural event: this delay reflects the
        time needed for blood oxygenation changes to propagate and for the fMRI
        signal to capture them. Therefore, this parameter introduces a delay
        between stimulus chunks and fMRI samples for a better correspondence
        between input stimuli and the brain response. For example, with a
        hrf_delay of 3, if the stimulus chunk of interest is 17, the
        corresponding fMRI sample will be 20.
    stimulus_window : int
        Integer indicating how many stimulus features' chunks are used to model
        each fMRI TR, starting from the chunk corresponding to the TR of
        interest, and going back in time. For example, with a stimulus_window of
        5, if the fMRI TR of interest is 20, it will be modeled with stimulus
        chunks [16, 17, 18, 19, 20]. Note that this only applies to visual and
        audio features, since the language features were already extracted using
        transcript words spanning several movie chunks (thus, each fMRI TR will
        only be modeled using the corresponding language feature chunk). Also
        note that a larger stimulus window will increase compute time, since it
        increases the amount of stimulus features used to train and test the
        fMRI encoding models.
    movies: list
        List of strings indicating the movies for which the fMRI responses and
        stimulus features are aligned, out of the first six seasons of Friends
        ["friends-s01", "friends-s02", "friends-s03", "friends-s04",
        "friends-s05", "friends-s06"], and the four movies from Movie10
        ["movie10-bourne", "movie10-figures", "movie10-life", "movie10-wolf"].

    Returns
    -------
    aligned_features : float
        Aligned stimulus features for the selected movies.
    aligned_fmri : float
        Aligned fMRI responses for the selected movies.

    """

    ### Empty data variables ###
    aligned_features = {"visual": [], "audio": [], "language": []}
    aligned_fmri = np.empty((0,1000), dtype=np.float32)

    ### Loop across movies ###
    for movie in movies:

        ### Get the IDs of all movies splits for the selected movie ###
        if movie[:7] == 'friends':
            id = movie[8:]
        elif movie[:7] == 'movie10':
            id = movie[8:]
        movie_splits = [key for key in fmri if id in key[:len(id)]]

        ### Loop over movie splits ###
        for split in movie_splits:

            ### Extract the fMRI ###
            fmri_split = fmri[split]
            # Exclude the first and last fMRI samples
            fmri_split = fmri_split[excluded_samples_start:-excluded_samples_end]
            aligned_fmri = np.append(aligned_fmri, fmri_split, 0)

            ### Loop over fMRI samples ###
            for s in range(len(fmri_split)):

                ### Loop across modalities ###
                for mod in features.keys():

                    ### Visual and audio features ###
                    # If visual or audio modality, model each fMRI sample using
                    # the N stimulus feature samples up to the fMRI sample of
                    # interest minus the hrf_delay (where N is defined by the
                    # 'stimulus_window' variable)
                    if mod == 'visual' or mod == 'audio':
                        # In case there are not N stimulus feature samples up to
                        # the fMRI sample of interest minus the hrf_delay (where
                        # N is defined by the 'stimulus_window' variable), model
                        # the fMRI sample using the first N stimulus feature
                        # samples
                        if s < (stimulus_window + hrf_delay):
                            idx_start = excluded_samples_start
                            idx_end = idx_start + stimulus_window
                        else:
                            idx_start = s + excluded_samples_start - hrf_delay \
                                - stimulus_window + 1
                            idx_end = idx_start + stimulus_window
                        # In case there are less visual/audio feature samples
                        # than fMRI samples minus the hrf_delay, use the last N
                        # visual/audio feature samples available (where N is
                        # defined by the 'stimulus_window' variable)
                        if idx_end > (len(features[mod][split])):
                            idx_end = len(features[mod][split])
                            idx_start = idx_end - stimulus_window
                        f = features[mod][split][idx_start:idx_end]
                        aligned_features[mod].append(f)

                    ### Language features ###
                    # Since language features already consist of embeddings
                    # spanning several samples, only model each fMRI sample
                    # using the corresponding stimulus feature sample minus the
                    # hrf_delay
                    elif mod == 'language':
                        # In case there are no language features for the fMRI
                        # sample of interest minus the hrf_delay, model the fMRI
                        # sample using the first language feature sample
                        if s < hrf_delay:
                            idx = excluded_samples_start
                        else:
                            idx = s + excluded_samples_start - hrf_delay
                        # In case there are fewer language feature samples than
                        # fMRI samples minus the hrf_delay, use the last
                        # language feature sample available
                        if idx >= (len(features[mod][split]) - hrf_delay):
                            f = features[mod][split][-1,:]
                        else:
                            f = features[mod][split][idx]
                        
                        aligned_features[mod].append(f)

                 ### Append the stimulus features of all modalities for this sample ###

    ### Convert the aligned features to a numpy array ###
    aligned_features['visual'] = np.asarray(aligned_features['visual'], dtype=np.float32)
    aligned_features['audio'] = np.asarray(aligned_features['audio'], dtype=np.float32)
    aligned_features['language'] = np.asarray(aligned_features['language'], dtype=np.float32)

    ### Output ###
    return aligned_features, aligned_fmri

def compute_encoding_accuracy(fmri_dir, fmri_val, fmri_val_pred, subject, modality):
    """
    Compare the  recorded (ground truth) and predicted fMRI responses, using a
    Pearson's correlation. The comparison is perfomed independently for each
    fMRI parcel. The correlation results are then plotted on a glass brain.

    Parameters
    ----------
    fmri_val : float
        fMRI responses for the validation movies.
    fmri_val_pred : float
        Predicted fMRI responses for the validation movies
    subject : int
        Subject number used to train and validate the encoding model.
    modality : str
        Feature modality used to train and validate the encoding model.

    """

    ### Correlate recorded and predicted fMRI responses ###
    encoding_accuracy = np.zeros((fmri_val.shape[1]), dtype=np.float32)
    for p in range(len(encoding_accuracy)):
        encoding_accuracy[p] = pearsonr(fmri_val[:, p],
            fmri_val_pred[:, p])[0]
    mean_encoding_accuracy = np.round(np.mean(encoding_accuracy), 3)

    ### Map the prediction accuracy onto a 3D brain atlas for plotting ###
    atlas_file = f'sub-0{subject}_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_desc-dseg_parcellation.nii.gz'
    atlas_path = os.path.join(fmri_dir, f'sub-0{subject}', 'atlas', atlas_file)
    atlas_masker = NiftiLabelsMasker(labels_img=atlas_path)
    atlas_masker.fit()
    encoding_accuracy_nii = atlas_masker.inverse_transform(encoding_accuracy)

    ### Plot the encoding accuracy ###
    title = f"Encoding accuracy, sub-0{subject}, modality-{modality}, mean accuracy: " + str(mean_encoding_accuracy)
    display = plotting.plot_glass_brain(
        encoding_accuracy_nii,
        display_mode="lyrz",
        cmap='hot_r',
        colorbar=True,
        plot_abs=False,
        symmetric_cbar=False,
        title=title
    )
    colorbar = display._cbar
    colorbar.set_label("Pearson's $r$", rotation=90, labelpad=12, fontsize=12)
    plotting.show()


def align_features_and_fmri_samples_friends_s7(features_friends_s7,
    root_data_dir):
    """
    Align the stimulus feature with the fMRI response samples for Friends season
    7 episodes, later used to predict the fMRI responses for challenge
    submission.

    Parameters
    ----------
    features_friends_s7 : dict
        Dictionary containing the stimulus features for Friends season 7.
    root_data_dir : str
        Root data directory.

    Returns
    -------
    aligned_features_friends_s7 : dict
        Aligned stimulus features for each subject and Friends season 7 episode.

    """

    ### Empty results dictionary ###
    aligned_features_friends_s7 = {}

    ### HRF delay ###
    # fMRI detects the BOLD (Blood Oxygen Level Dependent) response, a signal
    # that reflects changes in blood oxygenation levels in response to activity
    # in the brain. Blood flow increases to a given brain region in response to
    # its activity. This vascular response, which follows the hemodynamic
    # response function (HRF), takes time. Typically, the HRF peaks around 5–6
    # seconds after a neural event: this delay reflects the time needed for
    # blood oxygenation changes to propagate and for the fMRI signal to capture
    # them. Therefore, this parameter introduces a delay between stimulus chunks
    # and fMRI samples for a better correspondence between input stimuli and the
    # brain response. For example, with a hrf_delay of 3, if the stimulus chunk
    # of interest is 17, the corresponding fMRI sample will be 20.
    hrf_delay = 3

    ### Stimulus window ###
    # stimulus_window indicates how many stimulus feature samples are used to
    # model each fMRI sample, starting from the stimulus sample corresponding to
    # the fMRI sample of interest, minus the hrf_delay, and going back in time.
    # For example, with a stimulus_window of 5, and a hrf_delay of 3, if the
    # fMRI sample of interest is 20, it will be modeled with stimulus samples
    # [13, 14, 15, 16, 17]. Note that this only applies to visual and audio
    # features, since the language features were already extracted using
    # transcript words spanning several movie samples (thus, each fMRI sample
    # will only be modeled using the corresponding language feature sample,
    # minus the hrf_delay). Also note that a larger stimulus window will
    # increase compute time, since it increases the amount of stimulus features
    # used to train and validate the fMRI encoding models. Here you will use a
    # value of 5, since this is how the challenge baseline encoding models were
    # trained.
    stimulus_window = 5

    ### Loop over subjects ###
    subjects = [1, 2, 3, 5]
    desc = "Aligning stimulus and fMRI features of the four subjects"
    for sub in tqdm(subjects, desc=desc):
        aligned_features_friends_s7[f'sub-0{sub}'] = {}

        ### Load the Friends season 7 fMRI samples ###
        samples_dir = os.path.join(root_data_dir, f'sub-0{sub}', 'target_sample_number',
            f'sub-0{sub}_friends-s7_fmri_samples.npy')
        fmri_samples = np.load(samples_dir, allow_pickle=True).item()

        ### Loop over Friends season 7 episodes ###
        for epi, samples in fmri_samples.items():
            features_epi = {"visual": [], "audio": [], "language": []}

            ### Loop over fMRI samples ###
            for s in range(samples):
                # Empty variable containing the stimulus features of all
                # modalities for each sample
                f_all = np.empty(0)

                ### Loop across modalities ###
                for mod in features_friends_s7.keys():

                    ### Visual and audio features ###
                    # If visual or audio modality, model each fMRI sample using
                    # the N stimulus feature samples up to the fMRI sample of
                    # interest minus the hrf_delay (where N is defined by the
                    # 'stimulus_window' variable)
                    if mod == 'visual' or mod == 'audio':
                        # In case there are not N stimulus feature samples up to
                        # the fMRI sample of interest minus the hrf_delay (where
                        # N is defined by the 'stimulus_window' variable), model
                        # the fMRI sample using the first N stimulus feature
                        # samples
                        if s < (stimulus_window + hrf_delay):
                            idx_start = 0
                            idx_end = idx_start + stimulus_window
                        else:
                            idx_start = s - hrf_delay - stimulus_window + 1
                            idx_end = idx_start + stimulus_window
                        # In case there are less visual/audio feature samples
                        # than fMRI samples minus the hrf_delay, use the last N
                        # visual/audio feature samples available (where N is
                        # defined by the 'stimulus_window' variable)
                        if idx_end > len(features_friends_s7[mod][epi]):
                            idx_end = len(features_friends_s7[mod][epi])
                            idx_start = idx_end - stimulus_window
                        f = features_friends_s7[mod][epi][idx_start:idx_end]
                        # print(f.shape) 
                        # features_epi[mod] = np.array([features_epi[mod], f])
                        features_epi[mod].append(f)
                        # f_all = np.append(f_all, f.flatten())

                    ### Language features ###
                    # Since language features already consist of embeddings
                    # spanning several samples, only model each fMRI sample
                    # using the corresponding stimulus feature sample minus the
                    # hrf_delay
                    elif mod == 'language':
                        # In case there are no language features for the fMRI
                        # sample of interest minus the hrf_delay, model the fMRI
                        # sample using the first language feature sample
                        if s < hrf_delay:
                            idx = 0
                        else:
                            idx = s - hrf_delay
                        # In case there are fewer language feature samples than
                        # fMRI samples minus the hrf_delay, use the last
                        # language feature sample available
                        if idx >= (len(features_friends_s7[mod][epi]) - hrf_delay):
                            f = features_friends_s7[mod][epi][-1,:]
                        else:
                            f = features_friends_s7[mod][epi][idx]
                        # f_all = np.append(f_all, f.flatten())
                        features_epi[mod].append(f)
                        # features_epi[mod] = np.array([features_epi[mod], f])
                ### Append the stimulus features of all modalities for this sample ###
             
                # features_epi[mod] = np.asarray(features_epi[mod], dtype=np.float32)  
                
            for mod in features_epi:
                features_epi[mod] = np.asarray(features_epi[mod], dtype=np.float32)

            ### Add the episode stimulus features to the features dictionary ###
            aligned_features_friends_s7[f'sub-0{sub}'][epi] = features_epi

    return aligned_features_friends_s7


import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import nibabel as nib
from nilearn import plotting
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
import torch.nn.functional as F
import os

def calculate_metrics(pred, target, atlas_path=None, subject=None):
    """
    Calculate metrics and plot brain visualizations
    
    Args:
        pred: tensor of shape (1000,) - predicted fMRI activities
        target: tensor of shape (1000,) - target fMRI activities
        atlas_path: str - path to atlas NIfTI file (required for visualization)
        subject: int - subject number to display
    """
    # Convert to numpy for sklearn metrics
    pred_np = pred.float().detach().cpu().numpy()
    target_np = target.float().detach().cpu().numpy()
    
    # Calculate metrics
    mae = F.l1_loss(pred, target)
    mse = F.mse_loss(pred, target)
    r2 = r2_score(target_np, pred_np)
    pearson_r = pearsonr(pred_np.flatten(), target_np.flatten())[0]
    
    if atlas_path is None:
        return mae, mse, r2, pearson_r, None
            
    # Load the atlas image
    try:
        atlas_img = nib.load(atlas_path)
        atlas_data = atlas_img.get_fdata()
    except Exception as e:
        print(f"Error loading atlas image: {e}")
        return mae, mse, r2, pearson_r, None
    
    # Create a single integrated figure
    plt.figure(figsize=(20, 10))
    
    # Create nilearn plot objects but don't display them yet
    # Prepare predicted data
    pred_data = np.zeros_like(atlas_data)
    for parcel_index in range(1000):
        pred_data[atlas_data == (parcel_index + 1)] = pred_np[parcel_index]
    pred_img = nib.Nifti1Image(pred_data, affine=atlas_img.affine)
    
    # Create temporary files for the glass brain plots
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Save predicted brain image to temporary file
        pred_file = os.path.join(tmpdirname, 'pred.png')
        
        # Generate the brain plot and save it - no title
        display_pred = plotting.plot_glass_brain(
            pred_img,
            display_mode='lyrz',
            cmap='inferno',
            colorbar=True,
            plot_abs=False,
            output_file=pred_file,
            title=None  # Remove title
        )
        
        # Now create our custom figure
        fig = plt.figure(figsize=(20, 10))
        
        # Add the brain image to our main figure, taking the full width
        ax = fig.add_subplot(111)
        pred_img = plt.imread(pred_file)
        ax.imshow(pred_img)
        ax.axis('off')
        # Remove title from the axis
        
        # Add metrics text in top right with larger font
        metrics_text = (f'MAE: {mae:.3f}\n'
                       f'MSE: {mse:.3f}\n'
                       f'R²: {r2:.3f}\n'
                       f'Pearson r: {pearson_r:.3f}')
        plt.figtext(0.98, 0.98, metrics_text,
                    horizontalalignment='right',
                    verticalalignment='top',
                    fontsize=20,  # Increased font size
                    bbox=dict(facecolor='white', alpha=0.8))
        
        # Add subject number in top left with larger font
        if subject is not None:
            plt.figtext(0.02, 0.98, f'Subject {subject}',
                        horizontalalignment='left',
                        verticalalignment='top',
                        fontsize=30,  # Increased font size
                        bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave room for the text at the top
        plt.close()
    return mae, mse, r2, pearson_r, fig

class CosineLRSchedulerWithWarmup(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, max_lr, min_lr, warmup_steps, max_steps):
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        super().__init__(optimizer)
    
    def get_lr(self):
        it = self._step_count
        if it < self.warmup_steps:
            # Linear warmup
            lr_scale = it / self.warmup_steps
            return [self.max_lr * lr_scale for _ in self.optimizer.param_groups]
        
        if it > self.max_steps:
            return [self.min_lr for _ in self.optimizer.param_groups]
        
        # Cosine decay
        decay_ratio = (it - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return [self.min_lr + coeff * (self.max_lr - self.min_lr) for _ in self.optimizer.param_groups]

    def step(self, epoch=None):
        # Important: Need to call parent's step
        return super().step(epoch)
    
# def calculate_metrics(pred, target, atlas_path=None, subject=None):
#     """
#     Calculate metrics and plot brain visualizations
    
#     Args:
#         pred: tensor of shape (1000,) - predicted fMRI activities
#         target: tensor of shape (1000,) - target fMRI activities
#         atlas_path: str - path to atlas NIfTI file (required for visualization)
#         subject: int - subject number to display
#     """
#     # Convert to numpy for sklearn metrics
#     pred_np = pred.detach().cpu().numpy()
#     target_np = target.detach().cpu().numpy()
    
#     # Calculate metrics
#     mae = F.l1_loss(pred, target)
#     mse = F.mse_loss(pred, target)
#     r2 = r2_score(target_np, pred_np)
#     pearson_r = pearsonr(pred_np.flatten(), target_np.flatten())[0]
    
#     if atlas_path is None:
#         return mae, mse, r2, pearson_r, None
            
#     # Load the atlas image
#     try:
#         atlas_img = nib.load(atlas_path)
#         atlas_data = atlas_img.get_fdata()
#     except Exception as e:
#         print(f"Error loading atlas image: {e}")
#         return mae, mse, r2, pearson_r, None
    
#     # Create a single integrated figure
#     plt.figure(figsize=(20, 10))
    
#     # Create nilearn plot objects but don't display them yet
#     # Prepare predicted data
#     pred_data = np.zeros_like(atlas_data)
#     for parcel_index in range(1000):
#         pred_data[atlas_data == (parcel_index + 1)] = pred_np[parcel_index]
#     pred_img = nib.Nifti1Image(pred_data, affine=atlas_img.affine)
    
#     # Prepare target data
#     target_data = np.zeros_like(atlas_data)
#     for parcel_index in range(1000):
#         target_data[atlas_data == (parcel_index + 1)] = target_np[parcel_index]
#     target_img = nib.Nifti1Image(target_data, affine=atlas_img.affine)
    
#     # Create temporary files for the glass brain plots
#     import tempfile
#     with tempfile.TemporaryDirectory() as tmpdirname:
#         # Save predicted and target brain images to temporary files
#         pred_file = os.path.join(tmpdirname, 'pred.png')
#         target_file = os.path.join(tmpdirname, 'target.png')
        
#         # Generate the brain plots and save them
#         display_pred = plotting.plot_glass_brain(
#             pred_img,
#             display_mode='lyrz',
#             cmap='inferno',
#             colorbar=True,
#             plot_abs=False,
#             output_file=pred_file,
#             title='Predicted fMRI Activity'
#         )
        
#         display_target = plotting.plot_glass_brain(
#             target_img,
#             display_mode='lyrz',
#             cmap='inferno',
#             colorbar=True,
#             plot_abs=False,
#             output_file=target_file,
#             title='Target fMRI Activity'
#         )
        
#         # Now create our custom figure
#         fig = plt.figure(figsize=(20, 8))
        
#         # Use GridSpec for more control over the layout
#         gs = GridSpec(1, 2, figure=fig)
        
#         # Add the brain images to our main figure
#         ax1 = fig.add_subplot(gs[0, 0])
#         pred_img = plt.imread(pred_file)
#         ax1.imshow(pred_img)
#         ax1.axis('off')
#         ax1.set_title('Predicted fMRI Activity', fontsize=16)
        
#         ax2 = fig.add_subplot(gs[0, 1])
#         target_img = plt.imread(target_file)
#         ax2.imshow(target_img)
#         ax2.axis('off')
#         ax2.set_title('Target fMRI Activity', fontsize=16)
        
#         # Add metrics text in top right with larger font
#         metrics_text = (f'MAE: {mae:.3f}\n'
#                        f'MSE: {mse:.3f}\n'
#                        f'R²: {r2:.3f}\n'
#                        f'Pearson r: {pearson_r:.3f}')
#         plt.figtext(0.98, 0.98, metrics_text,
#                     horizontalalignment='right',
#                     verticalalignment='top',
#                     fontsize=16,
#                     bbox=dict(facecolor='white', alpha=0.8))
        
#         # Add subject number in top left with larger font
#         if subject is not None:
#             plt.figtext(0.02, 0.98, f'Subject {subject}',
#                         horizontalalignment='left',
#                         verticalalignment='top',
#                         fontsize=16,
#                         bbox=dict(facecolor='white', alpha=0.8))
        
#         plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave room for the text at the top
    
#     return mae, mse, r2, pearson_r, fig

class AlgonautsDataset(Dataset):
    def __init__(self, features_dir, fmri_dir, movies, subject, excluded_samples_start=5, excluded_samples_end=5, hrf_delay=3, stimulus_window=5):
        self.features_dir = features_dir
        self.fmri_dir = fmri_dir
        self.movies = movies
        self.subject = subject
        self.excluded_samples_start = excluded_samples_start
        self.excluded_samples_end = excluded_samples_end
        self.hrf_delay = hrf_delay
        self.stimulus_window = stimulus_window
        self.partition_indices = defaultdict(list)
        
        # First load all raw features
        stimuli_features = {"visual": {}, "audio": {}, "language": {}}
        # Load audio and video features first
        for movie in self.movies:
            if 'friends' in movie:
                season = movie.split('-')[1]
                dir_list = sorted(os.listdir(self.features_dir + 'audio')) #List of all audio for each subset of dataset
                for episode in dir_list:
                    if f"{season}e" in episode and '_features_' in episode:
                        episode_base = episode.split('_features_')[0] # friends_s01e01 and so on....
                        
                        for modality in ['audio', 'visual']:
                            with h5py.File(os.path.join(self.features_dir, modality, f"{episode_base}_features_{modality}.h5"), 'r') as f:
                                try:
                                    stimuli_features[modality][episode_base.split('_')[1]] = f[episode_base.split('_')[1]][modality][:]
                                except:
                                    f.visit(lambda x: print(x))
                lang_dir_list = sorted(os.listdir(self.features_dir + 'language'))
                for episode in lang_dir_list:
                    if f"{season}e" in episode and '_features_' in episode:
                        episode_base = episode.split('_features_')[0]
                        
                        with h5py.File(os.path.join(self.features_dir, 'language', f"{episode_base}_features_language.h5"), 'r') as f:
                            try:
                                st_season_episode = episode_base.split('_')[1]
                                stimuli_features['language'][st_season_episode] = f[st_season_episode]['language_pooler_output'][:]
                            except:
                                f.visit(lambda x: print(x))
            else:
                movie_name = movie.replace('movie10-', '')
                partitions = sorted([f for f in os.listdir(self.features_dir + 'audio') if movie_name in f and '_features_' in f])
                
                for partition in partitions:
                    partition_base = partition.split('_features_')[0]
                    
                    for modality in ['audio', 'visual']:
                        with h5py.File(os.path.join(self.features_dir, modality, f"{partition_base}_features_{modality}.h5"), 'r') as f:
                            try:
                                stimuli_features[modality][partition_base] = f[partition_base][modality][:]
                            except:
                                f.visit(lambda x: print(x))
                lang_partitions = sorted([f for f in os.listdir(self.features_dir + 'language') if movie_name in f and '_features_' in f])
                
                for partition in lang_partitions:
                    partition_base = partition.split('_features_')[0]
                    
                    with h5py.File(os.path.join(self.features_dir, 'language', f"{partition_base}_features_language.h5"), 'r') as f:
                        try:
                            stimuli_features['language'][partition_base] = f[partition_base]['language_pooler_output'][:]
                        except:
                            f.visit(lambda x: print(x))

        fmri_data = load_fmri(self.fmri_dir, self.subject)
        self.raw_stimuli = stimuli_features

        self.aligned_features, self.aligned_fmri = align_features_and_fmri_samples(
            stimuli_features, 
            fmri_data, 
            self.excluded_samples_start, 
            self.excluded_samples_end, 
            self.hrf_delay, 
            self.stimulus_window, 
            self.movies
        )

    def __len__(self):
        return self.aligned_features['audio'].shape[0]

    def __getitem__(self, idx):
        return {
            'audio': self.aligned_features['audio'][idx],
            'video': self.aligned_features['visual'][idx],
            'language': self.aligned_features['language'][idx],
            'fmri': self.aligned_fmri[idx]
        }
    
    def get_raw_stimuli(self):
        return self.raw_stimuli


def normalize(data):
    assert data.shape[1] == 1000, "Data does not have 1000 parcels"
    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0)

    # Avoid division by zero
    stds = np.where(stds == 0, 1, stds)

    # Standardize
    normalized_data = (data - means) / stds

    # Verify
    new_means = np.mean(normalized_data, axis=0)
    new_stds = np.std(normalized_data, axis=0)
    max_abs_mean = np.max(np.abs(new_means))
    max_std_diff = np.max(np.abs(new_stds - 1))
    # Check for problematic parcels
    if max_std_diff > 1e-5 or max_abs_mean > 1e-5:
        print("  Warning: Normalization not perfect. Check data.")
    else:
        return normalized_data
    
def normalize_across_episodes(fmri_data):
    # Step 1: Collect all episode data into a single array
    data_keys = list(fmri_data.keys())
    all_data = np.concatenate([fmri_data[key] for key in data_keys], axis=0)  # Shape: (total_clips, 1000)

    # Step 2: Compute global mean and std across all clips
    global_means = np.mean(all_data, axis=0)  # Mean for each parcel
    global_stds = np.std(all_data, axis=0)   # Std for each parcel
    global_stds = np.where(global_stds == 0, 1, global_stds)  # Avoid division by zero

    # Step 3: Apply global normalization to each episode
    for key in data_keys:
        fmri_data[key] = (fmri_data[key] - global_means) / global_stds

    # Step 4: Verify normalization (optional, for debugging)
    all_normalized_data = np.concatenate([fmri_data[key] for key in data_keys], axis=0)
    new_means = np.mean(all_normalized_data, axis=0)
    new_stds = np.std(all_normalized_data, axis=0)
    max_abs_mean = np.max(np.abs(new_means))
    max_std_diff = np.max(np.abs(new_stds - 1))
    print(f"Global normalization stats:")
    print(f"Max absolute mean: {max_abs_mean}")
    print(f"Max std difference: {max_std_diff}")
    if max_std_diff > 1e-5 or max_abs_mean > 1e-5:
        print("Warning: Global normalization not perfect. Check data.")
    return fmri_data


def check_fmri_centering(fmri_data: np.ndarray, tolerance: float = 1e-6):
    """
    Checks if the fMRI data is centered (mean close to zero) for each parcel.

    Args:
        fmri_data (np.ndarray): The fMRI data array with shape (n_samples, n_parcels).
                                Assumes samples are along axis 0 and parcels along axis 1.
        tolerance (float): The absolute tolerance for checking if the mean is close to zero.
                           Defaults to 1e-6.

    Returns:
        bool: True if the data is centered for all parcels within the tolerance, False otherwise.
    """
    if fmri_data.ndim != 2:
        raise ValueError(f"Expected fmri_data to be 2D (n_samples, n_parcels), but got shape {fmri_data.shape}")

    n_samples, n_parcels = fmri_data.shape
    print(f"Checking centering for fMRI data with shape: {fmri_data.shape}")

    # Calculate the mean for each parcel across all samples
    parcel_means = np.mean(fmri_data, axis=0) # Shape: (n_parcels,)

    # Check if all parcel means are close to zero
    # np.allclose checks if two arrays are element-wise equal within a given tolerance.
    # We compare parcel_means to an array of zeros of the same shape.
    is_centered = np.allclose(parcel_means, np.zeros(n_parcels), atol=tolerance)

    # --- Reporting ---
    if is_centered:
        print(f"\nData IS centered.")
        print(f"All {n_parcels} parcel means are within +/-{tolerance} of zero.")
    else:
        print(f"\nData IS NOT centered.")
        # Find parcels that are not centered
        non_centered_indices = np.where(np.abs(parcel_means) > tolerance)[0]
        num_non_centered = len(non_centered_indices)
        print(f"{num_non_centered} out of {n_parcels} parcels have means outside the tolerance +/-{tolerance}.")

        # Show range of means for context
        min_mean = np.min(parcel_means)
        max_mean = np.max(parcel_means)
        print(f"Overall range of parcel means: [{min_mean:.4g}, {max_mean:.4g}]")

        # Optionally, list some non-centered means
        if num_non_centered < 10:
             print("Means of non-centered parcels:")
             for idx in non_centered_indices:
                 print(f"  Parcel {idx}: {parcel_means[idx]:.4g}")
        else:
             print(f"Means of first 5 non-centered parcels:")
             for i in range(5):
                  idx = non_centered_indices[i]
                  print(f"  Parcel {idx}: {parcel_means[idx]:.4g}")

    return is_centered