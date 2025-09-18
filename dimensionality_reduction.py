#!/usr/bin/env python3
"""
PCA for EEG Data with Complete Feature Configuration
Implements Principal Component Analysis with all specified features that can be easily commented out.
"""

import numpy as np

# Feature Configuration 
FEATURE_TYPES = [
    'band_power',           # Band power for each frequency band
    'relative_power',       # Relative power for each frequency band
    'power_ratios',         # Power ratios between bands
    'spectral_entropy',     # Spectral entropy per channel
    'peak_frequency'        # Peak frequency per channel
]

FEATURE_CHANNELS = [
    'Fz',    # Frontal midline
    'Cz',    # Central midline
    'Pz',    # Parietal midline
    'C3',    # Left central
    'C4',    # Right central
    'F3',    # Left frontal
    'F4',    # Right frontal
    'P3',    # Left parietal
    'P4',    # Right parietal
    'O1',    # Left occipital
    'O2'     # Right occipital
]

FREQ_BANDS = [

    'alpha',    # 8-12 Hz
    'beta',     # 12-30 Hz
    'gamma'     # 30-50 Hz

    # will be removed
    'delta',    # 0.5-4 Hz 
    'theta',    # 4-8 Hz 
    
]

# Power ratio combinations 
POWER_RATIO_COMBINATIONS = [
    ('alpha', 'beta'),      # Alpha/Beta ratio 
    ('beta', 'gamma'),      # Beta/Gamma ratio
    ('gamma', 'alpha'),     # Gamma/Alpha ratio

    # will be commented below
    ('gamma', 'theta'),     # Gamma/Theta ratio
    ('theta', 'alpha'),     # Theta/Alpha ratio
    ('alpha', 'theta'),     # Alpha/Theta ratio
    ('theta', 'beta'),      # Theta/Beta ratio
    
    ('delta', 'theta'),     # Delta/Theta ratio
    ('delta', 'alpha'),     # Delta/Alpha ratio
    ('delta', 'beta'),      # Delta/Beta ratio

]

class SimplePCA:
    """
    Simple PCA implementation without sklearn dependency.
    """
    
    def __init__(self, n_components=4):
        """
        Initialize PCA.
        
        Parameters:
        -----------
        n_components : int
            Number of principal components
        """
        self.n_components = n_components
        self.components_ = None
        self.mean_ = None
        self.explained_variance_ = None
        
    def fit(self, X):
        """
        Fit PCA to data.
        
        Parameters:
        -----------
        X : ndarray, shape (n_samples, n_features)
            Input data
            
        Returns:
        --------
        self : object
        """
        # Center the data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        # Compute covariance matrix
        cov_matrix = np.cov(X_centered.T)
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort by eigenvalues (descending order)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Select top components
        self.components_ = eigenvectors[:, :self.n_components].T
        self.explained_variance_ = eigenvalues[:self.n_components]
        
        return self
        
    def transform(self, X):
        """
        Transform data using fitted PCA.
        
        Parameters:
        -----------
        X : ndarray, shape (n_samples, n_features)
            Input data
            
        Returns:
        --------
        X_transformed : ndarray, shape (n_samples, n_components)
            Transformed data
        """
        if self.components_ is None:
            raise ValueError("PCA must be fitted before transform")
            
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_.T)
    
    def fit_transform(self, X):
        """
        Fit PCA and transform data in one step.
        
        Parameters:
        -----------
        X : ndarray, shape (n_samples, n_features)
            Input data
            
        Returns:
        --------
        X_transformed : ndarray, shape (n_samples, n_components)
            Transformed data
        """
        return self.fit(X).transform(X)

class EEGDimensionalityReducer:
    """
    EEG dimensionality reduction using PCA with complete feature configuration.
    """
    
    def __init__(self, n_components=4, feature_types=None, feature_channels=None, 
                 freq_bands=None, power_ratio_combinations=None):
        """
        Initialize PCA-based dimensionality reducer with feature configuration.
        
        Parameters:
        -----------
        n_components : int
            Number of components
        feature_types : list or None
            List of feature types to use (defaults to FEATURE_TYPES)
        feature_channels : list or None
            List of channels to use (defaults to FEATURE_CHANNELS)
        freq_bands : list or None
            List of frequency bands to use (defaults to FREQ_BANDS)
        power_ratio_combinations : list or None
            List of power ratio combinations (defaults to POWER_RATIO_COMBINATIONS)
        """
        self.n_components = n_components
        self.feature_types = feature_types if feature_types is not None else FEATURE_TYPES
        self.feature_channels = feature_channels if feature_channels is not None else FEATURE_CHANNELS
        self.freq_bands = freq_bands if freq_bands is not None else FREQ_BANDS
        self.power_ratio_combinations = power_ratio_combinations if power_ratio_combinations is not None else POWER_RATIO_COMBINATIONS
        self.pca_ = None
        self.feature_names_ = None
        
        # Generate feature names based on configuration
        self._generate_feature_names()
        
    def _generate_feature_names(self):
        """
        Generate feature names based on configuration.
        """
        self.feature_names_ = []
        
        for feature_type in self.feature_types:
            if feature_type == 'band_power':
                # Band power: one feature per channel per frequency band
                for channel in self.feature_channels:
                    for band in self.freq_bands:
                        self.feature_names_.append(f"band_power_{channel}_{band}")
                        
            elif feature_type == 'relative_power':
                # Relative power: one feature per channel per frequency band
                for channel in self.feature_channels:
                    for band in self.freq_bands:
                        self.feature_names_.append(f"relative_power_{channel}_{band}")
                        
            elif feature_type == 'power_ratios':
                # Power ratios: one feature per channel per ratio combination
                for channel in self.feature_channels:
                    for band1, band2 in self.power_ratio_combinations:
                        self.feature_names_.append(f"power_ratio_{channel}_{band1}_{band2}")
                        
            elif feature_type == 'spectral_entropy':
                # Spectral entropy: one feature per channel
                for channel in self.feature_channels:
                    self.feature_names_.append(f"spectral_entropy_{channel}")
                    
            elif feature_type == 'peak_frequency':
                # Peak frequency: one feature per channel
                for channel in self.feature_channels:
                    self.feature_names_.append(f"peak_frequency_{channel}")
    
    def get_expected_feature_count(self):
        """
        Get the expected number of features based on configuration.
        
        Returns:
        --------
        int : Expected number of features
        """
        return len(self.feature_names_)
    
    def get_feature_names(self):
        """
        Get the list of feature names.
        
        Returns:
        --------
        list : List of feature names
        """
        return self.feature_names_.copy()
    
    def get_feature_breakdown(self):
        """
        Get a breakdown of features by type.
        
        Returns:
        --------
        dict : Dictionary with feature counts by type
        """
        breakdown = {}
        for feature_type in self.feature_types:
            count = len([name for name in self.feature_names_ if name.startswith(feature_type)])
            breakdown[feature_type] = count
        return breakdown
    
    def validate_input_features(self, X):
        """
        Validate that input data matches expected feature configuration.
        
        Parameters:
        -----------
        X : ndarray
            Input data
            
        Returns:
        --------
        bool : True if valid, raises ValueError if not
        """
        expected_features = self.get_expected_feature_count()
        
        if len(X.shape) == 2:
            actual_features = X.shape[1]
        elif len(X.shape) == 3:
            # For 3D data, assume it will be reshaped to 2D
            actual_features = X.shape[1] * X.shape[2]
        else:
            raise ValueError("Input data must be 2D or 3D")
        
        if actual_features != expected_features:
            print(f"Warning: Expected {expected_features} features based on configuration, "
                  f"but got {actual_features} features in input data.")
            self.print_configuration()
        
        return True
        
    def fit(self, X):
        """
        Fit the PCA reducer.
        
        Parameters:
        -----------
        X : ndarray
            Input data (2D or 3D)
            
        Returns:
        --------
        self : object
        """
        # Validate input features
        self.validate_input_features(X)
        
        self.pca_ = SimplePCA(n_components=self.n_components)
        
        # For 3D data (epochs, channels, times), reshape to 2D
        if len(X.shape) == 3:
            n_epochs, n_channels, n_times = X.shape
            X = X.reshape(n_epochs, n_channels * n_times)
            
        self.pca_.fit(X)
        return self
        
    def transform(self, X):
        """
        Transform data using the fitted PCA.
        
        Parameters:
        -----------
        X : ndarray
            Input data (2D or 3D)
            
        Returns:
        --------
        X_transformed : ndarray
            Transformed data
        """
        if self.pca_ is None:
            raise ValueError("Reducer must be fitted before transform")
            
        # Validate input features
        self.validate_input_features(X)
        
        # For 3D data, reshape to 2D
        if len(X.shape) == 3:
            n_epochs, n_channels, n_times = X.shape
            X = X.reshape(n_epochs, n_channels * n_times)
            
        return self.pca_.transform(X)
    
    def fit_transform(self, X):
        """
        Fit PCA and transform data in one step.
        
        Parameters:
        -----------
        X : ndarray
            Input data (2D or 3D)
            
        Returns:
        --------
        X_transformed : ndarray
            Transformed data
        """
        return self.fit(X).transform(X)
    
    def print_configuration(self):
        """
        Print the current feature configuration.
        """
        print("EEG Feature Configuration:")
        print(f"  Feature types ({len(self.feature_types)}): {self.feature_types}")
        print(f"  Channels ({len(self.feature_channels)}): {self.feature_channels}")
        print(f"  Frequency bands ({len(self.freq_bands)}): {self.freq_bands}")
        print(f"  Power ratio combinations ({len(self.power_ratio_combinations)}): {self.power_ratio_combinations}")
        print(f"  PCA components: {self.n_components}")
        print()
        
        breakdown = self.get_feature_breakdown()
        print("Feature breakdown:")
        total_features = 0
        for feature_type, count in breakdown.items():
            print(f"  {feature_type}: {count} features")
            total_features += count
        print(f"  Total: {total_features} features")
        
    def print_all_feature_names(self):
        """
        Print all generated feature names (useful for debugging).
        """
        print(f"All {len(self.feature_names_)} feature names:")
        for i, name in enumerate(self.feature_names_, 1):
            print(f"  {i:3d}. {name}")

# Example usage functions
def create_full_reducer():
    """
    Create a reducer with all features enabled.
    """
    reducer = EEGDimensionalityReducer(n_components=20)
    reducer.print_configuration()
    return reducer

def create_custom_reducer():
    """
    Example of creating a reducer with custom configuration.
    """
    # Comment out features you don't want
    custom_feature_types = [
        'band_power',
        # 'relative_power',    
        'power_ratios',
        'spectral_entropy',
        # 'peak_frequency' 
    ]
    
    # Comment out channels you don't want
    custom_channels = [
        'Fz',
        'Cz',
        'Pz',
        'C3',
        'C4',
        'F3',
        'F4',
        'P3',
        'P4',
        'O1',
        'O2' 
    ]
    
    # Comment out frequency bands you don't want
    custom_freq_bands = [
        'alpha',
        'beta',
        'gamma'
        # 'theta',
        # 'delta',        
    ]
    
    # Comment out power ratios you don't want
    custom_power_ratios = [
        ('alpha', 'beta'),      # Alpha/Beta ratio 
        ('beta', 'gamma'),      # Beta/Gamma ratio
        ('gamma', 'alpha'),     # Gamma/Alpha ratio

        # ('gamma', 'theta')      # Gamma/Theta ratio
        # ('theta', 'alpha'),     # Theta/Alpha ratio
        # ('alpha', 'theta'),     # Alpha/Theta ratio
        # ('theta', 'beta'),      # Theta/Beta ratio
        
        # ('delta', 'theta'),     # Delta/Theta ratio
        # ('delta', 'alpha'),     # Delta/Alpha ratio
        # ('delta', 'beta'),      # Delta/Beta ratio
    ]
    
    reducer = EEGDimensionalityReducer(
        n_components=10,
        feature_types=custom_feature_types,
        feature_channels=custom_channels,
        freq_bands=custom_freq_bands,
        power_ratio_combinations=custom_power_ratios
    )
    
    reducer.print_configuration()
    return reducer

