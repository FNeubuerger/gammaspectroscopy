import re
from scipy.signal import find_peaks
import pandas as pd
import numpy as np
from uncertainties import ufloat, unumpy
from scipy.optimize import curve_fit, OptimizeWarning
import warnings
from scipy.stats import norm
import matplotlib.pyplot as plt


def parse_polynomial(line):
    # Replace ',' with '.' and split by '+' or '-'
    line = line.replace(',', '.')
    line = line.replace('channel', 'x')
    line = line.replace('E', 'e')
    
    terms = re.split(r'(?=[+-,E,*])', line)

    polynomial = lambda x: float(terms[0]) + float(terms[1]) * x + float(terms[3]) * x**2 + float(terms[5]+terms[6])* x**3
    return polynomial

def fit_gaussian(data: pd.DataFrame, peaks: np.ndarray, properties: dict, polynomial: callable) -> np.ndarray:
    """
    Fits Gaussian functions to the identified peaks in the data and applies the polynomial to the fitted peaks.

    Parameters:
    -----------
    data : pd.DataFrame
        The processed data containing the counts.
    peaks : np.ndarray
        The indices of the identified peaks.
    widths : np.ndarray
        The widths of the identified peaks.
    polynomial : callable
        The polynomial function to be applied to the fitted peaks.

    Returns:
    --------
    np.ndarray
        The polynomial values at the fitted peaks with uncertainties.

    Notes:
    ------
    - The function uses the `curve_fit` method from `scipy.optimize` to fit Gaussian functions to the peaks.
    - The fitted peaks are represented as `uncertainties` arrays with mean and standard deviation.
    """


    warnings.filterwarnings("ignore", category=OptimizeWarning)
    def gaussian(x, a, x0, sigma):
        return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))
    fitted_peak_mean = []
    fitted_peak_std = []

    for peak, left_base, right_base in zip(peaks, properties['left_bases'], properties['right_bases']):
        try:
            x = np.arange(left_base, right_base)
            # fallback to +-5 if the range is too small to fit or too large to be realistic
            if len(x) <= 10:
                x = np.arange(peak-5, peak+5)
            elif len(x) > 30:
                x = np.arange(peak-5, peak+5)  
            y = data['counts'].iloc[x]
            popt, _ = curve_fit(gaussian, x, y, p0=[y.max(), x.mean(), 1], maxfev = 2000)
            fitted_peak_mean.append(np.abs(popt[1]))
            fitted_peak_std.append(np.abs(popt[2]))
        except RuntimeError:
            fitted_peak_mean.append(0)
            fitted_peak_std.append(1)
    fitted_peaks = unumpy.uarray(fitted_peak_mean, fitted_peak_std)
    return polynomial(fitted_peaks)

def identify_background(data: pd.DataFrame, wndw: int = 5, order: int = 3, scale: float = 1.5) -> np.ndarray:
    """
    Identify the background of a spectrum by analyzing the slopes and applying a moving average.
    Parameters:
    -----------
    data : pd.DataFrame
        The processed data containing the counts.
    wndw : int, optional
        The window size for the moving average (default is 5).
    order : int, optional
        The order of the moving average (default is 3).
    scale : float, optional
        The scale factor to determine the background threshold (default is 1.5).
    Returns:
    --------
    np.ndarray
        The interpolated background values.
    """
    counts = data['counts'].values
    slopes = np.abs(np.diff(counts))
    moving_avg = np.convolve(slopes, np.ones(wndw) / wndw, mode='same')
    threshold = np.mean(moving_avg) * scale
    background_mask = moving_avg < threshold
    background_mask = np.append(background_mask, True)  # Ensure the mask has the same length as counts
    background = np.interp(np.arange(len(counts)), np.arange(len(counts))[background_mask], counts[background_mask])
    return background

def get_isotopes_df()-> pd.DataFrame:
    """
    Create a DataFrame with gamma spectrum data for various isotopes.
    The DataFrame contains the following columns:
    - "Isotope": The name of the isotope.
    - "Energy (keV)": The energy of the gamma emission in kilo-electron volts (keV).
    - "Intensity (% Activity)": The intensity of the gamma emission as a percentage of the isotope's activity.
    - "Notes": Additional notes about the isotope or its gamma emissions.
    Returns:
        pd.DataFrame: A DataFrame containing the gamma spectrum data for various isotopes.
    """

    # Create a DataFrame with gamma spectrum data
    gamma_spectrum_data = {
                    "Isotope": [
                        "Cs-137",
                        "Co-60",
                        "I-131",
                        "Tc-99m",
                        "Ra-226",
                        "Th-232",
                        "U-238",
                        "K-40",
                        "Am-241",
                        "Na-22",
                        "Eu-152",
                        "Eu-154",
                    ],
                    "Energies (keV)": [
                        [661.7],
                        [1173.2, 1332.5],
                        [364.5],
                        [140.5],
                        [186.2, 609.3, 1120.3, 1764.5],
                        [338.3, 911.2, 2614.5],
                        [234.0, 1001.0],
                        [1460.8],
                        [59.5],
                        [511.0, 1274.5],
                        [121.8, 344.3, 1408.0],
                        [123.1, 1274.4,  723.32, 1004.8, 873.18, 996.29]
                    ],
                    "Notes": [
                        "Decay product: Ba-137m",
                        "Dual gamma peaks for decay",
                        "Used in medical diagnostics",
                        "Widely used in nuclear medicine",
                        "Includes a complex decay series",
                        "Part of the decay chain",
                        "Decay includes radon progeny",
                        "Natural isotope in potassium",
                        "Common in smoke detectors",
                        "Positron annihilation peak",
                        "Multi-line emitter",
                        "Multi-line emitter"
                    ]
    }

    return pd.DataFrame(gamma_spectrum_data)


def plot_spectrum_from_file(filename: str, bg: np.ndarray = np.array([])):
    """Plot a gamma spectrum from a data file."""
    data = pd.read_table(filename, header=3)
    data['energy in keV'] = data['energy in keV'].str.replace(',', '.').astype(float)

    plt.figure(figsize=(10, 6))
    plt.plot(data['energy in keV'], data['counts'], label=filename)
    if len(bg)!=0:
        plt.plot(data['energy in keV'], bg, label='Background', color='r')
    plt.xlabel('Energy (keV)')
    plt.ylabel('Counts')
    plt.title('Spectrum')
    plt.legend()
    plt.show()

def plot_spectrum(df: pd.DataFrame, semilogy=False):
    """Plot a gamma spectrum from a DataFrame."""
    df['filename'] = df['filename'].replace('\\', '/')
    data =  pd.read_table(df['filename'], header=3)
    data['energy in keV'] = data['energy in keV'].str.replace(',', '.').astype(float)
    print(data.shape)
    data.plot(x='energy in keV', y='counts', kind='line', figsize=(10, 6))
    plt.vlines(df['fitted_peaks_mean'], 0, max(data['counts']), color='r', linestyles='dashed', label='Fitted peaks')
    for i, (identified_peak, isotope) in enumerate(zip(df['identified_peaks'],df['identified_isotopes'])):
        colors = plt.cm.hsv(np.linspace(0.2, 0.8, len(df['identified_peaks'])))
        plt.vlines(identified_peak, 0,  max(data['counts']), color=colors[i], label=f'identified peaks of {isotope}')
        
    if semilogy:
        plt.semilogy('log')
    plt.title(f'Spectrum: {df["filename"]}')
    plt.legend()
    plt.xlabel('Energy (keV)')
    plt.ylabel('Counts')

def calculate_confidence(peak, energy, std):
    """Calculate confidence score for peak matching an energy value."""
    # Use a Gaussian probability density function
    confidence = norm.pdf(energy, loc=peak, scale=std)
    # Normalize to [0,1] range
    confidence = confidence / norm.pdf(peak, loc=peak, scale=std)
    return confidence

def identify_isotopes(fitted_peaks: unumpy.uarray, tolerance: float = 0.5, matching_ratio: float=1/3, verbose=False) -> list:
    """
        Identify isotopes based on fitted peak energies.
        This function compares the provided fitted peak energies with known isotope energies
        and identifies potential isotopes based on a confidence threshold and matching ratio.
        Parameters:
        -----------
        fitted_peaks : unumpy.uarray
            Array of fitted peak energies with uncertainties.
        tolerance : float, optional
            Confidence threshold for matching peaks to isotope energies (default is 0.5).
        matching_ratio : float, optional
            Minimum ratio of matched peaks to isotope energies required to identify an isotope (default is 0.5).
        verbose : bool, optional
            If True, prints detailed matching information (default is False).
        **kwargs : dict
            Additional keyword arguments (not used in this function).
        Returns:
        --------
        identified_isotopes : list
            List of identified isotopes.
        confidences : list
            List of confidence values for each matched peak.
        percentage_matched : list
            List of percentages of matched peaks for each identified isotope.
    """

    known_isotopes = get_isotopes_df()

    identified_isotopes = []   
    peak_confidences = []
    isotope_confidences = []
    percentage_matched = []
    identified_peaks = []
    
    for peak, std in zip(unumpy.nominal_values(fitted_peaks), unumpy.std_devs(fitted_peaks)):
        for index, row in known_isotopes.iterrows():
            matched = []
            for energy in row['Energies (keV)']:
                peak_confidence = calculate_confidence(peak, energy, std)
                if peak_confidence > tolerance:
                    matched.append(row['Isotope'])
                    peak_confidences.append(peak_confidence)
                    if verbose:
                        print(f"Peak at {peak:2f} +- {std:2f} keV matched to {row['Isotope']} at {energy} keV with confidence {peak_confidence:.2f}")
                    
            if len(matched)==0:
                continue

            elif len(matched)/len(row['Energies (keV)']) >= matching_ratio:
                    identified_isotopes.append(row['Isotope'])
                    isotope_confidences.append(np.mean(peak_confidences))
                    percentage_matched.append(len(matched)/len(row['Energies (keV)']))
                    identified_peaks.append(peak)
                    if verbose:
                        print(f"Isotope {row['Isotope']} identified with {len(matched)/len(row['Energies (keV)'])*100:.2f}% of peaks matched")

            elif len(matched)/len(row['Energies (keV)']) < matching_ratio:
                if verbose:
                    print(f"Isotope {row['Isotope']} not sufficiently identified")

    return identified_isotopes, identified_peaks, isotope_confidences, percentage_matched

def process_spectrum(filename: str, prominence: int = 1000, width: int = None, rel_height: float = None, tolerance: float = 0.5, verbose=False, **kwargs) -> pd.DataFrame:
    """
    Processes a gamma spectrum file and identifies peaks.
    This function reads a gamma spectrum data file, processes the data to identify peaks,
    fits Gaussian functions to the peaks, and returns a DataFrame with the results.
    
    Parameters:
    -----------
    filename : str
        The path to the gamma spectrum data file.
    prominence : int, optional
        The prominence of peaks to be identified (default is 1000).
    width : int, optional
        The width of peaks to be identified (default is None).
    rel_height : float, optional
        The relative height of peaks to be identified (default is None).
    **kwargs : dict
        Additional keyword arguments to be passed to the `find_peaks` function.
    Returns:
    --------
    pd.DataFrame
        A DataFrame containing the following columns:
        - 'filename': The name of the processed file.
        - 'data': The processed data as a DataFrame.
        - 'peaks': The indices of the identified peaks.
        - 'properties': The properties of the identified peaks.
        - 'calculated_polynomial': The polynomial values at the identified peaks.
        - 'fitted_peaks': The fitted Gaussian peaks.
    Notes:
    ------
    - The input file is expected to have a header with at least 3 lines.
    - The energy values in the file should be in a column named 'energy in keV' and should use commas as decimal separators.
    - The function reads a polynomial from a specific file ('Daten/2016-11-21_09-27-54_Summenspektrum.txt') to calibrate the energy values. When the polynomial is not found or readable in the input file.
    Example:
    --------
    >>> result = process_spectrum('spectrum_data.txt', prominence=1500)
    >>> print(result)
    Time Complexity:
    ----------------
    The time complexity of this function depends on the size of the input data and the complexity of the peak finding and fitting algorithms.
    - Reading and processing the data: O(n), where n is the number of lines in the file.
    - Finding peaks: O(m), where m is the number of data points.
    - Fitting Gaussian functions: O(p), where p is the number of peaks.
    Overall, the time complexity is approximately O(n + m + p).
    """

    data = pd.read_table(filename, header=3)
    data['energy in keV'] = data['energy in keV'].str.replace(',', '.').astype(float)

    try:
        with open(filename, 'r') as file:
            lines = file.readlines()
        line_3 = lines[2].strip()  # Line 3 (index 2) and strip any leading/trailing whitespace
        polynomial = parse_polynomial(line_3)
        test_poly = polynomial(1)
    except ValueError:
        with open('Daten/2016-11-21_09-27-54_Summenspektrum.txt', 'r') as file:
            lines = file.readlines()
        line_3 = lines[2].strip()  # Line 3 (index 2) and strip any leading/trailing whitespace
        polynomial = parse_polynomial(line_3)

    peaks, properties = find_peaks(data['counts'], prominence=prominence, width=width, rel_height=rel_height)
    fitted_peaks = fit_gaussian(data, peaks, properties, polynomial=polynomial)
    identified_isotopes, identified_peaks, confidences, matched = identify_isotopes(fitted_peaks, tolerance=tolerance, verbose=verbose)
    total_confidences = [c * p for c, p in zip(confidences, matched)]
    return pd.DataFrame({
        'filename': [filename],
        'data': [data],
        'peaks': [peaks],
        'properties': [properties],
        'calculated_polynomial': [polynomial(peaks)],
        'fitted_peaks': [fitted_peaks],
        'fitted_peaks_mean': [unumpy.nominal_values(fitted_peaks)],
        'fitted_peaks_std': [unumpy.std_devs(fitted_peaks)],
        'identified_isotopes': [identified_isotopes],
        'identified_peaks': [identified_peaks],
        'confidences': [confidences],
        'matched': [matched],
        'total_confidences': [total_confidences]
    })
    