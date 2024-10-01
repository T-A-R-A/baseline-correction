%matplotlib widget
import csv
import os
import numpy as np
import pandas as pd
import plotly.io as pio
import plotly.express as px
from scipy import sparse
from scipy.sparse.linalg import spsolve
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import tkinter as tk
from tkinter import filedialog, simpledialog
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science','nature'])
import seaborn as sns

def select_csv_file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select a CSV file",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    return file_path

def read_csv_data(file_path, synchroniser):
    tt, z, xx, yy = [], [], [], []
    with open(file_path, mode='r') as file:
        csv_reader = csv.reader(file)
        rows = list(csv_reader)
        for row in rows[2:]:
            if len(row) > 3:
                tt.append(float(row[0]))
                z.append(float(row[1]))
                xx.append(float(row[2]))
                yy.append(float(row[3]))
    t = [element + synchroniser for element in tt]
    return np.array(t), np.array(z), np.array(xx), np.array(yy)

def whittaker_smooth(x, w, lambda_, differences=1):
    X = np.matrix(x)
    m = X.size
    E = sparse.eye(m, format='csc')
    D = sparse.diags([1, -2, 1], offsets=[0, -1, -2], shape=(m, m-2))
    W = sparse.diags(w, 0, shape=(m, m))
    A = W + (lambda_ * D.T * D)
    B = W * X.T
    background = spsolve(A, B)
    return np.array(background).flatten()

def air_pls(x, lambda_=15, porder=1, itermax=15):
    m = x.shape[0]
    w = np.ones(m)
    z = x.copy()  # Initialize z
    for i in range(1, itermax+1):
        z = whittaker_smooth(x, w, lambda_, porder)
        d = x - z
        dssn = np.abs(d[d<0].sum())
        if dssn < 0.001 * (abs(x)).sum() or i == itermax:
            if i == itermax:
                print('WARNING max iteration reached!')
            break
        w[d >= 0] = 0
        w[d < 0] = np.exp(i * np.abs(d[d<0]) / dssn)
        w[0] = np.exp(i * (d[d<0]).max() / dssn)
        w[-1] = w[0]
    return z

def arpls_baseline(y, lam=1e4, ratio=0.05, itermax=100):
    N = len(y)
    D = sparse.diags([1, -2, 1], offsets=[0, -1, -2], shape=(N, N-2))
    D = lam * D.dot(D.transpose())
    w = np.ones(N)
    z = y.copy()
    for i in range(itermax):
        W = sparse.diags(w, 0, shape=(N, N))
        Z = W + D
        z = spsolve(Z, w * y)
        d = y - z
        dn = d[d < 0]
        m = np.mean(dn)
        s = max(np.std(dn), 1e-6)
        exp_arg = np.clip(2 * (d - (2 * s - m)) / s, -709, 709)
        wt = 1.0 / (1 + np.exp(exp_arg))
        if np.linalg.norm(w - wt) / np.linalg.norm(w) < ratio:
            break
        w = wt
    return z

def adaptive_arpls(y, lam=1e4, ratio=0.05, itermax=100):
    """
    Adaptive ARPLS baseline correction with noise estimation

    Parameters:
        y : array_like
            Input signal
        lam : float, optional
            Initial smoothness parameter
        ratio : float, optional
            Asymmetry parameter
        itermax : int, optional
            Maximum number of iterations

    Returns:
        baseline : ndarray
            The estimated baseline
    """
    diff = np.diff(y)
    noise_level = np.median(np.abs(diff)) / 0.6745

    lam_adjusted = float(lam * (noise_level**2))

    baseline = arpls_baseline(y, lam=lam_adjusted, ratio=ratio, itermax=itermax)

    return baseline

def process_data(t, z, xx, yy, baseline_params, method='airpls'):
    if method == 'airpls':
        z_lambda, z_porder, z_itermax = baseline_params['z']
        x_lambda, x_porder, x_itermax = baseline_params['x']
        y_lambda, y_porder, y_itermax = baseline_params['y']

        cz = z - air_pls(z, lambda_=z_lambda, porder=z_porder, itermax=z_itermax)
        cx = xx - air_pls(xx, lambda_=x_lambda, porder=x_porder, itermax=x_itermax)
        cy = yy - air_pls(yy, lambda_=y_lambda, porder=y_porder, itermax=y_itermax)
    elif method == 'arpls':
        z_lam, z_ratio, z_itermax = baseline_params['z']
        x_lam, x_ratio, x_itermax = baseline_params['x']
        y_lam, y_ratio, y_itermax = baseline_params['y']

        cz = z - adaptive_arpls(z, lam=z_lam, ratio=z_ratio, itermax=z_itermax)
        cx = xx - adaptive_arpls(xx, lam=x_lam, ratio=x_ratio, itermax=x_itermax)
        cy = yy - adaptive_arpls(yy, lam=y_lam, ratio=y_ratio, itermax=y_itermax)
    else:
        raise ValueError("Invalid method. Choose 'airpls' or 'arpls'.")

    return cz, cx, cy

def create_dataframes(t, z, cz, xx, cx, yy, cy):
    dt = pd.DataFrame({'Time': t, 'Time(s)': t})
    dz = pd.DataFrame({'RAW': z, 'Fit': cz})
    dx = pd.DataFrame({'RAW': xx, 'Fit': cx})
    dy = pd.DataFrame({'RAW': yy, 'Fit': cy})
    return dt, dz, dx, dy

def plot_data(dt, dz, dx, dy, file_path, plot_type='plotly'):
    colors = ['blue', 'red', 'green']
    base_filename = os.path.splitext(os.path.basename(file_path))[0]
    output_filename = f"{base_filename}_plot.png"

    if plot_type == 'plotly':
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=('Z AXIS', 'Y AXIS', 'X AXIS'))

        for i, (d, color) in enumerate(zip([dz, dy, dx], colors), start=1):
            fig.add_trace(go.Scatter(x=dt['Time'], y=d['Fit'], name=f'Fit Data {["Z", "Y", "X"][i-1]}', line=dict(color=color)), row=i, col=1)
            fig.add_trace(go.Scatter(x=dt['Time'], y=d['RAW'], name=f'Raw Data {["Z", "Y", "X"][i-1]}', line=dict(color='rgba(128, 128, 128, 0.2)')), row=i, col=1)

        fig.update_layout(height=900, width=1800, title_text=f"Combined Plot of ...{file_path[-20:]}", legend_title_text='Data Series')
        fig.update_xaxes(title_text="Time (s)")
        fig.update_yaxes(title_text="Field Strength (Oe)")
        fig.update_layout(xaxis3=dict(rangeslider=dict(visible=True, yaxis=dict(rangemode='auto'), bgcolor='darkgrey', thickness=0.05), type="linear"))
        fig.write_image(output_filename, scale=2)  # Save as high-quality PNG
        fig.show()

    elif plot_type == 'matplotlib':
        fig, axs = plt.subplots(3, 1, figsize=(18, 27), sharex=True, dpi=100)
        for ax, d, color, axis in zip(axs, [dz, dy, dx], colors, ['Z', 'Y', 'X']):
            ax.plot(dt['Time'], d['Fit'], label=f'Fit Data {axis}', color=color)
            ax.plot(dt['Time'], d['RAW'], label=f'Raw Data {axis}', alpha=0.2, color='gray')
            ax.set_title(f"{axis} AXIS")
            ax.set_ylabel("Field Strength (Oe)")
            ax.legend()
        axs[-1].set_xlabel("Time (s)")
        plt.suptitle(f"Combined Plot of ...{file_path[-20:]}")
        # plt.tight_layout()
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        plt.show()

    elif plot_type == 'seaborn':
        fig, axs = plt.subplots(3, 1, figsize=(18, 27), sharex=True, dpi=300)
        for ax, d, color, axis in zip(axs, [dz, dy, dx], colors, ['Z', 'Y', 'X']):
            sns.lineplot(x=dt['Time'], y=d['Fit'], ax=ax, label=f'Fit Data {axis}', color=color)
            sns.lineplot(x=dt['Time'], y=d['RAW'], ax=ax, label=f'Raw Data {axis}', alpha=0.2, color='gray')
            ax.set_title(f"{axis} AXIS")
            ax.set_ylabel("Field Strength (Oe)")
            ax.legend()
        axs[-1].set_xlabel("Time (s)")
        plt.suptitle(f"Combined Plot of ...{file_path[-20:]}")
        plt.tight_layout()
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        plt.show()

    print(f"Plot saved as {output_filename}")

def save_results(file_path, combined_array):
    # Get the base filename without path
    base_filename = os.path.basename(file_path)

    # Create the new filename with "baseline_" prefix
    new_filename = f"baseline_{base_filename}"

    # Save the file in the current directory
    np.savetxt(new_filename, combined_array, delimiter=',', fmt='%s',
               header='Time,Z,Fit Z,X,Fit X,Y,Fit Y', comments='')
    print(f"The combined CSV file has been saved as '{new_filename}' in the current directory")

def main():
    synchroniser = 0.00001
    air_pls_params = {
        'z': (15, 1, 15),  # (lambda, porder, itermax) for z-axis
        'y': (15, 1, 15),  # (lambda, porder, itermax) for y-axis
        'x': (15, 1, 15)   # (lambda, porder, itermax) for x-axis
    }
    arpls_params = {
        'z': (1e8, 0.083, 1000),  # (lam, ratio, itermax) for z-axis
        'y': (1e8, 0.078, 1000),  # (lam, ratio, itermax) for y-axis
        'x': (1e8, 0.080, 1000)   # (lam, ratio, itermax) for x-axis
    }
    plot_type = 'matplotlib'

    # baseline correction method
    method = 'arpls'
    # method = 'airpls'

    file_path = select_csv_file()
    if not file_path:
        print("No file selected")
        return

    t, z, xx, yy = read_csv_data(file_path, synchroniser)

    # Choose the appropriate parameters based on the selected method
    baseline_params = air_pls_params if method == 'airpls' else arpls_params

    cz, cx, cy = process_data(t, z, xx, yy, baseline_params, method)
    dt, dz, dx, dy = create_dataframes(t, z, cz, xx, cx, yy, cy)

    plot_data(dt, dz, dx, dy, file_path, plot_type)

    combined_array = np.column_stack((t, z, cz, xx, cx, yy, cy))
    save_results(file_path, combined_array)

if __name__ == '__main__':
    main()
