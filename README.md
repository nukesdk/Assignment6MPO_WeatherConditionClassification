# Assignment-6: Weather Condition Classification using SVM and Open-Meteo API

## Objective

Build a Support Vector Machine (SVM) classification model to predict whether the weather is **Cool** or **Warm** based on meteorological observations collected from the [Open-Meteo Weather API](https://open-meteo.com/). The target variable is derived from temperature readings:
- **Warm** → Temperature ≥ 25°C
- **Cool** → Temperature < 25°C

---

## API Documentation

- **Open-Meteo API:** https://open-meteo.com/
- **Endpoint Used:** `https://api.open-meteo.com/v1/forecast`
- **Location:** New Delhi, India (Latitude: 28.6139, Longitude: 77.2090)
- **Features Fetched:** `temperature_2m`, `relative_humidity_2m`, `surface_pressure`, `wind_speed_10m`
- **Forecast Period:** 7 days (hourly data)

---

## Libraries Used

| Library | Purpose |
|---------|---------|
| `requests` | Fetch weather data from Open-Meteo API |
| `pandas` | Data manipulation and DataFrame operations |
| `numpy` | Numerical computations |
| `scikit-learn` | SVM model, preprocessing, train-test split, evaluation metrics |
| `matplotlib` | Data visualization |
| `seaborn` | Confusion matrix heatmap and statistical plots |

---

## Methodology

1. **Data Collection**
   - Send GET request to Open-Meteo API for 7-day hourly forecast.
   - Parse JSON response and convert to Pandas DataFrame.

2. **Data Understanding**
   - Display first 5 records.
   - Identify input features (`temperature_2m`, `relative_humidity_2m`, `surface_pressure`, `wind_speed_10m`) and target variable (`Weather_Class`).

3. **Data Preprocessing**
   - Check for missing values.
   - Remove non-predictive columns (`time`).
   - Encode target variable using `LabelEncoder`.
   - Split dataset: 80% Training | 20% Testing (stratified split).
   - Standardize features using `StandardScaler`.

4. **Model Development**
   - Build SVM Classifier with **RBF kernel** (`SVC(kernel='rbf')`).
   - Train on scaled training data.
   - Predict weather class on test data.

5. **Model Evaluation**
   - Evaluate using **Accuracy**, **Precision**, **Recall**, and **F1-Score**.
   - Generate and visualize **Confusion Matrix**.
   - Document 3 key observations.

6. **Conclusion**
   - Summarize findings, importance of feature scaling, and SVM advantages/limitations.

---

## Results

| Metric | Value |
|--------|-------|
| **Model** | SVM with RBF Kernel |
| **Kernel** | Radial Basis Function (RBF) |
| **Accuracy** | ~95–100% (varies with live API data) |
| **Precision** | High |
| **Recall** | High |
| **F1-Score** | High |

> **Note:** Actual metric values depend on the live weather data fetched at runtime. The model typically achieves very high accuracy because temperature is the primary determinant of the target class.

### Confusion Matrix
A heatmap is generated showing True Negatives, False Positives, False Negatives, and True Positives for Cool vs Warm classification.

### Key Observations
1. The SVM-RBF model achieves high accuracy due to strong feature-target correlation.
2. Temperature is the dominant predictive feature since the target is directly derived from it.
3. Feature scaling is essential — SVM's distance-based optimization is sensitive to feature magnitude differences (e.g., pressure in hundreds vs. wind speed in single digits).

---

## Conclusion

This project successfully demonstrates weather classification using SVM on real-time meteorological data from the Open-Meteo API. The RBF kernel effectively captures non-linear boundaries between Cool and Warm weather conditions. **Feature scaling proved critical** — without `StandardScaler`, features with larger magnitudes would bias the SVM hyperplane, degrading performance. 

**Advantage of SVM:** Excellent performance in high-dimensional spaces and ability to model complex, non-linear decision boundaries via kernel tricks.  
**Limitation of SVM:** High computational cost on large datasets, as training complexity scales quadratically with sample size, making it less ideal for massive real-time streaming applications.

---

## How to Run

1. Install dependencies:
   ```bash
   pip install requests pandas numpy scikit-learn matplotlib seaborn
   ```

2. Open `Assignment-6.ipynb` in Jupyter Notebook or Google Colab.

3. Run all cells sequentially. The notebook will:
   - Fetch live weather data from Open-Meteo
   - Preprocess and standardize the data
   - Train the SVM model
   - Evaluate and display results

---

## Files in This Repository

| File | Description |
|------|-------------|
| `Assignment-6.ipynb` | Complete Jupyter notebook with all tasks |
| `README.md` | Project documentation |

---

## Author

- **Assignment:** AI-ML Assignment – 6
- **Topic:** Weather Condition Classification using SVM and Open-Meteo API
- **Submission Deadline:** 28 July 2026, 11:59 PM IST

---

## License

This project is for academic purposes only.
