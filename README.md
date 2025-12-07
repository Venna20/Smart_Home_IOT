# Smart Home IoT Analytics & Machine Learning

A comprehensive data analysis and machine learning project for predicting smart home device failures using SQL feature engineering, predictive modeling, and interactive visualizations.

## 📋 Project Overview

This project analyzes smart home IoT device data to:
- Predict device failures using machine learning
- Identify patterns in sensor readings and alerts
- Visualize insights through interactive dashboards
- Perform advanced SQL-based feature engineering

## 🗂️ Project Structure

```
PythonSQL/
├── Code/
│   ├── Project_Phase2.ipynb          # Main analysis notebook
│   └── Updated+Smart+Home+IoT+Task+2.pdf
├── Data/
│   ├── smart_home.db                  # SQLite database
│   └── SMART+HOME+IoT+–+Data+Dictionary+.pdf
├── Docs/
│   ├── AD599+Team+3+Final+Project+Phase+1 (1).pdf
│   └── Task3_ML_Insights_Summary (2).pdf
├── Outputs/
│   ├── confusion_matrices.png
│   ├── Dashboard.png
│   ├── lr_coefficients.png
│   └── model_comparison.png
├── README.md
└── requirements.txt
```

## 📊 Database Schema

The `smart_home.db` SQLite database contains four main tables:

- **homes** (200 records): Home information including city, type, and ownership details
- **devices** (925 records): Smart home devices with installation dates and status
- **sensor_readings** (222,000 records): Time-series sensor data from devices
- **alerts** (77 records): Alert notifications with severity levels

## 🤖 Machine Learning Models

### Models Implemented

1. **Logistic Regression (Baseline)** ⭐ Best Performance
   - Accuracy: 79.46%
   - AUC-ROC: 0.8136
   - F1-Score: 0.5476

2. **Random Forest**
   - Accuracy: 71.89%
   - AUC-ROC: 0.7669
   - F1-Score: 0.3158

3. **Gradient Boosting**
   - Accuracy: 68.65%
   - AUC-ROC: 0.7237
   - F1-Score: 0.3556

### Key Features

The model uses 27 engineered features including:
- Device age and type
- Sensor reading statistics
- Alert history
- Home-level metrics
- Recent activity patterns

### Top Predictive Features

1. **Home offline rate** - Strongest predictor of device failure
2. **Device age in days** - Older devices more likely to fail
3. **Offline devices in home** - Network/environmental factors
4. **Variance in numeric readings** - Unstable readings indicate issues
5. **Recent average values** - Recent activity patterns

## 📈 Key Insights

### Device Failure Analysis
- Overall device offline rate: **25.73%**
- Motion sensors have highest failure rate: **31.20%**
- Device failures increase with age, particularly after 18 months

### Alert Patterns
- 77 total alerts across all devices
- High severity alerts correlate with device failures
- Alert frequency varies by city and home type

### Feature Importance
- Home-level metrics are strongest predictors
- Recent activity patterns (last 7 days) highly informative
- Device type and location play secondary roles

## 🎨 Visualizations

The project generates multiple visualizations:

1. **Model Performance Comparisons**
   - Accuracy, AUC-ROC, and F1-Score metrics
   - Confusion matrices for all models
   - ROC curves comparing model performance

2. **Feature Analysis**
   - Feature importance rankings
   - Logistic regression coefficients
   - Top predictive factors visualization

3. **Business Insights**
   - Failure rates by device type
   - Alerts by city and home type
   - Daily alert trends
   - Device installation patterns

4. **Interactive Dashboard**
   - Multi-panel Plotly dashboard
   - Real-time alert monitoring
   - ML model comparison

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Jupyter Notebook
- SQLite3

### Installation

1. Clone the repository:
```bash
cd PythonSQL
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Launch Jupyter Notebook:
```bash
jupyter notebook Code/Project_Phase2.ipynb
```

### Running the Analysis

1. Open `Project_Phase2.ipynb` in Jupyter
2. Ensure `smart_home.db` is in the Data folder (or update the DB_PATH variable)
3. Run all cells sequentially (Cell → Run All)

The notebook will:
- Connect to the database
- Perform SQL-based feature engineering
- Train and evaluate ML models
- Generate visualizations
- Export results to CSV and PNG files

## 📦 Output Files

The analysis generates the following files:

**CSV Files:**
- `device_failure_features.csv` - Engineered features dataset
- `model_performance_summary.csv` - Comprehensive model metrics
- `model_comparison.csv` - Side-by-side model comparison
- `rf_feature_importance.csv` - Random Forest feature rankings
- `gb_feature_importance.csv` - Gradient Boosting feature rankings

**Visualizations:**
- `target_distribution.png` - Class balance visualization
- `model_comparison.png` - Model performance metrics
- `confusion_matrices.png` - Confusion matrices for all models
- `feature_importance.png` - Feature importance plots
- `lr_coefficients.png` - Logistic regression coefficients
- `failure_by_device_type.png` - Failure rates by device category

## 🔍 SQL Feature Engineering

The project uses advanced SQL techniques:

- **Common Table Expressions (CTEs)** for modular queries
- **Window functions** for time-series analysis
- **Aggregate functions** for statistical features
- **Date calculations** using Julian day functions
- **Conditional aggregation** with CASE statements

Example features extracted via SQL:
- Device age calculation
- Rolling averages for recent activity
- Home-level device failure rates
- Metric-specific reading counts
- Alert severity distribution

## 📊 Cross-Validation Results

5-fold stratified cross-validation (AUC-ROC):

| Model                | Mean AUC | Std Dev |
|----------------------|----------|---------|
| Logistic Regression  | 0.7786   | ±0.0487 |
| Random Forest        | 0.7333   | ±0.0416 |
| Gradient Boosting    | 0.7275   | ±0.0608 |

## 🎯 Business Applications

### Predictive Maintenance
- Identify devices likely to fail before they do
- Schedule proactive replacements
- Reduce customer support calls

### Inventory Management
- Prioritize stock for high-failure device types
- Optimize replacement part inventory

### Quality Assurance
- Identify manufacturing or installation issues
- Track device reliability metrics
- Improve product design based on failure patterns

### Customer Experience
- Proactive customer notifications
- Targeted warranty programs
- Improved service level agreements

## 🛠️ Technologies Used

- **Python 3.x** - Core programming language
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computations
- **SQLite3** - Database connectivity
- **Scikit-learn** - Machine learning algorithms
- **Matplotlib** - Static visualizations
- **Seaborn** - Statistical data visualization
- **Plotly** - Interactive dashboards
- **Jupyter Notebook** - Interactive development environment

## 📝 Key Findings Summary

1. **Logistic Regression outperforms** complex ensemble methods on this dataset
2. **Home-level metrics** are more predictive than device-level features
3. **Recent activity patterns** (7-day window) provide strong signals
4. **Motion sensors** have the highest failure rate at 31.2%
5. **Device age** is a significant but not dominant predictor
6. **Environmental factors** (home offline rate) strongly influence device reliability

## 🔮 Future Enhancements

- [ ] Implement time-series forecasting for alert prediction
- [ ] Add clustering analysis for device behavior patterns
- [ ] Develop real-time anomaly detection
- [ ] Create automated reporting pipeline
- [ ] Build web-based dashboard for stakeholders
- [ ] Integrate external data (weather, power outages)
- [ ] Implement deep learning models (LSTM for time-series)
- [ ] Add explainable AI (SHAP values) for model interpretability

## 👥 Contributors

This project was developed as part of AD599 - Data Analytics coursework.

## 📄 License

This project is for educational purposes.

## 📧 Contact

For questions or feedback, please refer to the documentation in the `Docs/` folder.

---

**Last Updated:** December 2025

