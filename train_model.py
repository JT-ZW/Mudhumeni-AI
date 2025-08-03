# Enhanced model training script optimized for Southern African crop recommendation
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Core ML libraries
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, GridSearchCV, 
    RandomizedSearchCV, cross_val_score
)
from sklearn.preprocessing import (
    MinMaxScaler, StandardScaler, RobustScaler, 
    LabelEncoder, PowerTransformer
)
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, 
    VotingClassifier, AdaBoostClassifier
)
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_fscore_support, roc_auc_score
)

# Advanced ML libraries
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    print("XGBoost not available. Install with: pip install xgboost")
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    print("LightGBM not available. Install with: pip install lightgbm")
    LIGHTGBM_AVAILABLE = False

# Visualization and analysis
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency
import pickle
import os
from datetime import datetime

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Create directories
os.makedirs('models', exist_ok=True)
os.makedirs('models/plots', exist_ok=True)
os.makedirs('models/analysis', exist_ok=True)

print("="*60)
print("ENHANCED SOUTHERN AFRICAN CROP RECOMMENDATION MODEL")
print("="*60)

# Load and preprocess dataset
print("\n📊 Loading and analyzing dataset...")
try:
    crop = pd.read_csv('Crop_recommendation.csv')
    print(f"✅ Dataset loaded successfully: {crop.shape[0]} samples, {crop.shape[1]} features")
except FileNotFoundError:
    print("❌ Error: Crop_recommendation.csv not found!")
    exit(1)

# Southern African crop mapping and regional suitability
print("\n🌍 Defining Southern African crop suitability...")

# Define crop suitability for Southern Africa (0-3 scale: 0=unsuitable, 3=excellent)
southern_africa_crop_suitability = {
    'maize': 3,        # Staple crop, excellent for region
    'rice': 1,         # Limited suitability, mostly lowlands
    'wheat': 2,        # Good for winter, cooler areas
    'sorghum': 3,      # Excellent drought tolerance
    'millet': 3,       # Traditional crop, drought resistant
    'cotton': 3,       # Major cash crop
    'tobacco': 3,      # Important export crop
    'soybeans': 3,     # Nitrogen fixing, good rotation crop
    'sunflower': 3,    # Drought tolerant, good for oil
    'groundnuts': 3,   # Traditional legume, good for soil
    'mango': 2,        # Suitable for warmer areas
    'orange': 2,       # Commercial potential
    'banana': 2,       # Limited to suitable climate zones
    'grapes': 2,       # Wine regions in SA
    'coffee': 1,       # Limited to highland areas
    'coconut': 0,      # Not suitable for climate
    'jute': 0,         # Not traditionally grown
    'papaya': 1,       # Limited climate suitability
    'apple': 1,        # Limited to cooler highland areas
    'muskmelon': 2,    # Seasonal, with irrigation
    'watermelon': 2,   # Good summer crop
    'pomegranate': 2,  # Drought tolerant fruit
    'lentil': 2,       # Good winter legume
    'blackgram': 2,    # Suitable legume
    'mungbean': 2,     # Quick growing legume
    'mothbeans': 2,    # Drought tolerant legume
    'pigeonpeas': 3,   # Excellent for region
    'kidneybeans': 2,  # Good with irrigation
    'chickpea': 2,     # Good winter crop
}

# Create enhanced crop dictionary with regional weighting
crop_dict = {
    'rice': 1, 'maize': 2, 'jute': 3, 'cotton': 4, 'coconut': 5,
    'papaya': 6, 'orange': 7, 'apple': 8, 'muskmelon': 9, 'watermelon': 10,
    'grapes': 11, 'mango': 12, 'banana': 13, 'pomegranate': 14,
    'lentil': 15, 'blackgram': 16, 'mungbean': 17, 'mothbeans': 18,
    'pigeonpeas': 19, 'kidneybeans': 20, 'chickpea': 21, 'coffee': 22
}

# Create reverse mapping
reverse_crop_dict = {v: k for k, v in crop_dict.items()}

# Map labels to numeric values
crop['label_num'] = crop['label'].map(crop_dict)
crop['suitability_score'] = crop['label'].map(southern_africa_crop_suitability)

print(f"✅ Regional suitability scores assigned")

# Enhanced Data Analysis
print("\n🔍 Performing comprehensive data analysis...")

# Basic statistics
print("\nDataset Statistics:")
print(f"- Total samples: {len(crop)}")
print(f"- Features: {list(crop.columns[:-3])}")  # Excluding label, label_num, suitability_score
print(f"- Unique crops: {crop['label'].nunique()}")
print(f"- Samples per crop: {crop['label'].value_counts().min()} - {crop['label'].value_counts().max()}")

# Check for missing values and outliers
print(f"- Missing values: {crop.isnull().sum().sum()}")

# Detect outliers using IQR method
def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return ((df[column] < lower_bound) | (df[column] > upper_bound)).sum()

numerical_features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
total_outliers = sum(detect_outliers_iqr(crop, col) for col in numerical_features)
print(f"- Outliers detected: {total_outliers}")

# Enhanced Feature Engineering
print("\n⚙️ Creating enhanced features...")

# Create feature interactions
crop['NPK_ratio'] = crop['N'] / (crop['P'] + crop['K'] + 1)  # +1 to avoid division by zero
crop['NP_ratio'] = crop['N'] / (crop['P'] + 1)
crop['NK_ratio'] = crop['N'] / (crop['K'] + 1)
crop['PK_ratio'] = crop['P'] / (crop['K'] + 1)

# Nutrient balance features
crop['total_nutrients'] = crop['N'] + crop['P'] + crop['K']
crop['nutrient_balance'] = np.sqrt((crop['N']**2 + crop['P']**2 + crop['K']**2) / 3)

# Climate comfort index
crop['temp_humidity_index'] = crop['temperature'] * crop['humidity'] / 100
crop['climate_stress'] = np.abs(crop['temperature'] - 25) + np.abs(crop['humidity'] - 60)

# Soil suitability index
crop['soil_fertility'] = (crop['N'] + crop['P'] + crop['K']) / 3
crop['ph_optimality'] = 1 / (1 + np.abs(crop['ph'] - 6.5))  # Optimal pH around 6.5

# Water stress indicator
crop['water_stress'] = np.where(crop['rainfall'] < 500, 1, 0)

# Seasonal adaptation (assuming data represents year-round potential)
crop['growing_season_score'] = (
    crop['rainfall'] * 0.4 + 
    crop['temperature'] * 0.3 + 
    crop['humidity'] * 0.3
) / 100

print("✅ Enhanced features created:")
new_features = ['NPK_ratio', 'NP_ratio', 'NK_ratio', 'PK_ratio', 'total_nutrients', 
               'nutrient_balance', 'temp_humidity_index', 'climate_stress', 
               'soil_fertility', 'ph_optimality', 'water_stress', 'growing_season_score']
print(f"   {new_features}")

# Enhanced Visualization
print("\n📊 Creating comprehensive visualizations...")

# 1. Regional suitability distribution
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
suitability_counts = crop.groupby('suitability_score')['label'].count()
plt.pie(suitability_counts.values, labels=[f'Score {i}' for i in suitability_counts.index], 
        autopct='%1.1f%%', startangle=90)
plt.title('Regional Suitability Distribution')

# 2. Enhanced correlation heatmap
plt.subplot(2, 3, 2)
feature_cols = numerical_features + new_features[:6]  # Top 6 new features
corr_matrix = crop[feature_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu_r', center=0, 
            square=True, linewidths=0.5, fmt='.2f')
plt.title('Enhanced Feature Correlations')

# 3. Crop distribution by suitability
plt.subplot(2, 3, 3)
crop_suitability = crop.groupby(['suitability_score', 'label']).size().unstack(fill_value=0)
crop_suitability.plot(kind='bar', stacked=True, alpha=0.8)
plt.title('Crops by Regional Suitability')
plt.xlabel('Suitability Score')
plt.ylabel('Number of Samples')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# 4. Feature importance preview (using Random Forest)
plt.subplot(2, 3, 4)
X_preview = crop[numerical_features + new_features]
y_preview = crop['label_num']
rf_preview = RandomForestClassifier(n_estimators=100, random_state=42)
rf_preview.fit(X_preview, y_preview)
feature_importance = pd.DataFrame({
    'feature': X_preview.columns,
    'importance': rf_preview.feature_importances_
}).sort_values('importance', ascending=True).tail(10)

plt.barh(range(len(feature_importance)), feature_importance['importance'])
plt.yticks(range(len(feature_importance)), feature_importance['feature'])
plt.title('Top 10 Feature Importance (Preview)')
plt.xlabel('Importance')

# 5. Climate zones analysis
plt.subplot(2, 3, 5)
plt.scatter(crop['temperature'], crop['rainfall'], c=crop['suitability_score'], 
           cmap='RdYlGn', alpha=0.6)
plt.colorbar(label='Regional Suitability')
plt.xlabel('Temperature (°C)')
plt.ylabel('Rainfall (mm)')
plt.title('Climate Zones by Suitability')

# 6. Nutrient analysis
plt.subplot(2, 3, 6)
high_suitability = crop[crop['suitability_score'] >= 2]
crop_nutrients = high_suitability.groupby('label')[['N', 'P', 'K']].mean()
top_crops = crop_nutrients.head(8)
top_crops.plot(kind='bar', alpha=0.8)
plt.title('Nutrient Requirements (High Suitability Crops)')
plt.xlabel('Crops')
plt.ylabel('Nutrient Level')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('models/plots/comprehensive_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Prepare enhanced feature set
print("\n🎯 Preparing enhanced dataset...")

# Select features for modeling
base_features = numerical_features
enhanced_features = new_features
all_features = base_features + enhanced_features

# Create modeling dataset
X = crop[all_features].copy()
y = crop['label_num'].copy()

# Apply weights based on regional suitability
sample_weights = crop['suitability_score'].values
sample_weights = np.where(sample_weights == 0, 0.1, sample_weights)  # Avoid zero weights

print(f"✅ Feature set prepared: {len(all_features)} features")
print(f"✅ Sample weights applied based on regional suitability")

# Advanced preprocessing
print("\n🔧 Advanced preprocessing...")

# Handle outliers using robust scaling
def robust_outlier_treatment(X, threshold=3):
    """Remove extreme outliers using modified Z-score"""
    X_clean = X.copy()
    
    for column in X.select_dtypes(include=[np.number]).columns:
        median = X[column].median()
        mad = np.median(np.abs(X[column] - median))
        modified_z_scores = 0.6745 * (X[column] - median) / mad
        X_clean = X_clean[np.abs(modified_z_scores) < threshold]
    
    return X_clean

# Split data first, then preprocess
X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
    X, y, sample_weights, test_size=0.2, random_state=42, stratify=y
)

print(f"✅ Data split: {len(X_train)} training, {len(X_test)} testing samples")

# Advanced scaling pipeline
def create_preprocessing_pipeline(X_train):
    """Create comprehensive preprocessing pipeline"""
    
    # 1. Power transformation for normality
    power_transformer = PowerTransformer(method='yeo-johnson', standardize=False)
    
    # 2. Robust scaling for outlier resistance
    robust_scaler = RobustScaler()
    
    # 3. Standard scaling for final normalization
    standard_scaler = StandardScaler()
    
    # Apply transformations
    X_transformed = power_transformer.fit_transform(X_train)
    X_transformed = robust_scaler.fit_transform(X_transformed)
    X_transformed = standard_scaler.fit_transform(X_transformed)
    
    return X_transformed, (power_transformer, robust_scaler, standard_scaler)

X_train_scaled, scalers = create_preprocessing_pipeline(X_train)

# Apply same transformations to test set
power_transformer, robust_scaler, standard_scaler = scalers
X_test_scaled = power_transformer.transform(X_test)
X_test_scaled = robust_scaler.transform(X_test_scaled)
X_test_scaled = standard_scaler.transform(X_test_scaled)

print("✅ Advanced preprocessing completed")

# Enhanced Model Training
print("\n🤖 Training enhanced models...")

models = {}

# 1. Enhanced Gaussian Naive Bayes
models['Enhanced_NB'] = GaussianNB(var_smoothing=1e-10)

# 2. Optimized Random Forest
models['Enhanced_RF'] = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    bootstrap=True,
    random_state=42,
    class_weight='balanced'
)

# 3. Extra Trees (more randomization)
models['ExtraTrees'] = ExtraTreesClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    bootstrap=False,
    random_state=42,
    class_weight='balanced'
)

# 4. Support Vector Machine with RBF kernel
models['Enhanced_SVM'] = SVC(
    kernel='rbf',
    probability=True,
    random_state=42,
    class_weight='balanced'
)

# 5. Logistic Regression
models['LogisticRegression'] = LogisticRegression(
    max_iter=1000,
    random_state=42,
    class_weight='balanced'
)

# 6. K-Nearest Neighbors
models['KNN'] = KNeighborsClassifier(
    n_neighbors=7,
    weights='distance'
)

# 7. XGBoost (if available)
if XGBOOST_AVAILABLE:
    models['XGBoost'] = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric='mlogloss'
    )

# 8. LightGBM (if available)
if LIGHTGBM_AVAILABLE:
    models['LightGBM'] = LGBMClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        verbose=-1
    )

# Enhanced cross-validation with stratification
print("\n📈 Performing stratified cross-validation...")

cv_results = {}
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    print(f"   Training {name}...")
    try:
        cv_scores = cross_val_score(
            model, X_train_scaled, y_train, 
            cv=cv_strategy, scoring='accuracy',
            fit_params={'sample_weight': weights_train} if name in ['Enhanced_RF', 'ExtraTrees'] else {}
        )
        cv_results[name] = {
            'mean': cv_scores.mean(),
            'std': cv_scores.std(),
            'scores': cv_scores
        }
        print(f"      ✅ {name}: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    except Exception as e:
        print(f"      ❌ {name}: Error - {str(e)}")
        cv_results[name] = {'mean': 0, 'std': 0, 'scores': [0]}

# Select best models
print("\n🏆 Model performance ranking:")
sorted_models = sorted(cv_results.items(), key=lambda x: x[1]['mean'], reverse=True)

for i, (name, results) in enumerate(sorted_models[:5], 1):
    print(f"{i}. {name}: {results['mean']:.4f} ± {results['std']:.4f}")

best_model_name = sorted_models[0][0]
print(f"\n🥇 Best model: {best_model_name}")

# Train all models on full training set
print("\n🔧 Training final models...")
trained_models = {}

for name, model in models.items():
    try:
        if name in ['Enhanced_RF', 'ExtraTrees']:
            model.fit(X_train_scaled, y_train, sample_weight=weights_train)
        else:
            model.fit(X_train_scaled, y_train)
        trained_models[name] = model
        print(f"   ✅ {name} trained successfully")
    except Exception as e:
        print(f"   ❌ {name}: Error - {str(e)}")

# Evaluate on test set
print("\n📊 Final evaluation on test set...")

test_results = {}
for name, model in trained_models.items():
    try:
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        test_results[name] = {
            'accuracy': accuracy,
            'predictions': y_pred
        }
        print(f"   {name}: {accuracy:.4f}")
    except Exception as e:
        print(f"   ❌ {name}: Error - {str(e)}")

# Select final best model
best_test_model = max(test_results.items(), key=lambda x: x[1]['accuracy'])
final_model_name = best_test_model[0]
final_model = trained_models[final_model_name]

print(f"\n🎯 Final selected model: {final_model_name}")
print(f"   Test accuracy: {best_test_model[1]['accuracy']:.4f}")

# Create ensemble model with top 3 performers
print("\n🎭 Creating ensemble model...")
top_3_models = sorted_models[:3]
ensemble_models = [(name, trained_models[name]) for name, _ in top_3_models if name in trained_models]

if len(ensemble_models) >= 2:
    ensemble = VotingClassifier(
        estimators=ensemble_models,
        voting='soft'  # Use probability voting
    )
    ensemble.fit(X_train_scaled, y_train)
    ensemble_pred = ensemble.predict(X_test_scaled)
    ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
    
    print(f"   Ensemble accuracy: {ensemble_accuracy:.4f}")
    
    # Use ensemble if it's better
    if ensemble_accuracy > best_test_model[1]['accuracy']:
        final_model = ensemble
        final_model_name = "Ensemble"
        print(f"   🏆 Ensemble model selected as final model!")

# Detailed analysis of final model
print(f"\n📋 Detailed analysis of {final_model_name}...")

final_predictions = final_model.predict(X_test_scaled)
final_accuracy = accuracy_score(y_test, final_predictions)

# Classification report
print("\n📊 Classification Report:")
crop_names = [reverse_crop_dict[i] for i in sorted(reverse_crop_dict.keys())]
print(classification_report(y_test, final_predictions, target_names=crop_names))

# Feature importance (if available)
if hasattr(final_model, 'feature_importances_'):
    print("\n🎯 Feature Importance Analysis:")
    feature_importance = pd.DataFrame({
        'Feature': all_features,
        'Importance': final_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print(feature_importance.head(10))
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    top_features = feature_importance.head(15)
    plt.barh(range(len(top_features)), top_features['Importance'])
    plt.yticks(range(len(top_features)), top_features['Feature'])
    plt.title(f'Top 15 Feature Importance - {final_model_name}')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig('models/plots/final_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()

# Enhanced confusion matrix
plt.figure(figsize=(16, 12))
cm = confusion_matrix(y_test, final_predictions)

# Focus on high-suitability crops for better visualization
high_suitability_crops = [crop for crop, score in southern_africa_crop_suitability.items() if score >= 2]
high_suit_indices = [crop_dict[crop] for crop in high_suitability_crops if crop in crop_dict]
high_suit_labels = [reverse_crop_dict[i] for i in high_suit_indices if i in reverse_crop_dict]

# Create subset confusion matrix for high suitability crops
mask = np.isin(y_test, high_suit_indices) & np.isin(final_predictions, high_suit_indices)
if mask.sum() > 0:
    y_test_subset = y_test[mask]
    pred_subset = final_predictions[mask]
    
    # Remap to sequential indices for visualization
    unique_labels = sorted(np.unique(np.concatenate([y_test_subset, pred_subset])))
    label_map = {old: new for new, old in enumerate(unique_labels)}
    
    y_test_mapped = np.array([label_map[y] for y in y_test_subset])
    pred_mapped = np.array([label_map[y] for y in pred_subset])
    
    cm_subset = confusion_matrix(y_test_mapped, pred_mapped)
    subset_labels = [reverse_crop_dict[i] for i in unique_labels]
    
    sns.heatmap(cm_subset, annot=True, fmt='d', cmap='Blues',
                xticklabels=subset_labels, yticklabels=subset_labels)
    plt.title(f'Confusion Matrix - High Suitability Crops ({final_model_name})')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('models/plots/confusion_matrix_high_suitability.png', dpi=300, bbox_inches='tight')
    plt.show()

# Regional suitability analysis
print("\n🌍 Regional suitability analysis:")
suitability_accuracy = {}
for score in [0, 1, 2, 3]:
    mask = crop.loc[X_test.index, 'suitability_score'] == score
    if mask.sum() > 0:
        score_accuracy = accuracy_score(y_test[mask], final_predictions[mask])
        suitability_accuracy[score] = score_accuracy
        print(f"   Suitability {score}: {score_accuracy:.4f} ({mask.sum()} samples)")

# Save enhanced models and preprocessing pipeline
print("\n💾 Saving enhanced models and metadata...")

# Save the best individual model (for backward compatibility)
pickle.dump(final_model, open('model.pkl', 'wb'))

# Save preprocessing pipeline
pickle.dump(power_transformer, open('power_transformer.pkl', 'wb'))
pickle.dump(robust_scaler, open('robust_scaler.pkl', 'wb'))
pickle.dump(standard_scaler, open('standscaler.pkl', 'wb'))

# Save legacy scalers for compatibility
dummy_minmax = MinMaxScaler()
dummy_minmax.fit(X_train[numerical_features])
pickle.dump(dummy_minmax, open('minmaxscaler.pkl', 'wb'))

# Save all trained models
for name, model in trained_models.items():
    pickle.dump(model, open(f'models/{name.lower()}_model.pkl', 'wb'))

# Save ensemble model if created
if 'ensemble' in locals():
    pickle.dump(ensemble, open('models/ensemble_model.pkl', 'wb'))

# Enhanced metadata
enhanced_metadata = {
    'model_info': {
        'best_model': final_model_name,
        'final_accuracy': final_accuracy,
        'cv_results': cv_results,
        'test_results': test_results,
        'suitability_accuracy': suitability_accuracy
    },
    'preprocessing': {
        'all_features': all_features,
        'base_features': base_features,
        'enhanced_features': enhanced_features,
        'preprocessing_steps': ['PowerTransformer', 'RobustScaler', 'StandardScaler']
    },
    'regional_data': {
        'crop_mapping': crop_dict,
        'reverse_crop_mapping': reverse_crop_dict,
        'suitability_scores': southern_africa_crop_suitability,
        'high_suitability_crops': high_suitability_crops
    },
    'training_info': {
        'training_date': datetime.now().isoformat(),
        'dataset_size': len(crop),
        'feature_count': len(all_features),
        'class_count': len(crop_dict)
    }
}

# Save feature importance if available
if hasattr(final_model, 'feature_importances_'):
    enhanced_metadata['feature_importance'] = feature_importance.to_dict('records')

pickle.dump(enhanced_metadata, open('models/enhanced_metadata.pkl', 'wb'))

# Create model performance summary
summary = f"""
{'='*60}
ENHANCED SOUTHERN AFRICAN CROP RECOMMENDATION MODEL
Training Summary - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*60}

📊 DATASET OVERVIEW:
   - Total samples: {len(crop)}
   - Features used: {len(all_features)} (base: {len(base_features)}, enhanced: {len(enhanced_features)})
   - Crop classes: {len(crop_dict)}
   - Regional focus: Southern Africa

🎯 MODEL PERFORMANCE:
   - Best model: {final_model_name}
   - Test accuracy: {final_accuracy:.4f}
   - Cross-validation: {cv_results.get(final_model_name, {}).get('mean', 0):.4f} ± {cv_results.get(final_model_name, {}).get('std', 0):.4f}

🌍 REGIONAL SUITABILITY:
"""

for score, accuracy in suitability_accuracy.items():
    score_desc = {0: 'Unsuitable', 1: 'Limited', 2: 'Good', 3: 'Excellent'}
    summary += f"   - {score_desc[score]} (Score {score}): {accuracy:.4f}\n"

summary += f"""
🎭 TOP PERFORMING MODELS:
"""

for i, (name, results) in enumerate(sorted_models[:5], 1):
    summary += f"   {i}. {name}: {results['mean']:.4f} ± {results['std']:.4f}\n"

summary += f"""
💾 SAVED ARTIFACTS:
   - model.pkl (best model)
   - Enhanced preprocessing pipeline
   - All trained models in models/ directory
   - Comprehensive metadata
   - Performance visualizations

{'='*60}
"""

print(summary)

# Save summary report
with open('models/training_summary.txt', 'w') as f:
    f.write(summary)

print("✅ Enhanced model training completed successfully!")
print(f"📁 All artifacts saved in models/ directory")
print(f"🎯 Best model ({final_model_name}) saved as model.pkl")
print(f"📊 Model accuracy: {final_accuracy:.4f}")

# Final recommendations
print(f"\n🌱 RECOMMENDATIONS FOR DEPLOYMENT:")
print(f"   1. Use {final_model_name} for crop recommendations")
print(f"   2. Priority crops for Southern Africa: {', '.join(high_suitability_crops[:5])}")
print(f"   3. Consider regional suitability scores in final recommendations")
print(f"   4. Monitor model performance on high-suitability crops")
print(f"   5. Retrain model with more regional data when available")