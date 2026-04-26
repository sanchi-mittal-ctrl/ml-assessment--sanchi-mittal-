# B1. Problem Formulation

## (a) Machine Learning Problem Formulation

This should be formulated as a **Supervised Machine Learning Regression Problem**.

### Target Variable

The target variable is:

**items_sold**

This represents the total number of items sold by a store during a promotion period.

---

### Candidate Input Features

Possible input features include:

* store_id
* store_size
* location_type (urban / semi-urban / rural)
* promotion_type
* monthly_footfall
* competition_density
* customer demographics
* is_weekend
* is_festival
* month
* season
* previous month sales
* historical promotion performance

---

### Type of ML Problem

This is a **Regression Problem** because the target variable (`items_sold`) is continuous numerical data.

We are predicting the quantity of sales, not assigning categories.

Regression is appropriate because the business wants to maximize sales volume and compare expected outcomes across different promotions.

---

## (b) Why Items Sold is Better than Revenue

Using **items_sold** is more reliable than total sales revenue because revenue can be influenced by factors unrelated to promotion effectiveness, such as:

* product price changes
* premium product launches
* inflation
* store pricing strategy

For example, high revenue may come from selling fewer expensive products rather than a successful promotion.

Items sold directly measures customer purchase response to promotions.

It better reflects promotion success because promotions are designed to increase buying behaviour, not just revenue value.

---

### Broader Principle

This illustrates an important ML principle:

**The target variable must directly represent the business objective.**

Choosing the wrong target leads to misleading models, even if prediction accuracy is high.

Target selection should reflect the actual decision the business wants to improve.

---

## (c) Better Modelling Strategy

Instead of one global model, a better strategy is:

### Segment-Based Modelling

Build separate models for:

* urban stores
* semi-urban stores
* rural stores

or use hierarchical modelling with store clusters.

---

### Justification

Customer behaviour differs significantly across locations.

The same promotion may work differently in urban vs rural stores.

For example:

* BOGO may work better in rural stores
* Loyalty Points may work better in urban stores

Separate models capture these differences more accurately and improve recommendation quality.

This reduces underfitting caused by one overly general model.

---

# B2. Data and EDA Strategy

## (a) Joining the Tables

The four tables are:

* transactions
* store attributes
* promotion details
* calendar table

---

### Joining Strategy

### Step 1

Join `transactions` with `store attributes` using:

**store_id**

### Step 2

Join with `promotion details` using:

**promotion_id**

### Step 3

Join with `calendar table` using:

**transaction_date**

---

### Final Dataset Grain

One row should represent:

**one store for one month under one promotion**

This is called **monthly store-level aggregation**

---

### Aggregations Before Modelling

Aggregate:

* total items sold
* average basket size
* monthly footfall
* total transactions
* average discount impact
* promotion frequency
* festival count
* weekend count

This creates meaningful features for prediction.

---

## (b) EDA Strategy

### 1. Promotion Type vs Items Sold

Use:

**Boxplot**

Purpose:

Check which promotions perform best overall.

Helps decide whether promotion type strongly influences target.

---

### 2. Correlation Heatmap

Use:

**Heatmap**

Purpose:

Identify relationships between:

* footfall
* competition
* store size
* items sold

Helps feature selection and multicollinearity detection.

---

### 3. Monthly Sales Trend

Use:

**Line Chart**

Purpose:

Check seasonality patterns such as:

* festive months
* end-of-year spikes

Helps create time-based features.

---

### 4. Location Type Analysis

Use:

**Grouped Bar Chart**

Purpose:

Compare promotion performance across:

* urban
* semi-urban
* rural

Supports segmented modelling strategy.

---

### How Findings Help

EDA improves:

* feature engineering
* target validation
* promotion strategy understanding
* model selection decisions

It prevents poor modelling assumptions.

---

## (c) No-Promotion Imbalance

Since 80% of transactions have no promotion, the model may learn to prefer “no promotion” and ignore promotion effects.

This creates bias and weak promotion recommendations.

---

### Solution

We should:

* use balanced sampling
* oversample promoted cases
* stratify by promotion presence
* evaluate promotion-specific performance separately

This ensures the model learns true promotion impact.

---

# B3. Model Evaluation and Deployment

## (a) Train-Test Split

Since data is monthly and time-ordered:

Use a **Temporal Split**

Example:

* First 2.5 years → Training
* Last 6 months → Testing

---

### Why Random Split is Wrong

Random split causes data leakage.

Future information may enter training data, creating unrealistic performance.

Real businesses predict future sales using past data only.

Temporal split reflects real deployment conditions.

---

### Evaluation Metrics

### RMSE

Measures large prediction errors strongly.

Useful when big mistakes are costly.

---

### MAE

Measures average prediction error.

Easy for business teams to understand.

Example:

“Prediction is off by 50 units on average.”

---

### Business Interpretation

Lower RMSE and MAE mean better promotion recommendations and less revenue loss from poor decisions.

---

## (b) Explaining Different Recommendations

Feature importance helps explain why Store 12 gets different promotions in different months.

December may have:

* festivals
* high customer traffic
* holiday shopping

So Loyalty Points may drive repeat purchases.

March may have:

* lower seasonal demand
* higher competition

So Flat Discount may create stronger immediate sales.

---

### Communication to Marketing Team

Show:

* top important features
* monthly demand differences
* seasonal impact
* historical promotion performance

This builds trust and helps business users understand model decisions.

---

## (c) Deployment Process

## Step 1: Save the Model

Use:

```python
import joblib
joblib.dump(model, "promotion_model.pkl")
```

This saves the trained pipeline.

---

## Step 2: Monthly Prediction Process

At the start of each month:

* collect latest store data
* update promotion options
* prepare features using same preprocessing pipeline
* load saved model
* generate predictions for all 50 stores

The model recommends the best promotion for each store.

---

## Step 3: Monitoring

Track:

* prediction accuracy
* RMSE / MAE over time
* sales drop after recommendations
* feature drift
* promotion response changes

---

## Step 4: Retraining Trigger

Retrain when:

* performance drops significantly
* customer behaviour changes
* new promotion types are introduced
* seasonal patterns shift

This keeps recommendations accurate and useful.
