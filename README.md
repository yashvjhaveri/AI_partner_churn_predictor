# AI_partner_churn_predictor
AI Partner Churn Predictor  This tool uses a Random Forest model to detect at-risk partners. It calculates a Risk Probability and identifies Churn Drivers to suggest actions. A second Lookup Model retrieves full partner profiles by ID. Built with Python and Scikit-Learn for proactive retention.

Partner Churn Predictor & Insights Engine

This repository hosts a dual-model AI framework designed to identify high-risk business partners, quantify churn probability, and deliver automated retention strategies. By synthesizing engagement metrics, financial performance, and support sentiment, the system transforms raw partner data into a proactive lifecycle management tool.

Core Functionalities

The project is structured into two specialized components that bridge the gap between predictive foresight and granular data retrieval:

1. Predictive Risk Model (The "Forecaster")

The primary engine utilizes a Random Forest Classifier to evaluate the likelihood of a partner entering a churn state. Rather than a simple binary output, it generates a Risk Probability Score, allowing teams to prioritize accounts based on the severity of the threat.

Feature Engineering: The model processes 12+ variables, including login frequency, referral trends, payout delays, and commission disputes.

Contextual Output: For every at-risk partner, the system maps the prediction to a specific Churn Driver (e.g., Support Friction or Competitive Pressure) and pulls a Recommended Action (e.g., Fast-track technical tickets) to ensure the intervention is relevant.

2. Partner Insights Model (The "Retriever")

The second component is a high-speed retrieval system optimized for Account Managers. By inputting a unique Partner_ID, the system instantly pulls a comprehensive profile of the partner’s historical behavior and current health metrics. This allows for a deep-dive into the "why" behind the risk scores without manual data mining.

Technical Stack
Machine Learning: Scikit-Learn (Random Forest)
Data Processing: Pandas, NumPy
Model Serialization: Joblib (for .pkl deployment)
Categorical Encoding: Label Encoding for multi-region and tiered partner types.

Business Impact
Traditional churn analysis is often reactive, identifying losses after they occur. This engine focuses on the "At-Risk" window, providing the exact probability and the specific remedy needed to stabilize a partnership.
