# Churn-Analysis-with-Random-Forest-Technique
The Churn Analysis enables businesses to track customer leaving rates. With the goal to helping businesses in enhancing customer loyalty and long-term profitability.
In this project, the client is facing challenges related to customer churn in their utility sector. As competition intensifies and customer expectations evolve, retaining existing customers has become a critical priority for sustainable growth. The primary objective of the analysis was to develop a predictive model for churn using advanced machine learning techniques. By identifying key predictors and patterns contributing to churn, we aimed to equip the client with actionable insights to preemptively address at-risk customer segments and implement targeted retention strategies.

## Key Findings and Recommendations
Through rigorous data exploration, feature engineering, and modeling, a robust predictive model leveraging the Random Forest technique was constructed. The model achieved 90% accuracy and 89% Recall on the dataset, demonstrating its effectiveness in accurately predicting churn propensity. Based on the findings, the following strategies was recommended:
Segmentation and Personalization: Utilize customer segmentation to tailor retention efforts based on distinct behavior and preferences.
Proactive Engagement: Implement proactive outreach campaigns targeting customers identified as high-risk for churn, offering personalized incentives or rewards to incentivize continued engagement.
Enhanced Customer Experience: Focus on improving the overall customer experience through streamlined processes, responsive customer support, and value-added services to foster long-term loyalty.
Continuous Monitoring and Iteration: Establish a framework for ongoing monitoring and evaluation of churn indicators, iteratively refining strategies based on real-time insights and feedback.

## Dependencies
pandas

numpy

seaborn

matplotlib

scikit-learn

## Files
data_ingestion.py: Module to load client and price data from CSV files.
data_preprocessing.py: Module to preprocess and combine client and price data.
data_transformation.py: Module to apply transformations to the preprocessed DataFrame.
model.py: Module containing the RandomForestClassifier model training and evaluation.
requirements.txt: File containing the required Python packages.

## Getting Started
Clone the repository.

Install the dependencies using pip install -r requirements.txt.

Run the scripts using python "script_name.py". Replace "script_name.py" with the actual name of the script.
