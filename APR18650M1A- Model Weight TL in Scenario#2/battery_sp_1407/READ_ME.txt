Step1. Using battery_3_optimization.py to train the model for source data, and the model weight will be saved at the model_check_point folder.

Step2. After saving more than equel to 10 ten models, you can run best_model_finding.py to find the best model for transfering (performance metric is DTW algorithm).

Step3. Using battery_4_inferance.py to find the RUL result for target data.

Step4. Finally, the program of battery_4_prediction_interval.py can help you to find the prediction interval by Mote-Carlo dropout appoarch.