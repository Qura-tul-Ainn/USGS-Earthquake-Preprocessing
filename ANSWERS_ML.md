\# Day 2–3 Machine Learning Lab Answers



\## 1. Why use a pipeline?



A scikit-learn pipeline combines preprocessing and the estimator into one workflow. During cross-validation, the scaler and imputer are fitted only on each training fold, then applied to that fold's validation data. This prevents information from validation data leaking into preprocessing and producing overly optimistic scores.



\## 2. Gradient descent and learning rate



The loop implementation is generally slower because Python executes each operation separately, while NumPy's vectorized operations run optimized numerical code. With a learning rate of 2.5, gradient descent can overshoot the minimum, causing the loss to increase or diverge rather than converge.



\## 3. Polynomial regression and Ridge



As polynomial degree increases, training error generally decreases, but validation error can increase because the model overfits noise. Ridge regression penalizes large coefficients, helping control model complexity and often improving validation performance.



\## 4. Why compare against a baseline?



A baseline provides a simple reference, such as always predicting the training-set mean for regression or the most frequent class for classification. A model is useful only if its performance improves meaningfully on an appropriate evaluation set.



\## 5. Classification thresholds



The default classification threshold is often 0.5, but it can be adjusted to reflect the cost of different errors. Lowering the threshold generally increases recall while potentially reducing precision. For earthquake classification, the chosen threshold should consider the consequences of missing a large earthquake versus issuing false alarms.



\## 6. Cross-validation and hyperparameter tuning



Cross-validation evaluates a model across multiple training and validation splits, providing a more reliable estimate than a single split. Hyperparameters such as polynomial degree, Ridge alpha, and logistic regression C should be selected using training data and cross-validation, leaving the final test set untouched until evaluation.



\## Hourly ETL scheduling



The ETL command is:



`python -m quakes.etl --mode batch`



\### Cron (Linux/macOS)



Run the job every hour using an absolute path to the project and the correct Python environment:



`0 \* \* \* \* cd /path/to/day1-quake-preprocessing \&\& /path/to/conda/envs/niai-ml/bin/python -m quakes.etl --mode batch >> logs/etl\_cron.log 2>\&1`



Replace both paths with the actual project directory and Python executable on that machine. This example assumes the environment is at `/path/to/conda/envs/niai-ml`.



\### Windows Task Scheduler



1\. Open \*\*Task Scheduler\*\* from the Windows Start menu.

2\. Select \*\*Create Basic Task\*\* and name it `Earthquake ETL Hourly`.

3\. Choose \*\*Daily\*\* as the trigger, then set the task to repeat every \*\*1 hour\*\* for \*\*1 day\*\* in the trigger's advanced settings.

4\. Select \*\*Start a program\*\* as the action.

5\. In \*\*Program/script\*\*, enter the full path to the Python executable in the `niai-ml` Conda environment, for example: `C:\\Users\\YOUR\_USER\\anaconda3\\envs\\niai-ml\\python.exe`.

6\. In \*\*Add arguments\*\*, enter: `-m quakes.etl --mode batch`

7\. In \*\*Start in\*\*, enter the project folder: `D:\\USGS\_Assignment (1)\\day1-quake-preprocessing`.

8\. Finish the wizard, open the task's properties, and verify the trigger repeats every hour. Run the task once manually and inspect `logs\\etl.log` to confirm it works.



Use the actual Python path from your computer. The task should run under a Windows account with access to the project and its data folders.



