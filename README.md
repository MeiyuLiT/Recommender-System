# Recommender System（Music）

In this project, we recommend the top 100 popular songs to each user. We implement popularity baseline model, alternating least square model, and an extension LightFM to recommend songs for each user accross music listening platforms. Then we use meanAP at 100 items to evaluate the performance.

## Dataset
This project uses the dataset ListenBrainz dataset, which consists of implicit feedback from music listening behavior, spanning several thousand users and tens of millions of songs. 

## Code
- _partition.py_: this uses PySpark and SQL to preprocess dataset. We calculate the number of times (frequency) of listening the song per person as the popularity of the song per person. Then we partition the data into 20% validation set and 80% training set across users (user_id).
- _baseline.py_: we use **baseline popularity model**. The popularity per each recording ($P[i]$) will be calculated by the sum of interactions for all users in the 𝑖th recording divided by the total number of plays for $i$ th recording plus the damping factor $\beta$. The popularity for each recording ($i$) is the formula below:
$$P[i] \rightarrow \dfrac{\sum_u R[u, i]}{|R[:, i]| + \beta}$$
To tune the hyperparameter $\beta$, we select a range of damping factors, $0, 10, 20, 30, 40, 50, 60, 70, 80, 90$. Then we sort the recordings based on popularity values calculated above, and choose the top $100$ most popular recordings as the prediction.
- _convert_string.py_: to prepare for alternating least square model, this converts the song id with from the string format to int format by using window in pyspark.
- _als.py_: this is the **als model** construction. We import ALS model from pyspark.ml.recommendation library. To tune the hyperparameters in the als model, we used grid search to find the best rank, regularization term and $\alpha$.  
- _lightfm_ext.py_: This is an extension model, **lightFM**, in our project. We implement lightFM known for its flexibility and ability to handle sparse data. It uses matrix factorization to make recommendations on the top 100 songs. First we use Dataset.fit and Dataset.build_interactions to transform the dataframe of numeric user ID, numeric recording ID to interaction matrix. Then we train and test the LightFM model on the matrix. 
- _test.py_: this calculates test accuracy using meanAP at 100 with the input of saved model from _als.py_ and _lightfm_ext.py_.
