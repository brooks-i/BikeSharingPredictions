**Aim of Project**

In this project, I aim to predict the bike sharing demand in Seoul,
South Korea, based on predictive weather condition and date variables. I
will create a heterogeneous learner from bagged linear regression
models, decision trees, and support vector machines.

**Data Reference**

Seoul Bike Sharing Demand \[Dataset\]. (2020). UCI Machine Learning
Repository. <https://doi.org/10.24432/C5F62R>.

## Data exploration and wrangling

**Loading data**

    url <- "https://drive.google.com/uc?id=11-EkwrPO-ZL_ED6B4urADuC26zrupwNk&export=download"

    df <- read.csv(url, fileEncoding = "Latin1",
                   header = T, stringsAsFactors = T)

**Initial look at data**

    head(df)

*Conclusions*: There are 8760 observations, and 14 features in this
data, including the target variable. The feature names have more
characters than is necessary for the information to come across, I will
rename them. There are 10 numeric features, 3 categorical features, and
1 date feature.

I will rename the columns to that they are easier to read. New column
names:

    ##  [1] "Date"              "Rented.Bike.Count" "Hour"              "Temperature"      
    ##  [5] "Humidity"          "Wind.Speed"        "Visibility"        "Dew.Point"        
    ##  [9] "Solar.Radiation"   "Rainfall"          "Snowfall"          "Season"           
    ## [13] "Holiday"           "Functioning.Day"

I will change date into a date object, so that I can use it for feature
engineering in the future.

    # making date object in year-month-day form
    df$Date <- as.Date(df$Date, "%d/%m/%Y")

### Encoding categorical features

I will perform one-hot encoding for the categorical features, Holiday,
Functioning.Day, and Season.

    df$Holiday <- ifelse((df$Holiday == "Holiday"), 1, 0)
    df$Functioning.Day <- ifelse((df$Functioning.Day == "Yes"), 1, 0)

    df <- dummy_cols(df, select_columns = c("Season"))

    # removing old season feature
    df <- subset(df, select = -c(Season))

Holiday and Functioning.Day are binary features, which makes one-hot
encoding ideal for them. Season only has 4 levels, which also makes
on-hot encoding an appropriate choice for it.

### Feature engineering

**Features I will engineer**

-   muggy: binary that is True when the temperature is above 24 degrees
    C, and humidity is over 65%

-   month: includes information about the month of year

-   weekend: binary feature that is True on Saturdays and Sundays

-   morning: binary feature that is True when the hour is noon or
    earlier

-   is\_rain: binary feature that is true when Rainfall is greater than
    0

-   is\_snow: binary feature that is true when Snowfall is greater than
    0

-   is\_radiation: binary feature that is true when Solar.Radiation is
    greater than 0

Looking at the data with new, engineered features

    head(df, 5)

### Categorizing features

**Numeric features**: Temperature, Humidity, Wind.Speed, Visibility,
Dew.Point, Solar.Radiation, Rainfall, Snowfall

**Categorical features**: Hour, Holiday, Functioning.Day, muggy, month,
weekend, Season\_Summer, Season\_Autumn, Season\_Winter, Season\_Spring,
is\_rain, is\_snow, is\_radiation

**Target feature**: Rented.Bike.Count

### Visualizing data

Visualizing distributions of numeric features

![](Brooks.I.BikeSharingProject_files/figure-markdown_strict/distribution_function-1.png)

-   Wind.Speed, Solar. Radiation, Rainfall, and Snowfall all appear to
    be right skewed, I will handle outliers
-   Visibility appears to be left-skewed, I will handle outliers.
-   Temperature, Humidity, and Dew.Point appear reasonably normally
    distributed.

**Seeing the relationship between month and rented bike count**

![](Brooks.I.BikeSharingProject_files/figure-markdown_strict/month_plot-1.png)

*Conclusions*: The relationship between date and rented bike count
appears to be bimodal. There are peaks in June and September. The rented
bike count drastically decreases in winter months.

**Seeing the relationship between Hour and Rented.Bike.Count**

![](Brooks.I.BikeSharingProject_files/figure-markdown_strict/hour_plot-1.png)

*Conclusions*: It appears that the relationship between Hour and
Rented.Bike.Count is bimodal. There is a peak in the morning around 8
am, and another peak in the evening around 6 pm.

### Missing values

This function counts the number of missing values in a dataset, and
reports them.

    count_missing <- function (x) {
      presence_na <- 0

      # iterate through each row
      for (i in colnames(df)) {
        # find rows with NA values
        na_rows <- which(is.na(df[i]))
        # find num NA's
        num_na <- length(na_rows)
        # report findings
        if (num_na > 0) {
          print(paste0(i, " has ", num_na, " NA values in rows: "))
          print(na_rows)
          presence_na <- presence_na + 1
        }
      } # end of for

      if (presence_na == 0) {
        print("There are no missing values in the dataset.")
      }
      
    } # end of function

    count_missing(0)

    ## [1] "There are no missing values in the dataset."

Since there are no missing values in this dataset, I will randomly
remove some values from Wind.Speed so that I can demonstrate how I would
impute them.

First, I will randomly removing some NA values from Wind.Speed

    set.seed(101010)
    # randomly generating 10 row numbers from df
    random_rows <- sample(nrow(df), 10, replace = F)

    # making 10 rows NA
    for (i in random_rows) {
      df$Wind.Speed[i] <- NA
    }

    # checking it worked
    count_missing(0)

    ## [1] "Wind.Speed has 10 NA values in rows: "
    ##  [1] 3793 5079 6079 6433 6646 6896 8076 8349 8391 8735

There is not a strong correlation between Wind.Speed and ant other
features, so I will not use other features to help guide imputation.

The distribution of Wind.Speed showed earlier demonstrated how it has a
slightly right-skewed distribution. In order to minimize the influence
skew may have on my handling of missing values, I will use
median-imputation.

    med_wind_speed <- median(df$Wind.Speed, na.rm = T)

    for (i in 1:nrow(df)) {
      # replacing missing values with median value
      if(is.na(df$Wind.Speed[i])) {
        df$Wind.Speed[i] <- med_wind_speed
      }
    }

    # checking it worked
    count_missing(0)

    ## [1] "There are no missing values in the dataset."

Making sure the distribution wasn’t heavily affected

    hist(df$Wind.Speed)

![](Brooks.I.BikeSharingProject_files/figure-markdown_strict/unnamed-chunk-3-1.png)

The Wind.Speed distribution seems largely unchanged, I am satisfied with
the quality of my imputation.

### Outliers

**Finding which columns have outliers**

    # keeping track of columns with outliers
    outlier_cols <- c()

    # going through numeric features only
    for (i in colnames(num_df)) {
      # z-score outlier identification
      outliers <- which(abs(scale(num_df[i])) >= 3)
      
      # message if outliers found
      if (length(outliers) > 0) {
        print(paste0(i, " has ", length(outliers), " outliers"))
        outlier_cols[i] <- i
      }
    }

    ## [1] "Wind.Speed has 63 outliers"
    ## [1] "Solar.Radiation has 85 outliers"
    ## [1] "Rainfall has 94 outliers"
    ## [1] "Snowfall has 173 outliers"

*Conclusions*: There are quite a lot of outliers in 4 numeric features.
I will use median-imputation for the outliers, in order to minimize the
impact their skews may have on imputation.

**Median-imputing outliers**

    # looping through columns that were found to have outliers
    for (i in outlier_cols) {
      # getting median value for the column
      median_val <- median(df[[i]])

      # identifying rows with outliers
      outliers <- which(abs(scale(num_df[i])) >= 3)
      
      for (j in outliers) {
        df[j, i] <- median_val
      }
    }

**Seeing distributions after outliers are handled**

    distributions(0)

![](Brooks.I.BikeSharingProject_files/figure-markdown_strict/after_outliers_handled_visual-1.png)

*Conclusions*: Wind.Speed, Solar.Radiation, Rainfall, and Snowfall still
appear quite heavily right-skewed after removing outliers, so I will
perform transformations to make them more reasonably normal. Same with
Visibility, but it is heavily left-skewed instead.

### Transforming data

I will perform a Yeo-Johnson transformation on my heavily skewed data in
an attempt to normalize it.

    transorm_intermed <- preProcess(df[c("Solar.Radiation", "Rainfall", "Visibility", "Snowfall", "Wind.Speed")], method = c("YeoJohnson"))

    df <- predict(transorm_intermed, df)

    distributions(0)

![](Brooks.I.BikeSharingProject_files/figure-markdown_strict/transforming_data-1.png)

*Conclusions*: There are still heavy skews in Solar.Radiation, Rainfall,
Snowfall, and Visibility. I will examine their correlation with the
target variable, and decide what next steps to take

### Decision Tree Data

I will use the data that has been pre-processed to this point for
decision tree models. Decision trees are robust to different data
distributions, so standardization is not necessary.

    tree_df <- df

### Standardization

I will continue to standardize data for use in SVM and linear regression
models, which use distance measurements in their algorithms, and are
sensitive to large differences in data ranges.

    numeric_features <- c("Temperature", "Humidity", "Wind.Speed", "Visibility", "Dew.Point", "Solar.Radiation", "Rainfall", "Snowfall")

    num_df <- df[numeric_features]
    cat_df <- df[, -which(names(df) %in% numeric_features)] 

    # applying z-score standardization to numeric features
    z_df <- data.frame(cat_df, apply(num_df, 2, scale))

    # checking it worked
    distributions(0)

![](Brooks.I.BikeSharingProject_files/figure-markdown_strict/standardization-1.png)

*Conclusions*: Standardizing has not improved the skew of
Solar.Radiation, Rainfall, Snowfall, and Visibility. These skews will
have to be considered when performing feature selection

### Exploring Correlations

I will explore the correlations between features.

*Conclusions*:

-   The target variable, Rented.Bike.Count, has the strongest
    correlations to Temperature, Season\_Winter, Hour, morning,
    Dew.Point, and is\_radiation.
-   There is strong multicollinearity between:
    -   Temperature and Dew.Point
    -   Temperature and Season\_Winter
    -   Temperature and Season\_Summer
    -   Dew.Point and Season\_Winter
    -   Dew.Point and Season\_Summer
-   Temperature has a stronger correlation to Rented.Bike.Count than
    Dew.Point and Season\_Winter, so I will select Temperature as a
    feature over Dew.Point and Season\_Winter.
-   Since the heavily skewed features do not have a strong correlation
    to Rented.Bike.Count, I will leave them as is, since they are not
    selected features.
-   *I will select independent features that have absolute correlations
    greater than 0.20 for modelling: Temperature, Hour, morning,
    Dew.Point, is\_radiation, Functioning.Day, Season\_Summer*

### Principal Component Analysis

Performing PCA on standardized numeric features

    pca_data <- princomp(z_df[numeric_features])
    summary(pca_data)

    ## Importance of components:
    ##                           Comp.1    Comp.2    Comp.3    Comp.4     Comp.5     Comp.6
    ## Standard deviation     1.5669753 1.3738625 1.0434707 0.9401653 0.88634697 0.79552515
    ## Proportion of Variance 0.3069615 0.2359642 0.1361194 0.1105015 0.09821258 0.07911656
    ## Cumulative Proportion  0.3069615 0.5429257 0.6790451 0.7895466 0.88775916 0.96687572
    ##                            Comp.7       Comp.8
    ## Standard deviation     0.51010898 0.0689405409
    ## Proportion of Variance 0.03253011 0.0005941676
    ## Cumulative Proportion  0.99940583 1.0000000000

Looking at breakdowns of the first four principal components

    pca_data$loadings[, 1:4]

    ##                     Comp.1     Comp.2        Comp.3       Comp.4
    ## Temperature      0.4011289  0.5297629  0.0212386413  0.155498530
    ## Humidity         0.5154221 -0.2945692 -0.0928698838  0.110874712
    ## Wind.Speed      -0.2693757  0.2926366 -0.5236747326 -0.060468030
    ## Visibility      -0.3187839  0.2896904  0.3581400801 -0.063568930
    ## Dew.Point        0.5489371  0.3306449  0.0005087084  0.187979898
    ## Solar.Radiation -0.1475023  0.5242850 -0.3348755459 -0.001408583
    ## Rainfall         0.2277763 -0.1390471 -0.4934609343 -0.668166618
    ## Snowfall        -0.1558150 -0.2439029 -0.4824717386  0.688508938

*Conclusions*:

-   1st Principal Component: High positive values for Dew.Point and
    Humidity.

-   2nd Principal Component: High values for Temperature,
    Solar.Radiation, and Visibility

## Modeling

### Splitting data

I will split the data prepared for the linear regression, decision tree,
and SVM algorithms, so that I can evaluate the models using the holdout
method.

    seeds <- c(101010, 1234, 5678)

    lr_svm_train <- list()
    lr_svm_test <- list()
    tree_train <- list()
    tree_test <- list()

    for (i in 1:length(seeds)) {
      # setting a unique seed for each split
      set.seed(seeds[i])
      # randomly splitting 80-20 for lr and svm
      lr_svm_train_index <- createDataPartition(z_df$Rented.Bike.Count, p = 0.8, list = F, times = 1)
      
      # splitting for tree
      set.seed(seeds[i])
      # randomly splitting 80-20 for tree
      tree_train_index <- createDataPartition(tree_df$Rented.Bike.Count, p = 0.8, list = F, times = 1)

      # adding to list of data frames
      lr_svm_train[[i]] <- z_df[lr_svm_train_index,]
      lr_svm_test[[i]] <- z_df[-lr_svm_train_index,]
      tree_train[[i]] <- tree_df[tree_train_index,]
      tree_test[[i]] <- tree_df[-tree_train_index,]
    }

    # checking it worked
    length(lr_svm_train)

    ## [1] 3

    nrow(lr_svm_train[[1]])

    ## [1] 7009

### Evaluation function

I will build a generalized evaluation function that will return RMSE and
R2 statistics in order to evaluate model fit and accuracy.

    model_stats <- function(model, data) {
      # get predictions
      predictions <- predict(model, data)
      
      # calculate RMSE
      sq_err <- (data["Rented.Bike.Count"] - predictions)^2
      MSE <- mean(sq_err[["Rented.Bike.Count"]])
      RMSE <- sqrt(MSE)
      
      # calculate R2
      RSS <- sum(sq_err)
      TSS <- sum((data["Rented.Bike.Count"] - mean(data[["Rented.Bike.Count"]]))^2)
      R2 <- 1 - (RSS / TSS)
      
      # printing message
      print(paste0("The RMSE is: ", round(RMSE, 3)))
      print(paste0("The R2 is: ", round(R2, 3)))
      
      return(list(RMSE, R2))
    }

### Homogeneous ensemble function

I will create a generalized homogeneous ensemble method to use bagging
to get consensus predictions from the models created from different
splits of the data

    homogeneous_ensemble <- function(first_model, second_model, third_model, new_data) {
      final_predictions <- c()
      
      # going through each row in the data
      for (i in 1:nrow(new_data)) {
        
        # making predictions with each model
        model1_prediction <- predict(first_model, new_data[i,])
        model2_prediction <- predict(second_model, new_data[i,])
        model3_prediction <- predict(third_model, new_data[i,])
        
        # averaging predictions
        average_prediction <- (model1_prediction + model2_prediction + 
                                 model3_prediction) / 3
        
        # adding to vector of final prediction values
        final_predictions[i] <- average_prediction
      }
      
      return(final_predictions)
    }

### Linear Regression

#### Linear Regression k-fold cross validation

I will use cross validation for the linear regression model in order to
evaluate fit.

    set.seed(101010)
    control <- trainControl(method = "cv", number = 5)

    lr_cv_model <- train(Rented.Bike.Count ~ Temperature +
                     Hour +
                     morning +
                     is_radiation +
                     Functioning.Day +
                     is_rain +
                     Season_Summer,
                   method = "lm",
                   data = z_df,
                   trControl = control)

    print(lr_cv_model)

    ## Linear Regression 
    ## 
    ## 8760 samples
    ##    7 predictor
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 7007, 7009, 7008, 7008, 7008 
    ## Resampling results:
    ## 
    ##   RMSE      Rsquared   MAE     
    ##   437.0212  0.5411447  329.4325
    ## 
    ## Tuning parameter 'intercept' was held constant at a value of TRUE

*Conclusions*:

-   There are no hyperparameters to tune for linear regression models

-   The RMSE for this model is 437.021, which is fairly high

-   The R squared for this model is 0.541 which is not indicative of a
    very good fit for the model.

-   Perhaps linear regression is not the most optimal modelling
    algorithm to use for this project.

#### Model 1

    lm_model_1 <- lm(Rented.Bike.Count ~ Temperature + Hour + morning + is_radiation + Functioning.Day + is_rain + Season_Summer,
                     data = lr_svm_train[[1]])

    stats_model_1 <- model_stats(lm_model_1, lr_svm_test[[1]])

    ## [1] "The RMSE is: 445.819"
    ## [1] "The R2 is: 0.529"

*Conclusions*: Neither the R2 nor the RMSE are very good for this model.
The statistics generated in k-fold cross validation are better.

#### Model 2

    lm_model_2 <- lm(Rented.Bike.Count ~ Temperature + Hour + morning + is_radiation + Functioning.Day + is_rain + Season_Summer, data = lr_svm_train[[2]])

    stats_model_2 <- model_stats(lm_model_2, lr_svm_test[[2]])

    ## [1] "The RMSE is: 452.85"
    ## [1] "The R2 is: 0.528"

*Conclusions*: The R2 has gotten marginally worse in this model compared
to the first one, but the RMSE has slightly improved

#### Model 3

    lm_model_3 <- lm(Rented.Bike.Count ~ Temperature + Hour + morning + is_radiation + Functioning.Day + is_rain + Season_Summer, data = lr_svm_train[[3]])

    stats_model_3 <- model_stats(lm_model_3, lr_svm_test[[3]])

    ## [1] "The RMSE is: 439.604"
    ## [1] "The R2 is: 0.514"

*Conclusions*: Both the R2 and RMSE are at their highest values with
this model.

#### Linear Regression Ensemble

    # making ensemble predictions
    lr_ensemble_predictions <- homogeneous_ensemble(lm_model_1,
                                                    lm_model_2,
                                                    lm_model_3,
                                                    z_df)

    # evaluating homogeneous ensemble
    lr_ensemble_stats <- sqrt(mean((z_df$Rented.Bike.Count - lr_ensemble_predictions)^2))

    RSS <- sum((z_df$Rented.Bike.Count - lr_ensemble_predictions)^2)
    TSS <- sum((z_df$Rented.Bike.Count - mean(z_df$Rented.Bike.Count))^2)
    lr_ensemble_R2 <- 1 - (RSS / TSS)


    print(paste0("The RMSE is: ", round(lr_ensemble_stats, 3)))

    ## [1] "The RMSE is: 436.871"

    print(paste0("The R2 is: ", round(lr_ensemble_R2, 3)))

    ## [1] "The R2 is: 0.541"

*Conclusions*: The homogeneous ensemble for the linear regression models
has the best RMSE and R2 yet. Bagging improved the performance of linear
regression.

### Decision Tree

#### Decision Tree k-fold cross validation

I will use cross validation for the decision tree model in order to
evaluate fit, and to determine which decision tree algorithm is the most
appropriate for this data.

    set.seed(101010)
    control <- trainControl(method = "cv", number = 5)

    tree_cv <- train(Rented.Bike.Count ~ Temperature +
                     Hour +
                     morning +
                     is_radiation +
                     Functioning.Day +
                     is_rain +
                     Season_Summer,
                   method = "ctree",
                   data = tree_df,
                   trControl = control)

    print(tree_cv)

    ## Conditional Inference Tree 
    ## 
    ## 8760 samples
    ##    7 predictor
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 7007, 7009, 7008, 7008, 7008 
    ## Resampling results across tuning parameters:
    ## 
    ##   mincriterion  RMSE      Rsquared   MAE     
    ##   0.01          298.0947  0.7865509  188.6604
    ##   0.50          305.0577  0.7764174  194.7984
    ##   0.99          314.4031  0.7625914  203.6840
    ## 
    ## RMSE was used to select the optimal model using the smallest value.
    ## The final value used for the model was mincriterion = 0.01.

*Conclusions*: I used a few different methods to determine which
decision tree algorithm would also be the most appropriate for this
data. Settled on the conditional inference tree algorithm, which
performed the best. This model is well-suited to the data, as the target
variable is continuous.

#### Model 1

Building first decision tree model with optimized hyperparameters

    tree_model_1 <- ctree(Rented.Bike.Count ~ Temperature +
                            Hour + morning +
                            is_radiation +
                            Functioning.Day +
                            is_rain +
                            Season_Summer,
                          data = tree_train[[1]],
                          control = ctree_control(mincriterion = 0.01))

    # evaluating with holdout method
    stats_tree_model_1 <- model_stats(tree_model_1, tree_test[[1]])

    ## [1] "The RMSE is: 299.635"
    ## [1] "The R2 is: 0.787"

*Conclusions*: The RMSE and R2 have greatly improved for this model,
compared to the linear regression models. This model is more suitable
for this project.

#### Model 2

Building second decision tree model for bagging with optimized
hyperparameters

    tree_model_2 <- ctree(Rented.Bike.Count ~ Temperature +
                            Hour + morning +
                            is_radiation +
                            Functioning.Day +
                            is_rain +
                            Season_Summer,
                          data = tree_train[[2]],
                          control = ctree_control(mincriterion = 0.01))

    # evaluating with holdout method
    stats_tree_model_2 <- model_stats(tree_model_2, tree_test[[2]])

    ## [1] "The RMSE is: 291.645"
    ## [1] "The R2 is: 0.804"

*Conclusions*: The RMSE and R2 have slightly improved for this model.
This split of data has created a marginally better model.

#### Model 3

Building third decision tree model for bagging with optimized
hyperparameters

    tree_model_3 <- ctree(Rented.Bike.Count ~ Temperature +
                            Hour + morning +
                            is_radiation +
                            Functioning.Day +
                            is_rain +
                            Season_Summer,
                          data = tree_train[[3]],
                          control = ctree_control(mincriterion = 0.01))

    # evaluating with holdout method
    stats_tree_model_3 <- model_stats(tree_model_3, tree_test[[3]])

    ## [1] "The RMSE is: 312.126"
    ## [1] "The R2 is: 0.755"

*Conclusions*: The RMSE and R2 are the lowest they have been for this
model. The R2 indicates that this is still a model that has a pretty
good fit for the data, but the other decision trees are better.

#### Decision Tree Ensemble

    # making homogeneous predictions
    tree_ensemble_predictions <- homogeneous_ensemble(tree_model_1,
                                                      tree_model_2,
                                                      tree_model_3,
                                                      tree_df)

    # evaluating bagged model
    tree_ensemble_stats <- sqrt(mean((tree_df$Rented.Bike.Count - tree_ensemble_predictions)^2))

    RSS <- sum((tree_df$Rented.Bike.Count - tree_ensemble_predictions)^2)
    TSS <- sum((tree_df$Rented.Bike.Count - mean(tree_df$Rented.Bike.Count))^2)
    tree_ensemble_R2 <- 1 - (RSS / TSS)


    print(paste0("The RMSE is: ", round(tree_ensemble_stats, 3)))

    ## [1] "The RMSE is: 267.834"

    print(paste0("The R2 is: ", round(tree_ensemble_R2, 3)))

    ## [1] "The R2 is: 0.828"

*Conclusions*: The RMSE and R2 are the best they have ever been for this
homogeneous learner. Bagging improved this model. The R2 indicates that
this model is an appropriate choice for this project. This model is
better than the linear regression model for this project

### SVM

#### SVM k-fold cross validation

I will use cross validation for the SVM models in order to evaluate fit,
and to determine which SVM kernel is the most appropriate for this data.

    set.seed(101010)
    control <- trainControl(method = "cv", number = 5)

    svm_cv <- train(Rented.Bike.Count ~ Temperature +
                            Hour + morning +
                            is_radiation +
                            Functioning.Day +
                            is_rain +
                            Season_Summer,
                           method = "svmRadial",
                           data = z_df,
                           trControl = control)

    print(svm_cv)

    ## Support Vector Machines with Radial Basis Function Kernel 
    ## 
    ## 8760 samples
    ##    7 predictor
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 7007, 7009, 7008, 7008, 7008 
    ## Resampling results across tuning parameters:
    ## 
    ##   C     RMSE      Rsquared   MAE     
    ##   0.25  310.8927  0.7694946  193.3514
    ##   0.50  308.7418  0.7721447  190.5061
    ##   1.00  307.2021  0.7741325  188.5957
    ## 
    ## Tuning parameter 'sigma' was held constant at a value of 0.3832033
    ## RMSE was used to select the optimal model using the smallest value.
    ## The final values used for the model were sigma = 0.3832033 and C = 1.

*Conclusions*:

-   I looked at SVM using a linear kernel, and using a radial kernel.
    The model performed better when using a radial kernel. This
    indicates that the data on it’s own was not linearly separable. SVM
    with a radial kernel is well-suited to this data when predicting
    Rented.Bike.Count.

-   The R2 and RMSE for this model are comparable to the R2 and RMSE of
    the decision tree models. The R2 indicates that this model is a
    decent fit for the data, and that SVM is an appropriate choice for
    this project

#### Model 1

Building first SVM model for bagging with optimized hyperparameters

    svm_model_1 <- svm(Rented.Bike.Count ~ Temperature +
                            Hour + morning +
                            is_radiation +
                            Functioning.Day +
                            is_rain +
                            Season_Summer,
                       kernel = "radial",
                       cost = 1,
                       data = lr_svm_train[[1]])


    stats_svm_model_1 <- model_stats(svm_model_1, lr_svm_test[[1]])

    ## [1] "The RMSE is: 317.485"
    ## [1] "The R2 is: 0.761"

*Conclusions*: The RMSE and R2 for this model are similar to the
statistics produced by the decision tree models. They are neither the
worst, not the best values seen so far.

#### Model 2

Building second SVM model for bagging with optimized hyperparameters

    svm_model_2 <- svm(Rented.Bike.Count ~ Temperature +
                            Hour + morning +
                            is_radiation +
                            Functioning.Day +
                            is_rain +
                            Season_Summer,
                       kernel = "radial",
                       cost = 1,
                       data = lr_svm_train[[2]])

    stats_svm_model_2 <- model_stats(svm_model_2, lr_svm_test[[2]])

    ## [1] "The RMSE is: 310.361"
    ## [1] "The R2 is: 0.778"

*Conclusions*: The R2 and RMSE have slightly improved for this model
compared to the first SVM model.

#### Model 3

Building third SVM model for bagging with optimized hyperparameters

    svm_model_3 <- svm(Rented.Bike.Count ~ Temperature +
                            Hour + morning +
                            is_radiation +
                            Functioning.Day +
                            is_rain +
                            Season_Summer,
                       kernel = "radial",
                       cost = 1,
                       data = lr_svm_train[[3]])

    stats_svm_model_3 <- model_stats(svm_model_3, lr_svm_test[[3]])

    ## [1] "The RMSE is: 332.261"
    ## [1] "The R2 is: 0.722"

*Conclusions*: The R2 and RMSE are the worst that they have been for the
SVM models so far.

### SVM Ensemble

    svm_ensemble_predictions <- homogeneous_ensemble(svm_model_1, svm_model_2, svm_model_3, z_df)

    # evaluating model
    svm_ensemble_stats <- sqrt(mean((z_df$Rented.Bike.Count - svm_ensemble_predictions)^2))

    RSS <- sum((z_df$Rented.Bike.Count - svm_ensemble_predictions)^2)
    TSS <- sum((z_df$Rented.Bike.Count - mean(z_df$Rented.Bike.Count))^2)
    svm_ensemble_R2 <- 1 - (RSS / TSS)

    print(paste0("The RMSE is: ", round(svm_ensemble_stats, 3)))

    ## [1] "The RMSE is: 312.633"

    print(paste0("The R2 is: ", round(svm_ensemble_R2, 3)))

    ## [1] "The R2 is: 0.765"

*Conclusions*: The RMSE and R2 for this ensemble model are not better
than the first model, which indicates that bagging has not improved this
model’s performance. However, the R2 indicates that this model is still
a good fit for the data, and that this algorithm is appropriate for this
project. This model has a higher RMSE than the decision-tree ensemble,
but it is much better than the linear regression models.

### Heterogeneous Ensemble

#### Functions

This function takes a weighted average of predictions from each of the
linear regression, decision tree, and SVM ensemble models.

    heterogeneous_ensemble <- function(linreg_models, tree_models, svm_models, weights, lin_data, tree_data) {
      T
      # vector of final, consensus predictions
      consensus_predictions <- c()
      
      # weights that each model has an impact on the final prediction
      linreg_weight <- weights[1]
      tree_weight <- weights[2]
      svm_weight <- weights[3]
      
      # defending against wrong data being inputted
      num_linreg <- nrow(lin_data)
      num_tree <- nrow(tree_data)
      if (num_linreg != num_tree) {
        # warning message
        print("Data prepped for linear regression and svm is a different length from data prepped for decision trees")
        # stop ensemble model
        break
      }
      
      final_predictions <- c()

      # going through each row
      for (i in 1:num_linreg) {
        # making prediction with linear regression homogeneous ensemble
        linreg_prediction <- homogeneous_ensemble(linreg_models[[1]],
                                                        linreg_models[[2]],
                                                        linreg_models[[3]],
                                                        lin_data[i,])

        # making prediction with decision tree homogeneous ensemble
        tree_prediction <- homogeneous_ensemble(tree_models[[1]],
                                         tree_models[[2]],
                                         tree_models[[3]],
                                         tree_data[i,])

        # making prediction with SVM homogeneous ensemble
        svm_prediction <- homogeneous_ensemble(svm_models[[1]],
                                       svm_models[[2]],
                                       svm_models[[3]],
                                       lin_data[i,])

        
        # averaging predictions from each homogeneous ensemble, including weights
        avg_prediction <- ((linreg_prediction * linreg_weight) +
                             (tree_prediction * tree_weight) +
                             (svm_prediction * svm_weight)) / sum(weights)

        final_predictions[i] <- avg_prediction
        
      }
        
      return(final_predictions)
    }

This function evaluates the R2 and RMSE of the heterogeneous ensemble

    heterogeneous_stats <- function(model_predictions, data) {
      # calculate RMSE
      sq_err <- (data["Rented.Bike.Count"] - model_predictions)^2
      MSE <- mean(sq_err[["Rented.Bike.Count"]])
      RMSE <- sqrt(MSE)
      
      # Calculate R2
      RSS <- sum(sq_err)
      TSS <- sum((data["Rented.Bike.Count"] - mean(data[["Rented.Bike.Count"]]))^2)
      R2 <- 1 - (RSS / TSS)
      
      # printing message
      print(paste0("The RMSE is: ", round(RMSE, 3)))
      print(paste0("The R2 is: ", round(R2, 3)))
    }

#### Predicting and Evaluating

Setting up inputs for heterogeneous ensemble

    linreg_model_list <- list()
    tree_model_list <- list()
    svm_model_list <- list()
      
    linreg_model_list[[1]] <- lm_model_1
    linreg_model_list[[2]] <- lm_model_2
    linreg_model_list[[3]] <- lm_model_3
      
    tree_model_list[[1]] <- tree_model_1
    tree_model_list[[2]] <- tree_model_2
    tree_model_list[[3]] <- tree_model_3

    svm_model_list[[1]] <- svm_model_1
    svm_model_list[[2]] <- svm_model_2
    svm_model_list[[3]] <- svm_model_3

Predicting with heterogeneous ensemble, and evaluating model

    weights <- c(1, 1, 1)

    heterogeneous_preds_1 <- heterogeneous_ensemble(linreg_model_list,
                                                    tree_model_list,
                                                    svm_model_list,
                                                    weights,
                                                    z_df,
                                                    tree_df)

    heterogeneous_stats(heterogeneous_preds_1, z_df)

    ## [1] "The RMSE is: 307.755"
    ## [1] "The R2 is: 0.772"

*Conclusions*: When all algorithms are weighted equally, the R2
indicates that the model is a good fit for the data. However, previous
models have had higher performance, and I will adjust the weights that
each model has for the final prediction and see if that can improve the
modle.

#### Tuning Heterogenous Ensemble

Changing weights

    weights <- c(1, 2, 3)

    heterogeneous_preds_2 <- heterogeneous_ensemble(linreg_model_list,
                                                    tree_model_list,
                                                    svm_model_list,
                                                    weights,
                                                    z_df,
                                                    tree_df)

    heterogeneous_stats(heterogeneous_preds_2, z_df)

    ## [1] "The RMSE is: 294.717"
    ## [1] "The R2 is: 0.791"

*Conclusions*: Model performance has slightly improved by adjusting the
weights once again. The R2 and RMSE indicate that the model is a good
fit for the data, but previous models have still performed better, and I
want to adjust the

    weights <- c(1, 2, 2)

    heterogeneous_preds_final <- heterogeneous_ensemble(linreg_model_list,
                                                    tree_model_list,
                                                    svm_model_list,
                                                    weights,
                                                    z_df,
                                                    tree_df)

    heterogeneous_stats(heterogeneous_preds_final, z_df)

    ## [1] "The RMSE is: 293.109"
    ## [1] "The R2 is: 0.793"

*Conclusions*: The R2 and RMSE have improved by giving the decision tree
and SVM models higher weights for the final predictions. This makes
sense, as the SVM and decision tree models were evaluated to have better
R2 and RMSE values than.

## Final Model Prediction

The heterogeneous ensemble model that has weights 1, 2, 2 for the linear
regression, decision tree, and SVM models respectively is the best
performing heterogeneous model. The predictions for that model were
saved in the vector heterogeneous\_preds\_final. I will isolate the
prediction for the first observation of the data

    row1_prediction <- heterogeneous_preds_final[1]

    print(paste0("The heterogenous model predicts that the first observation will have the value: ", round(row1_prediction, 3)))

    ## [1] "The heterogenous model predicts that the first observation will have the value: 100.013"

    print(paste0("The actual value of the first observation is: ", z_df$Rented.Bike.Count[1]))

    ## [1] "The actual value of the first observation is: 254"
