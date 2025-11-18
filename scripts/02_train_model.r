library(tidymodels)
library(readr)
library(xgboost)

train <- read_csv("data/processed/train.csv")

set.seed(123)
spl <- initial_split(train, prop = 0.80)
train_data <- training(spl)
test_data  <- testing(spl)

rec <- recipe(applied ~ ., data = train_data) %>%
  update_role(ID, new_role = "id") %>%
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors())

xgb_mod <- boost_tree(
  trees = 500,
  learn_rate = 0.03,
  tree_depth = 6,
  loss_reduction = 1,
  sample_size = 0.8,
  mtry = tune()
) %>% 
  set_mode("classification") %>%
  set_engine("xgboost")

wf <- workflow() %>%
  add_recipe(rec) %>%
  add_model(xgb_mod)

grid <- grid_regular(mtry(range = c(5, 15)), levels = 5)

cv_folds <- vfold_cv(train_data, v = 5)

tuned <- tune_grid(
  wf,
  resamples = cv_folds,
  grid = grid
)

best <- select_best(tuned, metric = "roc_auc")

final_fit <- finalize_workflow(wf, best) %>%
  fit(train)

saveRDS(final_fit, "models/xgb_model.rds")
