library(tidymodels)
library(readr)

model <- readRDS("models/xgb_model.rds")
df <- read_csv("data/processed/predict2026.csv")

pred <- predict(model, df, type = "prob") %>%
  bind_cols(df %>% select(ID)) %>%
  rename(prob_apply = .pred_1)

write_csv(pred, "output/2026_predictions.csv")
