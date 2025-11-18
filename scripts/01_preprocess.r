library(tidyverse)
library(lubridate)
library(janitor)
library(readr)

raw <- read_csv("data/raw/funnel_export.csv", show_col_types = FALSE) %>%
  clean_names()    # "Element ID" -> element_id, "Date of Inquiry" -> date_of_inquiry

df <- raw %>%
  mutate(
    # quietly parse ISO timestamps like 2022-10-14T15:14:17+00:00
    date_of_inquiry = lubridate::ymd_hms(date_of_inquiry, tz = "UTC", quiet = TRUE) |> 
      as_date(),

    inquiry_age = as.numeric(Sys.Date() - date_of_inquiry),
    entry_year  = readr::parse_number(active_term_calculated),
    applied     = as.numeric(target_variable)
  ) %>%
  select(
    element_id,
    applied,
    entry_year,
    major,
    active_term_calculated,
    recruitment_source,
    engagement_score,
    inquiry_age
  )


train <- df %>% filter(entry_year <= 2025)
to_predict <- df %>% filter(entry_year == 2026)

write_csv(train, "data/processed/train.csv")
write_csv(to_predict, "data/processed/predict2026.csv")


