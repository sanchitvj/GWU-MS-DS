# EDL dear data data, January 2022
# Spotify top songs 2022
# exported using Sporitfy app

library(tidyverse)
library(janitor)

# read in songs file
my_songs <- readr::read_csv("your_top_songs_2022.csv")

# clean up variable names
my_songs <- janitor::clean_names(my_songs)

# make a table of most common artists
my_songs|>
  group_by(artist_name_s)|>
  summarize(Freq=n())|> 
  arrange(desc(Freq))|> 
  print(n = 20)


