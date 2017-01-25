library(jpeg)
library(tiff)
path <- "C:/Users/Zach/Documents/Earth Lab/Deep-learning-class/raw-data/"
files <- list.files(path=path)
numfiles <- length(files)
for (i in 1:numfiles){
  file <- paste0(path, files[i])
  print(file)
  name <- sub("tif", "jpg", file)
  name <- sub("raw", "processed", name)
  tf <- readTIFF(file)
  writeJPEG(tf, target=name)
}
