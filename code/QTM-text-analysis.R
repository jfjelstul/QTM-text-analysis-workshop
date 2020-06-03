###########################################################################
# Josh Fjelstul, Ph.D.
# Institute for Quantitative Theory and Methods (QTM)
# Text Analysis Workshop
###########################################################################

###########################################################################
# setup
###########################################################################

# set working directory
setwd("~/Desktop/") # Mac
setwd("c:/Desktop/") # PC

# clean workspace
rm(list = ls())

# clear console
cat("\014")

###########################################################################
###########################################################################
# intro: data structures in R
###########################################################################
###########################################################################

# assign values to objects
x <- 9
x <- "text"
x

# objects have classes
x <- 9
class(x)
x <- "text"
class(x)

# vectors
vector <- c(1, 2, 3)
vector <- c("element1", "element2", "element3")

# indexing 
vector[1]
vector[1:2]
index <- 1:2
vector[index]

# length of a vector
length(vector)

# length of unique elements
vector <- c("element1", "element1", "element2")
length(unique(vector))

# lists
list1 <- list(vector1 = c("element1", "element2", "element3"), 
              vector2 = c("element4", "element5", "element6"), 
              vector3 = c("element7", "element8"))

# viewing information in lists
list1
list1[1]
list1[[1]]
list1$vector1

# flaten a list
x <- unlist(list1)
names(x)
names(x) <- NULL

# data frames
# a data frame is really a structured list
var1 <- c(1, 2, 3)
data <- data.frame(var1 = var1, var2 = c(4, 5, 6))

# call a variable
data$var1

# change a variable
data$var1 <- c(7, 8, 9)
data$var1 <- c("element1", "element2", "element3")

# check class
class(data)
class(data$var1)
class(data$var2)

###########################################################################
###########################################################################
# Unit 1: getting text data
###########################################################################
###########################################################################

# clean workspace
rm(list = ls())

# clear console
cat("\014")

###########################################################################
# 1.1: CSV files
###########################################################################

# read in data
data <- read.csv("data/inauguration.csv")

# see variable names
names(data)

# view data
View(data)

# check class
class(data$Event)
# this is a factor
# a factor is a variable that takes a finite number of discrete values

# store text as strings
data <- read.csv("data/inauguration.csv", stringsAsFactors = FALSE)

# check class again
class(data$Event)
# now it's a string

# replace variable names
names(data) <- c("number", "date", "event", "location", "oath", "document", "address", "notes")

# view data again
View(data)

###########################################################################
# 1.2: TXT files
###########################################################################

# install the dplyr package
# install.packages("dplyr")

# load the dplyr package
library(dplyr)

# load a .txt file
text <- readLines("data/emory.txt")
# this warning is ok

# make a data frame
data <- data.frame(id = 1:length(text), text = text, stringsAsFactors = FALSE)

# remove empty lines
data <- filter(data, text != "")

###########################################################################
# 1.3: HTML pages
###########################################################################

# install package
# install.packages("XML")

# load library
library(XML)

# URL address
url <- "https://en.wikipedia.org/wiki/Emory_University"

# new file name
file <- "webpage.html"

# download HTML page to a folder
download.file(url = url, destfile = file)

# read HTML into R
webpage <- readLines(url)
webpage <- readLines(file)

# preview contents
head(webpage) # that's no good

# we can use tools from the XML package to extract the text we're interested in from the HTML code

# the first step is to parse the HTML code
text <- htmlParse(webpage)

# then we can use HTML tags to extract what we're interested in
# check the code to see what the right tags are
text <- xpathSApply(text, "//p", xmlValue)
text <- xpathSApply(text, "//h3", xmlValue)

# we can also download multiple webpages at a time 

# URL addresses
urls <- c("https://en.wikipedia.org/wiki/William_Shakespeare", "https://en.wikipedia.org/wiki/Calculus", "https://en.wikipedia.org/wiki/Biology")

# new file names
files <- c("shakespeare.html", "calculus.html", "biology.html")

# download multiple files
for(i in 1:length(urls)) {
  download.file(url = urls[i], destfile = files[i])
  Sys.sleep(runif(n = 1, min = 1, max = 3))
}

###########################################################################
# 1.4: HTML tables
###########################################################################

# install the XML package
# install.packages("XML")

# load the XML package
library(XML)

# URL address
url <- "https://en.wikipedia.org/wiki/United_States_presidential_inauguration"

# download file to working directory
table <- readLines(url)

# collapse vector of HTML code to a single string
table <- str_c(table, collapse = " ")

# extract tables
table <- readHTMLTable(table)

# get a summary of the object
summary(table)

# keep only the 4th element in the list
table <- table[[4]]

# view the table
View(table)

# drop first row
table <- table[-1, ]

# view variable names
names(table)

# replace variable names
names(table) <- c("number", "date", "event", "location", "oath", "length")

# view table again
View(table)

###########################################################################
# 1.5: The Twitter API
###########################################################################

# install packages
# install.packages("twitteR")
# install.packages("base64enc")

# load package
library(twitteR)

# API information
# consumer_key
# consumer_secret
# access_token
# access_secret

# authenticate
setup_twitter_oauth(consumer_key,
                    consumer_secret,
                    access_token,
                    access_secret)

# check API limits
getCurRateLimitInfo()

# search tweets by user
trump.raw <- userTimeline("@realDonaldTrump", n = 100)

# convert list to a data frame
trump <- twListToDF(trump.raw) # wrapper function

# search tweets by keyword
inauguration <- searchTwitter("#inauguration", n = 100)

# convert list to a data frame
inauguration <- twListToDF(inauguration)

###########################################################################
###########################################################################
# Unit 2: cleaning and manipulating text data
###########################################################################
###########################################################################

# clean workspace
rm(list = ls())

# clear console
cat("\014")

###########################################################################
# 2.1: the stringr package
###########################################################################

# load package
# install.packages("stringr")

# load package
library(stringr)

# R has built-in functions but these are better
# stringr creates wrapers for built-in functions
# note that all functions are vectorized

# sample text
data <- read.csv("data/emory.csv", stringsAsFactors = FALSE)
text <- data$text[1]

# replace single occurence
str_replace(text, "the", "X")

# replace multiple occurences
str_replace_all(text, "the", "X")

# detect expression
str_detect(text, "Emory")

# make a dummy variable
as.numeric(str_detect(text, "Emory"))

# count the number of occurences
str_count(text, "Emory")

###########################################################################
# 2.2: regular expressions
###########################################################################

# a syntax to match patterns in text

# spaces
# \n line break
# \t tab
# \r carriage return (rare)

##################################################
# 2.2.1: metacharacters
##################################################

# ? * + . ^ $ | \ ( ) [ ] { }
# if you want the character, escape the metacharacter using \
# in R, you have to escape again 
# to match a period: \\.

# ? match the preceding item at most once
# * match the preceding item zero or more times
# + match the preceding item one or more times
# . match any single character
# ^ matches the empty character at the beginning of a string
# $ matches the empty character at the end of a string
# | means or

# using ^
# match the first character in the string
str_replace(text, "^.", "")

# using $
# match the last character in the string
str_replace(text, ".$", "")

# using \\
# extract the word vote followed by a period
str_extract_all(text, "Emory.")[[1]] # this isn't what we want
str_extract_all(text, "Emory\\.")[[1]]
# note that str_extract_all returns a list

# using \\
# remove a left bracket
str_replace(text, "[", "") # this doesn't work
str_replace(text, "\\[", "") # this does

# remove left and right brackets
str_replace_all(text, "\\[|\\]", "")

# remove brackets and contents
str_replace_all(text, "\\[.+\\]", "")
str_replace_all(text, "\\[.*?\\]", "")

# this expression is especially useful
# .*?x matches any character any number of times until x occurs
# extract Trump's vote total
str_extract(text, "In 1915.*?University")
str_replace(text, "In 1915.*?University", "")

##################################################
# 2.2.2: sub-expressions
##################################################

# use parentheses to create sub-expressions
# use \\# to copy expressions
# if the regular expression is "example (text)", "\\1" gives you "text"

# example
str_replace(text, "Georgia, (United States)", "\\1")

##################################################
# 2.2.3: character classes
##################################################

# character classes
# a set of characters to match
# list characters inside brackets
# [abc] the lower letters a, b, and c
# [^abc] any characters but the lower letters a, b, and c
# [A-Z] all capital letters 
# [A-Za-z] all letters 
# [0-9] numbers
# note that only ^ - \ are special inside a character class
# if you want - as a character, put it first or last

# extract years
str_extract(text, "[0-9][0-9][0-9][0-9]") # we'll see a better way in a second

# remove all characters that aren't letters or spaces
str_replace_all(text, "[^A-Za-z ]", "")

# remove capitalized words
str_replace_all(text, "[A-Z][a-z]+", "")

##################################################
# 2.2.4: predefined character classes
##################################################

# note that these must go inside another set of brackets
# [:alnum:] alphanumeric characters
# same as [0-9A-Za-z] but also includes diacritics
# [:alpha:] alphabetic characters
# same as [A-Za-z] but also includes diacritics
# [:blank:] blank characters
# [:digit:] numbers
# same as [0-9]
# [:lower:] lower case letters
# [:upper:] upper case letters
# [:print:] printable characters and space
# [:punct:] punctuation characters 
# ! " # $ % & ' ( ) * + , - . / : ; < = > ? @ [ \ ] ^ _ ` { | } ~
# [:space:] space, tab, new line, carriage return, etc.

# the nice thing about [:alnum:] and [:alpha:] is that 
# they'll match characters with diacritic marks

# remove punctuation
str_replace_all(text, "[[:punct:]]", " ") 

# remove numbers
str_replace_all(text, "[[:digit:]]", " ") # leaves a lot of white space 
str_replace_all(text, "[[:space:]]*[[:digit:]]+[[:space:]]*", "") # better 

##################################################
# 2.2.5: repetition quantifiers
##################################################

# use metacharacters to address repetition
# {n} match the preceding item exactly n times
# {n,} match the preceding item n or more times
# {n,m} match the preceding item between n and m times

# extract years
str_extract_all(text, "[0-9]+")[[1]] # not what we want
str_extract_all(text, "[0-9]{4}")[[1]] # use custom character set
str_extract_all(text, "[[:digit:]]{4}")[[1]] # use predefined character set

###########################################################################
# 2.3: using regular expressions to clean text (example 1)
###########################################################################

# make a list of words
list_words <- function(x) {
  
  # collapse vector of paragraphs into a single string
  words <- str_c(x, collapse = " ")
  
  # remove punctuation
  words <- str_replace_all(words, "[[:punct:]]", " ")
  
  # remove numbers
  words <- str_replace_all(words, "[[:digit:]]", " ")
  
  # remove space
  words <- str_replace_all(words, "[[:space:]]+", " ")
  
  # convert to lower case
  words <- tolower(words)
  
  # split string into a vector of words
  words <- str_split(words, " ")[[1]]
  
  # return a vector of words
  return(words)
}

# run function
words <- list_words(text)

# view results
head(words, 15)

# number of words
length(words)

# number of unique words
length(unique(words))

###########################################################################
# 2.4: using regular expressions to clean text (example 2)
###########################################################################

# read in data
trump <- read.csv("data/trump-data.csv", stringsAsFactors = FALSE)

# choose variables
trump <- select(trump, text, created)

# check class of date variable
class(trump$created)

# convert date and time to a string
trump$created <- as.character(trump$created)

# check class again
class(trump$created)

# drop time
trump$created <- str_replace(trump$created, " .*", "")

# year variable
trump$year <- str_extract(trump$created, "^[0-9]+")
trump$year <- as.numeric(trump$year)

# month variable
trump$month <- str_extract(trump$created, "[0-9]+$")
trump$month <- as.numeric(trump$month)

# day variable
trump$day <- str_extract(trump$created, "-[0-9]+-")
trump$day <- str_replace_all(trump$day, "-", "")
trump$day <- as.numeric(trump$day)

# drop created variable
trump$created <- NULL

# remove URL links
trump$text <- str_replace_all(trump$text, "http.*", "")

# remove quotes
trump$text <- str_replace_all(trump$text, "\"", "")

# remove HTML &
trump$text <- str_replace_all(trump$text, "&amp;", " ")

# extract hashtag
trump_hashtags <- str_extract_all(trump$text, "#[[:alnum:]]+")

# remove user names
trump$text <- str_replace_all(trump$text, "@[[:alnum:]]+", " ")

# remove punctuation
trump$text <- str_replace_all(trump$text, "[[:punct:]]+", " ")

# remove other characters
trump$text <- str_replace_all(trump$text, "\\$", "")
trump$text <- str_replace_all(trump$text, "\\+", "")

# remove numbers
trump$text <- str_replace_all(trump$text, "[[:digit:]]+", " ")

# remove spaces
trump$text <- str_replace_all(trump$text, "[[:space:]]+", " ")

# trim white space
trump$text <- str_trim(trump$text)

# to lower case
trump$text <- tolower(trump$text)

# preview results
head(trump$text)

###########################################################################
# 2.5: the tm package
###########################################################################

# install package
# install.packages("tm")
# install.packages("SnowballC")

# load package
library(tm)
library(SnowballC)

# read in data
trump <- read.csv("data/trump-data.csv", stringsAsFactors = FALSE)

# choose variables
trump <- select(trump, text, created)

# make corpus
corpus <- Corpus(VectorSource(trump$text))

# remove punctuation
corpus <- tm_map(corpus, removePunctuation)
# don't worry about this warning

# remove numbers
corpus <- tm_map(corpus, removeNumbers)

# make lower case
corpus <- tm_map(corpus, tolower)

# drop stop words
corpus <- tm_map(corpus, removeWords, stopwords("english"))

# stem words
corpus <- tm_map(corpus, stemDocument, language = "english")

# remove spaces
corpus <- tm_map(corpus, stripWhitespace)

###########################################################################
# 2.6: document term matrices (DTM)
###########################################################################

# make a document term matrix
# this is the imput for most text analysis techniques
dtm <- DocumentTermMatrix(corpus)

# viewing a DTM
dtm_matrix <- as.matrix(dtm)

# see the dimensions
dim(dtm)
# rows are documents
# columns are terms

# remove sparce terms
dtm <- removeSparseTerms(dtm, 0.97)

# check dimensions again
dim(dtm)

# the transpose is a term document matrix
tdm <- TermDocumentMatrix(corpus)

###########################################################################
###########################################################################
# Unit 3: text analysis
###########################################################################
###########################################################################

# clean workspace
rm(list = ls())

# clear console
cat("\014")

# let's make some data to analyze

# vector of file names
files <- c("shakespeare.html", "calculus.html", "biology.html")

# read in data
data <- NULL
for(i in 1:length(files)){
  
  # read in data
  text <- readLines(files[i])
  
  # parse HTML
  text <- htmlParse(text)
  
  # extract paragraphs
  text <- xpathSApply(text, "//p", xmlValue)
  
  # document name
  document <- str_replace(files[i], ".html", "")
  
  # make a data frame
  text <- data.frame(document = document, text = text, stringsAsFactors = FALSE)
  
  # append this new data frame to the bottom of "data"
  data <- rbind(data, text)
}

# clean text
data$text <- str_squish(data$text)

# drop paragraphs with no text
data <- filter(data, text != "")

# make a text corpus
corpus <- Corpus(VectorSource(data$text))
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, tolower)
corpus <- tm_map(corpus, removeWords, stopwords("english"))
corpus <- tm_map(corpus, stripWhitespace)

# make a DTM
dtm <- DocumentTermMatrix(corpus)

# convert to a matrix
dtm_matrix <- as.matrix(dtm)

# make a version with only the most common words
# "sparse" is the maximum allowed sparsity for a term to be kept
# sparsity is the percent of documents a term does NOT appear in
# larger maximum allowed sparsity = more terms retained
dtm_small <- removeSparseTerms(dtm, sparse = 0.90)
# this drops words that appear in fewer than 10% of the documents

# make a version with only the most common words 
dtm_small_matrix <- as.matrix(dtm_small)

###########################################################################
# 3.1: Frequencies and associations
###########################################################################

# view sufficiently frequent terms
findFreqTerms(dtm, lowfreq = 10)
findFreqTerms(dtm, lowfreq = 20)

# word frequencies
frequencies <- colSums(dtm_matrix)
head(frequencies, 25)

# make a data frame with frequent words
summary <- data.frame(word = names(frequencies), frequency = frequencies)   
row.names(summary) <- NULL
summary <- arrange(summary, desc(frequency))

# associated words
findAssocs(dtm, "shakespeare", corlimit = 0.37)
findAssocs(dtm, "biology", corlimit = 0.4)
findAssocs(dtm, "calculus", corlimit = 0.35)

###########################################################################
# 3.2: Cosine similarity
###########################################################################

# imagine we have to vectors originating from the same point
# the angle between them indicates how similar they are
# one vector is the adjacent leg, the other is the hypotenuse
# remember from trig, cosine of angle = length of adjacent / length of hypotenuse
# the vectors are rows from a DTM
# each vector is therefore the same length

a <- as.numeric(dtm_matrix[1,])
b <- as.numeric(dtm_matrix[2,])

# function to calculate cosine similarity
cos_sim <- function(a, b) {
  
  # two ways to write the formula
  # c <- sum(a * b) / sqrt(sum(a ^ 2) * sum(b ^ 2))
  c <- crossprod(a, b) / sqrt(crossprod(a, a) * crossprod(b, b))
  
  # coerce to numeric
  c <- as.numeric(c)
  
  # return the result
  return(c)
}

# run the function
cos_sim(a, b)

###########################################################################
# 3.3: Word clouds
###########################################################################

# install package
# install.packages("wordcloud")

# load package
library(wordcloud)

# frequencies
frequencies <- colSums(dtm_matrix)

# frequency data
summary <- data.frame(word = names(frequencies), frequency = frequencies)   

# based on minimum frequency
wordcloud(words = summary$word, freq = summary$frequency, min.freq = 15, random.order = FALSE)

# based on maximum terms
wordcloud(words = summary$word, freq = summary$frequency, max.words = 100, random.order = FALSE)

###########################################################################
# 3.4: Hierarchical clustering
###########################################################################

# install package
# install.packages("cluster")

# load library
library(cluster)

# suppose each document has coordinates based on word frequences (in n-dimensional space)
# we can calculate the Euclidean distance between them
# this is the sum of the squares of the differences between the coordinates

# hierarchical clustering starts with each document in its own cluster and aggregates up
# 1) merge closest cluster (based on distance)
# 2) recompute distances
# 3) repeate until there is only one cluster

# figure out the "right" number of clusters by looking at a dendrogram

# calculate the distance between words
distance <- dist(t(dtm_small_matrix), method = "euclidian")

# run the clustering algorithm
clusters <- hclust(distance, method = "ward.D")

# make a dendrogram
plot(clusters)

# add borders around the clusters
rect.hclust(clusters, k = 5, border = "blue")

# the number of clusters is arbitrary
plot(clusters)
rect.hclust(clusters, k = 7, border = "blue")

###########################################################################
# 3.5: K-means clustering
###########################################################################

# partitions observations into clusters such that each observation belongs to the cluster with the closest mean
# this means we're minimizing the sum of the within-cluster distances between points in that cluster and the centroid
# we have to decide the "right" number of clusters up front
# there's no theoretically rigorous way of doing that

# calculate the distance between words
distance <- dist(t(dtm_small_matrix), method = "euclidian")

# run k-means algorithm
kmeans <- kmeans(distance, 5)

# make a data frame
summary <- data.frame(names(kmeans$cluster), kmeans$cluster, stringsAsFactors = FALSE)
names(summary) <- c("word", "cluster")
rownames(summary) <- NULL
summary <- arrange(summary, cluster)

# plot clusters
clusplot(as.matrix(distance), kmeans$cluster, color = TRUE, shade = FALSE, labels = 2, lines = 0)   
# this plot reduces the dimensionality to 2 dimensions so we can visualize the clusters (dimensionality reduction)

###########################################################################
# 3.6: Topic modeling
###########################################################################

# latent dirichlet allocation (LDA)
# each topic contains all words across all documents
# each topic gives each word a different probability
# each document is a mix of all topics
# we want to infer the latent topic structure of the documents
# still no good way of choosing the number of clusters

# install package
# install.packages("topicmodels")

# load library
library(topicmodels)

# run a 3 topic model
lda <- LDA(dtm_matrix, k = 3, method = "Gibbs", control = list(burnin = 4000, iter = 2000))

# extract top 5 terms
topics <- terms(lda, 5)
names <-  terms(lda, 1)

# view topics
topics

# probabilities
probs <- as.data.frame(lda@gamma)
names(probs) <- names
summary <- cbind(data[rowSums(dtm_matrix) != 0, ], probs)

# which topic is most likely?
for(i in 1:nrow(probs)) {
  summary$topic[i] <- str_c(names[which(summary[i, names] == max(summary[i, names]))], collapse = ", ")
}

# is the topic correct?
summary$correct <- summary$document == summary$topic

# percent correct
mean(summary$correct)

###########################################################################
###########################################################################
# 4: Applications
###########################################################################
###########################################################################

###########################################################################
# 4.1: Maps
###########################################################################

# install packages
install.packages("rworldextra")
install.packages("rworldmap")
install.packages("countrycode")
install.packages("ggplot2")
install.packages("dplyr")

# load libraries
library(countrycode)
library(rworldmap)
library(rworldxtra)
library(ggplot2)
library(dplyr)

# URL address
url <- "https://en.wikipedia.org/wiki/List_of_countries_by_GDP_(nominal)_per_capita"

# download HTML page
webpage <- readLines(url)

# extract table
table <- readHTMLTable(webpage)
summary(table)
table <- table[[4]]

# rename variables
names(table) <- c("rank", "country", "value")

# drop first row
table <- table[-1, ]

# convert from a factor to a string
table$value <- as.character(table$value)

# the numbers have commas!
# we can use regular expressions to remove them
table$value <- str_replace_all(table$value, ",", "")

# now convert from a string to a number
table$value <- as.numeric(table$value)

# take the log so the color scale has more variation
table$value <- log(table$value)

# convert names to country codes
table$country <- countrycode(table$country, origin = "country.name", destination = "wb")

# map coordinates
world_map <- fortify(spTransform(getMap(resolution = "low"), CRS("+proj=wintri")))
world_map$order <- 1:nrow(world_map)
world_map$country <- countrycode(world_map$id, origin = "country.name", destination = "wb")

# merge GDPPC data with map coordinates
world_map <- left_join(world_map, table, by = "country")

# with legend
plot <- ggplot() + 
  geom_map(data = world_map, map = world_map, mapping = aes(long, lat, map_id = id, fill = value), color = "black", size = 0.2) + 
  scale_fill_gradient(limits = c(min(world_map$value), max(world_map$value)), na.value = "grey80", name = "GDPPC\n") +
  coord_equal() + 
  theme_void()

# view plot
plot

###########################################################################
# 4.2: Network analysis
###########################################################################

# install a package to draw networks
# install.packages("igraph")
# install.packages("ggplot2")

# load libraries
library(igraph)
library(ggplot2)

# make DTM
dtm_network <- as.matrix(removeSparseTerms(dtm, sparse = 0.96))

# adjacency matrix
adjacency <- t(dtm_network) %*% dtm_network

# set zeroes in diagonal
diag(adjacency) <- 0

# make a network object
network <- graph.adjacency(adjacency, weighted = TRUE, mode = "undirected")

# choose network layout
network_layout <- layout.fruchterman.reingold(network)

# data frame of nodes
nodes <- as.data.frame(network_layout, stringsAsFactors = FALSE)
names(nodes) <- c("x", "y")

# add node names
nodes$node <- V(network)$name

# data frame of edges
edges <- as.data.frame(get.edgelist(network), stringsAsFactors = FALSE)
names(edges) <- c("from", "to")

# add edge weights
edges$weight <- E(network)$weight

# merge data together
edges <- left_join(edges, nodes, by = c("from" = "node"))
edges <- left_join(edges, nodes, by = c("to" = "node"))
names(edges) <- c("from", "to", "weight", "from.x", "from.y", "to.x", "to.y")

# network theme
network_theme <- function(base_size = 12, base_family = "Helvetica"){
  require(grid)
  theme_bw(base_size = base_size, base_family = base_family) %+replace%
    theme(rect = element_blank(),
          line = element_blank(),
          text= element_blank())
}

# edge color
edges$color <- "gray85"
edges$color[edges$from == "shakespeare" | edges$to == "shakespeare"] <- "#F8766D"
edges$color[edges$from == "calculus" | edges$to == "calculus"] <- "#00BA38"
edges$color[edges$from == "biology" | edges$to == "biology"] <- "#619CFF"

# edge opacity
edges$alpha <- ifelse(edges$from %in% c("calculus", "biology", "shakespeare") | edges$to %in% c("calculus", "biology", "shakespeare"), 0.7, 0.2)

# node color
nodes$color <- "Black"
nodes$color[nodes$node == "shakespeare"] <- "#F8766D"
nodes$color[nodes$node == "calculus"] <- "#00BA38"
nodes$color[nodes$node == "biology"] <- "#619CFF"

# node size
nodes$size <- ifelse(nodes$node %in% c("calculus", "biology", "shakespeare"), 7, 5)

# make an empty plot object
plot <- ggplot() + 
  geom_segment(data = edges, aes(x = from.x, xend = to.x, y = from.y, yend = to.y), size = 0.5, color = edges$color, alpha = edges$alpha) + 
  geom_point(data = nodes, aes(x = x, y = y), size = 3, color = nodes$color) +
  geom_text(data = nodes, aes(x = x, y = y, label = node), size = nodes$size, vjust = -1, color = nodes$color) + 
  network_theme()

# view plot
plot

###########################################################################
# end R script
###########################################################################
