# Amazon_data_cds_and_vinyls

Testimonials left by customers and/or rating by stars after a purchase on sales platforms such as Amazon, Cdiscount, EBay are nowadays of great importance for these platforms since these reviews are not only used for marketing purposes but also serve as effective tools to reassure online shoppers. The use of online evaluations has therefore become second nature to consumers. Since 1996, Amazon has implemented its book review system, which has now become widespread and extended to all these products. The platform uses a rating system based on stars ranging from 1 to 5 stars or other acronyms and at cote the number of notices used to establish the calculation is displayed. The number of opinions acts as an audience metric and a signal of confidence: a large number of opinions ensures the credibility of the overall score, at the same time as it attests to the popularity of the product evaluated (Beauvisage et al., 2013). Customers in their search process want to find useful information as quickly as possible. However, searching and comparing text reviews can be frustrating for users, as they feel overwhelmed with information (Ganu et al., 2009). Indeed, the massive number of text-based notices at quantity and its unstructured text format prevent the user from choosing a product with ease. The automatic classification of texts has always been an important subject of research and application since the creation of digital documents. Today, the classification of texts is a necessity because of the very large amount of textual documents that we have to process on a daily basis.

The objective of this project is to predict the score from the review text for "CDs and vinyl" data from the Amazon database. In the same perspective, we will divide the data set into learning data and test data. To do this, we will use automatic learning techniques for classifier and analyze the opinion notes in text form. The implementation of models capable of predicting users' scores based on text revision is of great importance because obtaining an overall idea of a text review could in turn improve the consumer experience.

To do this work, we will move towards supervised automatic learning techniques such as text classification, which automatically classifies a document into a fixed set of classes after being trained on previous annotated data. For these techniques and given the nature of the variable to be predicted, we will consider two approaches. Binary classification and multi-class classification and we will compare the two methods later. In order to assess the relevance of the classifications we will perform, we will use three distinct classifiers that are highly recognized and applied as part of the automatic learning process: Naïve Bayes, Random Forest and Support Vector Machine (SVM). Using metrics such as accuracy, recall and f1-score, we will evaluate the performance of these classifiers.

# Materials and Methods

In this section, we will detail all the methods used to carry out this project. First, the database used for this project will be discussed in a third step. We would define and explain how binary classification and multi-class classification work in a second step. In a third step, we will define and explain the three classifiers and end with a brief explanation of the different metrics.


# A) Project database

We used a file containing consumer reviews for products related to CDs ("CDs and vinyl") sold by Amazon. The data in this file, like so many others, was collected by Julian McAuley, a researcher at Université́ in California, and is available on the following website: http://jmcauley.ucsd.edu/data/amazon/. The file we extracted represents a subset of the data in which all items and users had at least 5 ratings represented by 5 stars. The variables in the database are either quantitative, qualitative or textual. They are listed below.

- reviewerID: the client ID;
- asin(Amazon Standard Identification Number):the identifier of the product under examination; - reviewerName: the name of the examiner ;
- helpful: Note of the usefulness of the opinion;
- reviewText: the text of the review corresponding to the reviewer's comment;
- overall: the product's rating, on five stars;
- summary: the summary of the evaluation text;
- unixReviewTime: time of revision;
- reviewTime: the time of the evaluation in months/days/year.

An extract from the database is presented in Table 1 below. The CDs and Vinyl database consisted of approximately 1,097,592 customer reviews.

# Analysis variables

Predictive variable or response variable: the variable to be predicted in this study is the score
which here corresponds to "overall".

Explanatory variables: the explanatory variables concern the entire reviewText evaluation test, which could be correlated with the score and the summary of the summary evaluation test. We will compare the models from these two tests. In this project, the variables considered necessary and relevant were the overall evaluation score of the opinion, the review text of the evaluation (reviewText), and the summary of the text (see Table 1).

<img width="453" alt="image" src="https://user-images.githubusercontent.com/50669298/58752676-7d6fc480-84b3-11e9-9584-a6914da9f136.png">

![im2](https://user-images.githubusercontent.com/50669298/58752778-10f5c500-84b5-11e9-8c3d-a61a2333023b.png)
