# Email-Campaign-Analytics
Campaign analytics for departmental stores

Example of Performance boost by Feature engineeering, If an Email campaign subject has a Emoji, it goes to the spam.

[Notebook](https://github.com/Seeker875/Email-Campaign-Analytics/blob/master/EmailCamapaign.ipynb)

### Goal

* Studying factors affecting the success of Email campaigns.
* Benchmarking performance of email campaigns for 1 merchant against the industry.
* Importance of content of the subject of the Email.
* Predicting Read Rate Percent od the email.

### Dataset

* The data set has data from 57 brands. 
* JCPenney has 2nd highest number of rows in the dataset.
* Some important variables in the dataset are Subject of the emails, Read Rate Percent.

### Factors affecting read rate percent

* Whether the email has marketing creative.
* Is the email optimized for mobile devices.
* Subject of the email.
* Personalization of the email.
* Whether the email has link to view the image.


### Content of the subject

* The content of the subject is the most important factor for readability.
* Choice of words affects the readability a lot.
* Marked differences in content of top performers from others.

### Feature Engineering
* HasEmoji is the most important feature.
* Reason: If the subject contains Emojis there are very high chances that the email directly goes to the spam.

### Results
* With the Regression model using XgBoost, The root mean squared error is around 0.08







