# Text-Segmentation-for-Summarization
The project consists in improving the performance of a pre-trained summary model, identifying the introductory, development and conclusion sentences in the text.

# Inspiration
<p align="justify">
When it comes to extractive summarization, many pretrained models are available online. However, by using some of them, I noticed how they was focusing a lot only on certain parts of the text, completely neglecting the others. For example, sentences at the very end of the text (which are useful to conclude the summary), were extracted hardly never. The reason this happens is that <b>each text has its own writing style</b>. Let’s consider journal articles and let’s compare them with short stories. Journals articles tend to give all the facts at the very beginning of the text. Very few important concepts are left for the final part. On the other hand, short stories have the most relevant informations at the beginning, in the introduction, (where the story is contextualized, and the protagonists are introduced) and towards the end (where the ending is revealed). As anticipated, these 2 structures are really different. So, the main concept is:
</p>

<p align="center">
“<i>Contents are distributed in different part of the text according to the text genre</i>”
</p>

<p align="justify">
But why summarizer should prefer different part of the text instead of others? Well, it depends on which dataset was used during the training. If newspaper articles were used for the training, clearly the model will tend to exclude information towards the end of the text.
</p>


# The Project
To improve the efficiency of any pretrained extractive summarization model, the idea of the project consists in a text segmentation model, able to identify in the text introduction, body and conclusions. In this way, we can use different summarizer for each part and extract a personalized number of  sentences from each one of them, according to their genre. 



